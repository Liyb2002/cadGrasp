"""Integrate the original continuous load measure, eliminating force magnitude.

For each work point/direction, b(m)=b0+m*b1 intersects the UNBOUNDED
nonnegative reaction cone in an interval. Primal bases and dual planes batch
bound those intervals. Surface/angular quadrature remains numerical guidance,
never a certificate for the entire continuous domain.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from whole_search.classify import classify


@dataclass
class DemandGrid:
    base: np.ndarray
    slopes: np.ndarray
    weights: np.ndarray
    maximum: float
    metadata: dict

    @classmethod
    def from_task(cls, task, level=1):
        domain=task.domain
        bary=(np.array([[1/3]*3]) if level==1 else
              np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]]))
        nodes,weights=np.polynomial.legendre.leggauss(level)
        cosine=np.cos(domain.half_angle)+(nodes+1)*(1-np.cos(domain.half_angle))/2
        radial=weights/weights.sum();azimuth=np.arange(4*level)*2*np.pi/(4*level)
        faces=[];uv=[];theta=[];phi=[];density=[]
        areas=np.asarray(domain.data['geometry']['work_face_areas_m2'])
        for face,area in enumerate(areas):
            for point in bary:
                for c,w in zip(cosine,radial):
                    for a in azimuth:
                        faces.append(face);uv.append(point[1:]);theta.append(np.arccos(c));phi.append(a)
                        density.append(area*w/len(bary)/len(azimuth))
        uv=np.asarray(uv)
        values=domain.evaluate(faces,uv[:,0],uv[:,1],theta,phi,magnitude_mg=1.*domain.k)
        keep=values['tool_reachable'];q=values['q_m'][keep];d=values['d'][keep]
        scale=task.scale;slopes=np.c_[-d,-np.cross(q-domain.com,d)]*scale
        slopes=np.c_[slopes,np.zeros(len(slopes))]
        base=np.r_[-domain.gravity,np.zeros(4)]
        mass=np.asarray(density)[keep]
        if not len(mass):raise ValueError('No reachable continuous quadrature nodes')
        return cls(base,slopes,mass/mass.sum(),domain.k,
                   dict(level=level,position_nodes_per_face=len(bary),radial_nodes=level,
                        azimuth_nodes=4*level,reachable_nodes=len(mass),
                        reachability='original object self-occlusion',
                        measure='original face area x uniform solid angle x uniform magnitude',
                        magnitude_integrated_by_cone_segment_intersection=True,
                        continuous_domain_certified=False))


def _linear_interval(a,b,maximum,tol=1e-10):
    """a + m*b >= 0, including rank/zero-coefficient cases."""
    n=len(b);lo=np.zeros(n);hi=np.full(n,maximum);valid=np.ones(n,bool)
    for j in range(b.shape[1]):
        positive=b[:,j]>tol;negative=b[:,j]<-tol
        lo[positive]=np.maximum(lo[positive],-a[j]/b[positive,j])
        hi[negative]=np.minimum(hi[negative],-a[j]/b[negative,j])
        valid[(~positive)&(~negative)&(a[j]<-tol)]=False
    valid &= hi>=lo-tol
    return lo,hi,valid


def magnitude_intervals(rays,base,slopes,maximum,tolerance=1e-6):
    """Return numerical feasible intervals for every affine demand segment.

    This is a guidance solver. LP/batch residuals are checked, and uncertainty
    is reported. Final acceptance still uses original strict physical workers.
    """
    rays=np.asarray(rays,float);base=np.asarray(base,float);slopes=np.asarray(slopes,float)
    count,dimension=slopes.shape
    if rays.shape[1]!=dimension:raise ValueError('Reaction and demand dimensions differ')
    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    gravity=classify(rays,base[None])[0][0]
    lo=np.zeros(count);lower=np.zeros(count);upper=np.full(count,maximum)
    endpoint=classify(rays,base+maximum*slopes)[0]
    lower[endpoint]=maximum
    calls=0;planes=[]

    def solve(index,sign):
        nonlocal calls
        matrix=np.column_stack([rays.T,-slopes[index]])
        objective=np.r_[np.zeros(len(rays)),sign]
        result=linprog(objective,A_eq=matrix,b_eq=base,
                       bounds=[(0,None)]*len(rays)+[(0,maximum)],method='highs',options=options)
        calls+=1
        if result.status==2:return None
        if not result.success:raise RuntimeError('Magnitude interval LP unresolved: '+result.message)
        if np.max(np.abs(matrix@result.x-base))>2e-8 or result.x.min()<-1e-9:
            raise RuntimeError('Magnitude interval primal replay failed')
        return result

    if not gravity:
        # General convex interval, including an empty interval or a nonzero
        # lower endpoint. Never assume zero process force is feasible.
        for index in range(count):
            left=solve(index,1);right=solve(index,-1)
            if left is None or right is None:lo[index]=maximum;upper[index]=0.;continue
            lo[index]=left.x[-1];upper[index]=right.x[-1]
        fraction=np.maximum(0,upper-lo)/maximum
        return dict(lower=lo,upper=upper,fraction=fraction,lp_calls=calls,
                    gravity_supported=False,maximum_interval_uncertainty=0.)

    while np.any(upper-lower>tolerance*maximum):
        index=int(np.argmax(upper-lower))
        result=solve(index,-1)
        if result is None:upper[index]=lower[index]=0.;continue
        value=float(np.clip(result.x[-1],0,maximum))
        lower[index]=upper[index]=value
        # One optimal dual plane bounds ALL directions at once.
        y=result.eqlin.marginals
        if np.max(rays@y)<=1e-8:
            denominators=slopes@y;positive=denominators>1e-10
            cap=-base@y/denominators[positive]
            upper[positive]=np.minimum(upper[positive],np.maximum(0,cap)+1e-8)
            planes.append(y)
        # A primal basis certifies affine intervals for all other directions.
        ids=np.flatnonzero(result.x[:-1]>1e-9)
        if len(ids):
            basis=rays[ids];inverse=np.linalg.pinv(basis)
            a=base@inverse;b=slopes@inverse
            left,right,valid=_linear_interval(a,b,maximum)
            valid &= np.max(np.abs(a@basis-base))<=2e-8
            valid &= np.max(np.abs(b@basis-slopes),axis=1)<=2e-8
            lower[valid]=np.maximum(lower[valid],np.maximum(0,right[valid]))
        upper=np.maximum(upper,lower)
    fraction=(lower+upper)/(2*maximum)
    return dict(lower=np.zeros(count),upper=(lower+upper)/2,fraction=np.clip(fraction,0,1),
                lp_calls=calls,dual_planes=len(planes),gravity_supported=True,
                maximum_interval_uncertainty=float(np.max(upper-lower)))


def covered_measure(rays,grid):
    result=magnitude_intervals(rays,grid.base,grid.slopes,grid.maximum)
    result['coverage']=float(grid.weights@result['fraction'])
    result['quadrature']=grid.metadata
    return result
