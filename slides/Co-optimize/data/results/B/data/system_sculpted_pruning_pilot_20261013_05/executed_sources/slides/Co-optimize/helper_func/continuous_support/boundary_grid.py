"""Original work measure with positive-weight angular boundary quadrature."""
import numpy as np
from .coverage import DemandGrid


class BoundaryGrid(DemandGrid):
    @classmethod
    def from_task(cls,task,level=1):
        domain=task.domain
        bary=(np.array([[1/3]*3]) if level==1 else np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]]))
        if level==1:nodes=np.array([-1.,1.]);radial=np.array([.5,.5])
        elif level==2:nodes=np.array([-1.,0.,1.]);radial=np.array([1/6,2/3,1/6])
        else:nodes=np.array([-1.,-1/np.sqrt(5),1/np.sqrt(5),1.]);radial=np.array([1/12,5/12,5/12,1/12])
        cosine=np.cos(domain.half_angle)+(nodes+1)*(1-np.cos(domain.half_angle))/2
        azimuth=np.arange(4*level)*2*np.pi/(4*level)
        faces=[];uv=[];theta=[];phi=[];density=[]
        areas=np.asarray(domain.data['geometry']['work_face_areas_m2'])
        for face,area in enumerate(areas):
            for point in bary:
                for c,w in zip(cosine,radial):
                    for angle in azimuth:
                        faces.append(face);uv.append(point[1:]);theta.append(np.arccos(c));phi.append(angle)
                        density.append(area*w/len(bary)/len(azimuth))
        uv=np.asarray(uv)
        values=domain.evaluate(faces,uv[:,0],uv[:,1],theta,phi,magnitude_mg=domain.k)
        keep=values['tool_reachable'];q=values['q_m'][keep];d=values['d'][keep]
        slopes=np.c_[-d,-np.cross(q-domain.com,d)]*task.scale
        slopes=np.c_[slopes,np.zeros(len(slopes))]
        mass=np.asarray(density)[keep]
        if not len(mass):raise ValueError('No reachable original boundary quadrature nodes')
        return cls(np.r_[-domain.gravity,np.zeros(4)],slopes,mass/mass.sum(),domain.k,
            dict(level=level,position_nodes_per_face=len(bary),radial_nodes=len(nodes),azimuth_nodes=4*level,
                 reachable_nodes=len(mass),reachability='original object self-occlusion',
                 measure='original face area x uniform solid angle x uniform magnitude',
                 angular_rule='Gauss-Lobatto in cosine',work_cone_boundary_has_positive_objective_weight=True,
                 continuous_domain_certified=False))
