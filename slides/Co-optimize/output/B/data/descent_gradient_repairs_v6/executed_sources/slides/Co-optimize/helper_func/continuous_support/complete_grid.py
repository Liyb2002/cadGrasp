"""Fine quadrature includes working-triangle vertices with positive weights."""
import numpy as np
from .boundary_grid import BoundaryGrid


def position_rule(level):
    if level==1:return np.asarray([[1/3]*3]),np.ones(1)
    # Positive degree-two rule for the SAME uniform triangular area measure.
    # Interior-only nodes can miss the largest torque lever arms.
    return np.vstack([np.eye(3),[1/3]*3]),np.asarray([1/12]*3+[3/4])


class CompleteGrid(BoundaryGrid):
    @classmethod
    def from_task(cls,task,level=1):
        if level==1:return super().from_task(task,level)
        domain=task.domain;bary,position_weights=position_rule(level)
        if level==2:nodes=np.asarray([-1.,0.,1.]);radial=np.asarray([1/6,2/3,1/6])
        else:nodes=np.asarray([-1.,-1/np.sqrt(5),1/np.sqrt(5),1.]);radial=np.asarray([1/12,5/12,5/12,1/12])
        cosine=np.cos(domain.half_angle)+(nodes+1)*(1-np.cos(domain.half_angle))/2
        azimuth=np.arange(4*level)*2*np.pi/(4*level)
        faces=[];uv=[];theta=[];phi=[];density=[]
        for face,area in enumerate(domain.data['geometry']['work_face_areas_m2']):
            for point,pw in zip(bary,position_weights):
                for c,w in zip(cosine,radial):
                    for angle in azimuth:
                        faces.append(face);uv.append(point[1:]);theta.append(np.arccos(c));phi.append(angle)
                        density.append(area*pw*w/len(azimuth))
        uv=np.asarray(uv);values=domain.evaluate(faces,uv[:,0],uv[:,1],theta,phi,magnitude_mg=domain.k)
        keep=values['tool_reachable'];q=values['q_m'][keep];d=values['d'][keep]
        slopes=np.c_[-d,-np.cross(q-domain.com,d)]*task.scale
        slopes=np.c_[slopes,np.zeros(len(slopes))];mass=np.asarray(density)[keep]
        if not len(mass):raise ValueError('No reachable original complete quadrature nodes')
        return cls(np.r_[-domain.gravity,np.zeros(4)],slopes,mass/mass.sum(),domain.k,
            dict(level=level,position_nodes_per_face=len(bary),radial_nodes=len(nodes),azimuth_nodes=4*level,
                 reachable_nodes=len(mass),reachability='original object self-occlusion',
                 measure='original face area x uniform solid angle x uniform magnitude',
                 position_rule='positive vertex-centroid degree-two triangular quadrature',
                 angular_rule='Gauss-Lobatto in cosine',work_cone_boundary_has_positive_objective_weight=True,
                 continuous_domain_certified=False))
