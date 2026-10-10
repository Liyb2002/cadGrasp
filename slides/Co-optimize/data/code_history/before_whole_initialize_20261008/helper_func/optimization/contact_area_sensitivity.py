"""Fast sampled contact-area changes under the union of all pose locks.

Area weights measure changes, never limit or scale unbounded contact reactions.
Only a fixed original wrench is projected during local direction differences.
"""
import time
import numpy as np
from co_common import U, transform_points
from contact_lock_chain import ContactLocks
from physics_guided_geometry import tangent_frames,retract
from physics_guided_cone_iterative import cone_projection


def area_changes(before,after,weights):
    released=~before & after
    killed=before & ~after
    return dict(released_area_m2=float(weights[released].sum()),
                newly_locked_area_m2=float(weights[killed].sum()),
                available_area_m2=float(weights[after].sum()))


class ContactAreaSensitivity:
    def __init__(self,search):
        triangles=search.tri
        bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3],[1/3,1/3,1/3]])
        self.points=np.einsum('av,tvc->tac',bary,triangles).reshape(-1,3)
        sources=np.repeat(search.src,4)
        outward=search.mesh.face_normals[sources]
        areas=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        self.areas=np.repeat(areas/4,4)
        self.locks=ContactLocks(search.mesh,self.points,outward,search.length)
        self.floors=search.floors
        self.rays=[]
        for task,T in search.states:
            points=transform_points(self.points,T)
            normals=-task.domain.mesh.face_normals[sources]
            self.rays.append(U.heads(np.c_[normals,np.cross(points-task.domain.com,normals)],task.scale))

    def state(self,directions,previous=None):
        locks,count,active=self.locks.update(directions,previous)
        return dict(directions=directions.copy(),locks=locks,lock_count=count,active=active)

    def loss(self,state,pose,target):
        rays=np.vstack([self.floors[pose],self.rays[pose][state['active']]])
        value=cone_projection(rays,target)
        if value['kkt_max_violation']>1e-7*max(1.,np.linalg.norm(target)):
            raise RuntimeError('area contact projection KKT unresolved')
        return float(value['loss'])


def area_gradient(search,model,directions,target,h):
    began=time.perf_counter();base=model.state(directions)
    wrench=U.target(search.states[target['pose_index']][0].targets[target['load_index']])
    base_loss=model.loss(base,target['pose_index'],wrench)
    frames=tangent_frames(directions);gradient=np.zeros((len(directions),2));probes=[]
    for pose in range(len(directions)):
        for axis in range(2):
            row=dict(pose_index=pose,axis=axis,step_radians=h,base_proxy_loss=base_loss)
            values={}
            for sign in [-1,1]:
                z=np.zeros_like(gradient);z[pose,axis]=sign*h
                d=retract(directions,frames,z)
                if np.min(np.sum(d*search.normals,axis=1)) < -1e-12:
                    row[str(sign)]=dict(error='outside legal hemisphere');continue
                trial=model.state(d,base)
                value=model.loss(trial,target['pose_index'],wrench)
                values[sign]=value
                row[str(sign)]=dict(loss=value,**area_changes(base['active'],trial['active'],model.areas))
            if len(values)==2:
                gradient[pose,axis]=(values[1]-values[-1])/(2*h);row['formula']='central difference'
            elif 1 in values:
                gradient[pose,axis]=(values[1]-base_loss)/h;row['formula']='forward difference at boundary'
            elif -1 in values:
                gradient[pose,axis]=(base_loss-values[-1])/h;row['formula']='backward difference at boundary'
            else:row['unresolved']=True
            row['derivative']=float(gradient[pose,axis]);probes.append(row)
    print('AREA GRADIENT',np.linalg.norm(gradient),'seconds',time.perf_counter()-began,flush=True)
    return gradient,probes
