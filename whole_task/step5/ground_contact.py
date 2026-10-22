"""Ground frames and installation repair for feet at membrane contact planes."""
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from whole_task.operations.state import Layout, transform
from whole_task.step3.membranes import MembraneModel


def axial_installations(layout, angles, axis, head):
    result=layout.copy()
    for i,angle in enumerate(angles):
        R=Rotation.from_rotvec(np.asarray(axis)*angle).as_matrix()
        V=np.eye(4);V[:3,:3]=R;V[:3,3]=head-R@head
        result.poses[i]=V@layout.poses[i]
        result.material[i]=layout.material[i]@np.linalg.inv(V)
        result.exits[i]=R@layout.exits[i]
    return result


def normalize_ground(layout, heights):
    result=layout.copy()
    result.poses[:,2,3]-=heights
    result.material[:,:3,3]+=np.asarray(heights)[:,None]*result.material[:,:3,2]
    return result


def resting_heights(mesh,layout):
    normals=layout.material[:,:3,2]
    return (mesh.vertices@normals.T).min(0)-np.einsum('ij,ij->i',normals,layout.material[:,:3,3])


class GroundFrameModel(MembraneModel):
    """Per-task ground coordinates, with one unchanged physical needle point.

    Translation to z=0 changes coordinates of the entire station. Its tool
    point is translated by exactly the same amount, never moved physically.
    """
    def __init__(self,base,heights):
        super().__init__(base.source,base.outer,base.records,base.workspace,base.physical,base.policy,base.name)
        self.ground_heights=np.asarray(heights,dtype=float)
        self.children=[]
        for record,height in zip(self.records,self.ground_heights):
            workspace=deepcopy(base.workspace)
            workspace['reference_head_front_world_m'][2]-=float(height)
            self.children.append(MembraneModel(base.source,base.outer,[record],workspace,
                                               base.physical,base.policy,base.name))

    def prepare(self,layout):
        rows=[]
        for i,child in enumerate(self.children):
            q=Layout(layout.poses[i:i+1],layout.material[i:i+1],layout.exits[i:i+1])
            row=child.prepare(q)[0].copy();row['index']=i
            rows.append(row)
        return rows

    def subset(self,indices):
        base=MembraneModel(self.source,self.outer,[self.records[i] for i in indices],
                           self.workspace,self.physical,self.policy,self.name)
        return GroundFrameModel(base,self.ground_heights[indices])


class FloorObjective:
    """Only propose rotations. All acceptance still uses actual full solids."""
    def __init__(self,model,layout,core):
        self.model=model;self.layout=layout;self.core=core
        self.extreme=core.vertices[ConvexHull(core.vertices).vertices]
        self.rows=[];self.caches=[{} for _ in model.ids]
        body_ids=ConvexHull(model.source.vertices).vertices
        for record,P in zip(model.records,layout.poses):
            data=model.world_data(record,P);D=data['domain'];E=data['endpoints']
            if D.direction_model!='world_needle_direction_cone':
                raise ValueError('Installation repair expects the declared world needle cone')
            F=E['endpoint_need_wrench'][:,:3]
            source=E['source_vertex_ids'];application=D.mesh.vertices[np.maximum(source,0)]
            self.rows.append(dict(F=F,push=-F-D.gravity,gravity=D.gravity,
                application=application,com=D.com,
                body=transform(model.source.vertices[body_ids],P),reference_body=data['body_world'],
                work_ids=D.work_ids))
        self.evaluations=0
        self.allowed_ranges=None

    def force_ranges(self):
        """Screen angle intervals with real contacts and paired load targets."""
        from whole_task.operations.geometry import contact_surface,moved_mesh
        from whole_task.step3.independent_physics import contacts
        from whole_task.step3.legacy import kernel as K
        angles=np.linspace(-np.pi,np.pi,181)
        rotations=Rotation.from_rotvec(angles[:,None]*self.model.axis).as_matrix()
        ranges=[]
        for i,row in enumerate(self.rows):
            support=moved_mesh(self.core,np.linalg.inv(self.layout.material[i]))
            _,_,full,scale=contacts(row['reference_body'],contact_surface(row['reference_body'],support),
                                    row['work_ids'],row['com'])
            ids=np.unique(np.r_[0,np.linspace(0,len(row['F'])-1,33).astype(int)])
            push=np.einsum('nj,ajk->ank',row['push'][ids],rotations)
            gravity=np.einsum('j,ajk->ak',row['gravity'],rotations)
            force=-push-gravity[:,None]
            torque=-np.cross(row['application'][ids]-row['com'],push)
            targets=np.concatenate([force,torque],axis=2)*scale
            mask=K.J.classify(full,targets.reshape(-1,6))[0].reshape(len(angles),len(ids)).all(1)
            intervals=[];start=None
            for k in range(len(mask)+1):
                good=k<len(mask) and mask[k]
                if good and start is None:start=k
                if not good and start is not None:
                    if k-start>=3:
                        low=angles[start]+(0. if start==0 else np.deg2rad(1.))
                        high=angles[k-1]-(0. if k==len(mask) else np.deg2rad(1.))
                        intervals.append((float(low),float(high)))
                    start=None
            if not intervals:raise ValueError('No contact-feasible installation angle interval: '+self.model.ids[i])
            intervals.sort(key=lambda p: (not p[0]<=0<=p[1],p[0]))
            ranges.append(intervals)
            if getattr(self,'log',True):print('CONTACT ANGLE RANGES',self.model.ids[i],np.degrees(intervals).tolist(),flush=True)
        self.allowed_ranges=ranges
        return ranges

    def one(self,i,angle):
        key=round(float(angle),13);cache=self.caches[i]
        if key in cache:return cache[key]
        m=self.model;row=self.rows[i]
        R=Rotation.from_rotvec(m.axis*angle).as_matrix()
        old=self.layout.material[i]
        M=old.copy();M[:3,:3]=old[:3,:3]@R.T
        M[:3,3]=old[:3,3]+old[:3,:3]@(m.head-R.T@m.head)
        n=M[:3,2];b=float((self.extreme@n).min());height=b-n@M[:3,3]
        application=(row['application']-m.head)@R.T+m.head
        com=(row['com']-m.head)@R.T+m.head
        moment=-np.cross(application,row['push'])-np.cross(com,row['gravity'])
        F=row['F']
        xy=np.c_[(height*F[:,0]-moment[:,1])/F[:,2],(moment[:,0]+height*F[:,1])/F[:,2]]
        hull=xy[ConvexHull(xy).vertices]
        material=transform(np.c_[hull,np.full(len(hull),height)],M)
        body=(row['body']-m.head)@R.T+m.head
        value=dict(normal=n,offset=b,pressure=material,height=height,
                   body_gap=float(body[:,2].min()-height),exit_z=float((R@self.layout.exits[i])[2]))
        if len(cache)>96:cache.pop(next(iter(cache)))
        cache[key]=value;return value

    def slacks(self,angles):
        rows=[self.one(i,angle) for i,angle in enumerate(angles)]
        normals=np.array([r['normal'] for r in rows]);offsets=np.array([r['offset'] for r in rows])
        slacks=[]
        for i,row in enumerate(rows):
            gaps=(row['pressure']@normals.T-offsets).min(0)
            different=np.linalg.norm(normals-normals[i],axis=1)>1e-5
            gaps-=different*.00025
            slacks.extend(gaps[np.arange(len(rows))!=i])
            slacks.extend([row['body_gap']-.00001,(row['exit_z']-.05)*.1])
        return np.asarray(slacks)

    def snap_angles(self,angles):
        """Make deliberately shared floor frames exactly identical."""
        result=np.asarray(angles).copy()
        for i in range(len(result)):
            for j in range(i):
                same=np.linalg.norm(self.layout.material[i,:3,:3]-self.layout.material[j,:3,:3])<1e-8
                close=abs(np.angle(np.exp(1j*(result[i]-result[j]))))<.001
                if same and close:result[i]=result[j];break
        return result

    def loss(self,angles):
        self.evaluations+=1
        bad=np.minimum(self.slacks(angles),0.)/.1
        return float(bad@bad)

    def exclude(self,i,angle,width=np.radians(8.)):
        """Remove an actually failed installation angle from the screened interval."""
        if self.allowed_ranges is None:return False
        kept=[]
        for a,b in self.allowed_ranges[i]:
            if angle-width-a>np.radians(2.):kept.append((a,min(b,angle-width)))
            if b-angle-width>np.radians(2.):kept.append((max(a,angle+width),b))
        if not kept:return False
        self.allowed_ranges[i]=kept;self.caches[i].clear()
        return True

    def minimax(self,angles,bounds):
        z=np.r_[angles,self.slacks(angles).min()]
        result=minimize(lambda z:-z[-1],z,jac=lambda z:np.r_[np.zeros(len(angles)),-1.],method='SLSQP',
            constraints=[dict(type='ineq',fun=lambda z:self.slacks(z[:-1])-z[-1])],
            bounds=list(bounds)+[(None,None)],options=dict(maxiter=300,ftol=1e-12))
        return result.x[:-1]

    def proposals(self,attempts=64,seed=547,log=True):
        rng=np.random.default_rng(seed);count=len(self.rows)
        for trial in range(attempts):
            bounds=([r[0] if trial==0 else r[int(rng.integers(len(r)))] for r in self.allowed_ranges]
                    if self.allowed_ranges is not None else [(-np.pi,np.pi)]*count)
            start=(np.clip(np.zeros(count),*[np.array(x) for x in zip(*bounds)]) if trial==0
                   else np.array([rng.uniform(a,b) for a,b in bounds]))
            result=minimize(self.loss,start,method='L-BFGS-B',bounds=bounds,
                options=dict(maxiter=350,ftol=1e-15,gtol=1e-9,eps=2e-6,maxls=40))
            angles=self.snap_angles(result.x)
            gap=float(self.slacks(angles).min())
            if -.03<gap<-1e-9:
                # The squared hinge loss has vanishing gradients just short of
                # feasibility. Maximize the smallest slack directly instead.
                angles=self.snap_angles(self.minimax(angles,bounds))
                gap=float(self.slacks(angles).min())
            self.best_gap=max(getattr(self,'best_gap',-np.inf),gap)
            if log:print('GROUND ANGLES',self.model.ids,'trial',trial,'minimum_mm',gap*1000,
                  'loss',result.fun,flush=True)
            if gap>=-1e-9:
                q=axial_installations(self.layout,angles,self.model.axis,self.model.head)
                heights=resting_heights(self.core,q)
                yield q,heights,dict(trial=trial,angles_rad=angles.tolist(),
                                     minimum_margin_m=gap,objective_evaluations=self.evaluations,
                                     contact_screened_angle_ranges_rad=bounds)
