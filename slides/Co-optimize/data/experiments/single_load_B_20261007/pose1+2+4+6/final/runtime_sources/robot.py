"""Franka FK/IK and exact triangle-solid hand collision screening.

IK uses the official Menagerie Panda joint limits; collision Booleans use
its collision meshes, rather than MuJoCo's convex hull of the object.
"""
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from co_common import S, material_volume

MODEL = Path(__file__).parent/'assets/franka_emika_panda/panda.xml'

class Panda:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(MODEL))
        self.data=mujoco.MjData(self.model)
        self.hand=self.model.body('hand').id
        self.lower=self.model.jnt_range[:7,0];self.upper=self.model.jnt_range[:7,1]
        self.home=np.array([0.,0.,0.,-1.57079,0.,1.57079,-.7853])
        self.geometry=[]
        for g in range(self.model.ngeom):
            if self.model.geom_group[g]!=3:continue
            typ=self.model.geom_type[g]
            if typ==mujoco.mjtGeom.mjGEOM_MESH:
                m=self.model.geom_dataid[g];v=self.model.mesh_vertadr[m];f=self.model.mesh_faceadr[m]
                mesh=trimesh.Trimesh(self.model.mesh_vert[v:v+self.model.mesh_vertnum[m]].copy(),self.model.mesh_face[f:f+self.model.mesh_facenum[m]].copy(),process=True)
            elif typ==mujoco.mjtGeom.mjGEOM_BOX:
                mesh=trimesh.creation.box(2*self.model.geom_size[g])
            else:continue
            if not mesh.is_watertight:mesh=mesh.convex_hull
            body=self.model.body(self.model.geom_bodyid[g]).name
            self.geometry.append((g,body,typ==mujoco.mjtGeom.mjGEOM_BOX,mesh))

    def ik(self,T,start=None):
        best=None
        for seed in [self.home if start is None else start,self.home+np.array([.5,.3,-.5,0,.5,0,0])]:
            q=np.clip(seed,self.lower+1e-5,self.upper-1e-5)
            for _ in range(160):
                self.data.qpos[:7]=q;mujoco.mj_forward(self.model,self.data)
                p=self.data.xpos[self.hand];R=self.data.xmat[self.hand].reshape(3,3)
                err=np.r_[T[:3,3]-p,Rotation.from_matrix(T[:3,:3]@R.T).as_rotvec()]
                if np.linalg.norm(err[:3])<.0005 and np.linalg.norm(err[3:])<.005:
                    return q.copy()
                jp=np.zeros((3,self.model.nv));jr=jp.copy()
                mujoco.mj_jacBody(self.model,self.data,jp,jr,self.hand)
                jac=np.vstack([jp[:,:7],jr[:,:7]])
                dq=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(6),err)
                q=np.clip(q+np.clip(dq,-.15,.15),self.lower+1e-5,self.upper-1e-5)
        return best

    def hand_solids(self,T,width):
        self.data.qpos[:7]=self.home;self.data.qpos[7:9]=width/2
        mujoco.mj_forward(self.model,self.data)
        H=np.eye(4);H[:3,:3]=self.data.xmat[self.hand].reshape(3,3);H[:3,3]=self.data.xpos[self.hand]
        root=T@np.linalg.inv(H)
        result=[]
        for g,body,pad,mesh in self.geometry:
            if body not in ('hand','left_finger','right_finger'):continue
            P=np.eye(4);P[:3,:3]=self.data.geom_xmat[g].reshape(3,3);P[:3,3]=self.data.geom_xpos[g]
            moved=mesh.copy();moved.apply_transform(root@P)
            result.append((body,pad,S.solid(moved)))
        return result

    def clear(self,T,width,obstacle,allow_pads=False):
        for body,pad,solid in self.hand_solids(T,width):
            if allow_pads and pad:continue
            if material_volume(solid^obstacle)>1e-11:return False
        return True

    def approach_clear(self,T,width,obstacle):
        for distance in np.linspace(.08,0,9):
            P=T.copy();P[:3,3]-=distance*T[:3,2]
            if not self.clear(P,.08,obstacle):return False
        for opening in np.linspace(.08,width,7):
            if not self.clear(T,opening,obstacle,allow_pads=True):return False
        return True

    def arm_clear(self,q,width,world_obstacle):
        self.data.qpos[:7]=q;self.data.qpos[7:9]=width/2
        mujoco.mj_forward(self.model,self.data)
        for contact in self.data.contact:
            if contact.dist < -.0001:return False
        for g,body,pad,mesh in self.geometry:
            if pad:continue
            P=np.eye(4);P[:3,:3]=self.data.geom_xmat[g].reshape(3,3);P[:3,3]=self.data.geom_xpos[g]
            moved=mesh.copy();moved.apply_transform(P)
            if moved.vertices[:,2].min() < -1e-6:return False
            if material_volume(S.solid(moved)^world_obstacle)>1e-11:return False
        return True

    def carry_path(self,hand,width,assembly,transforms):
        """Lift, rotate/translate at a common height, lower; sampled checks."""
        from scipy.spatial.transform import Slerp
        q=self.home.copy();records=[]
        station=np.array([.45,0.,.15])
        for index,(first,second) in enumerate(zip(transforms,transforms[1:])):
            high=max(first[2,3],second[2,3])+.25
            A=first.copy();B=second.copy();A[2,3]=B[2,3]=high
            waypoints=[first,A,B,second]
            for stage,(start,end) in enumerate(zip(waypoints,waypoints[1:])):
                rotations=Slerp([0,1],Rotation.from_matrix([start[:3,:3],end[:3,:3]]))
                for fraction in np.linspace(0,1,7):
                    T=np.eye(4);T[:3,:3]=rotations(fraction).as_matrix();T[:3,3]=(1-fraction)*start[:3,3]+fraction*end[:3,3]+station
                    q=self.ik(T@hand,q)
                    if q is None:return dict(passed=False,reason='path_ik',transition=index,stage=stage,fraction=float(fraction))
                    placed=assembly.copy();placed.apply_transform(T)
                    if not self.arm_clear(q,width,S.solid(placed)):
                        return dict(passed=False,reason='path_collision',transition=index,stage=stage,fraction=float(fraction))
                    records.append(dict(T=T.tolist(),q=q.tolist()))
        return dict(passed=True,samples=records,continuous_collision_certificate=False)

    def loading_path(self,hand,width,mesh,fixture,T,direction):
        q=self.home.copy();records=[]
        fixed=fixture.copy();world=T.copy();world[:3,3]+=[.45,0,.15]
        fixed.apply_transform(world)
        for distance in np.linspace(.20,0,13):
            moved=mesh.copy();move=np.eye(4);move[:3,3]=distance*direction
            moved.apply_transform(world@move)
            local=hand.copy();local[:3,3]+=distance*direction
            q=self.ik(world@local,q)
            if q is None:return dict(passed=False,reason='loading_ik',distance=float(distance))
            if not self.arm_clear(q,width,S.solid(moved)+S.solid(fixed)):
                return dict(passed=False,reason='loading_arm_collision',distance=float(distance))
            if distance==.20 and material_volume(S.solid(moved)^S.solid(fixed))>1e-11:
                return dict(passed=False,reason='loading_start_not_outside_fixture')
            records.append(dict(distance=float(distance),q=q.tolist()))
        return dict(passed=True,samples=records,continuous_collision_certificate=False)
