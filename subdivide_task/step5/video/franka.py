"""Native Panda kinematics, with the existing collision model and joint limits."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from subdivide_task.step1.common import ROOT
MODEL=ROOT/'slides/Co-optimize/idea/assets/franka_kinematics/panda.xml'


class Franka:
    def __init__(self,base=(1.05,.04,.62),yaw=np.pi):
        self.model=mujoco.MjModel.from_xml_path(str(MODEL))
        self.data=mujoco.MjData(self.model)
        self.model.body_pos[self.model.body('link0').id]=base
        self.model.body_quat[self.model.body('link0').id]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.table_z=float(base[2])
        self.hand=self.model.body('hand').id
        self.lower=self.model.jnt_range[:7,0].copy()
        self.upper=self.model.jnt_range[:7,1].copy()
        self.home=np.array([0.,-.15,0.,-1.65,0.,1.55,-.7853])
        self.parts=[]
        for g in range(self.model.ngeom):
            body=self.model.body(self.model.geom_bodyid[g]).name
            if body not in ('hand','left_finger','right_finger'):continue
            if self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                m=self.model.geom_dataid[g];a=self.model.mesh_vertadr[m]
                vertices=self.model.mesh_vert[a:a+self.model.mesh_vertnum[m]].copy()
            elif self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                vertices=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*self.model.geom_size[g]
            else:continue
            self.parts.append((g,vertices))

    def fk(self,q,gap=.08):
        self.data.qpos[:7]=q
        self.data.qpos[7:9]=np.clip((gap-.003)/2,0,.04)
        mujoco.mj_forward(self.model,self.data)
        T=np.eye(4);T[:3,:3]=self.data.xmat[self.hand].reshape(3,3);T[:3,3]=self.data.xpos[self.hand]
        return T

    def hand_vertices(self,H,gap):
        origin=self.fk(self.home,gap);root=H@np.linalg.inv(origin);cloud=[]
        for g,v in self.parts:
            P=np.eye(4);P[:3,:3]=self.data.geom_xmat[g].reshape(3,3);P[:3,3]=self.data.geom_xpos[g]
            T=root@P;cloud.append(v@T[:3,:3].T+T[:3,3])
        return np.vstack(cloud)

    def ik(self,T,start=None,multistart=True):
        seed=self.home if start is None else np.asarray(start)
        seeds=[seed]
        if multistart:
            seeds += [self.home,self.home+np.array([.7,.35,-.4,-.1,.45,.2,.4]),
                      self.home+np.array([-.7,.25,.5,-.2,-.5,.2,-.3])]
        for initial in seeds:
            q=np.clip(initial,self.lower+1e-6,self.upper-1e-6)
            for _ in range(180):
                P=self.fk(q)
                error=np.r_[T[:3,3]-P[:3,3],Rotation.from_matrix(T[:3,:3]@P[:3,:3].T).as_rotvec()]
                if np.linalg.norm(error[:3])<1e-5 and np.linalg.norm(error[3:])<1e-4:return q.copy()
                jp=np.zeros((3,self.model.nv));jr=jp.copy()
                mujoco.mj_jacBody(self.model,self.data,jp,jr,self.hand)
                J=np.vstack([jp[:,:7],jr[:,:7]])
                damping=.00004 if np.linalg.norm(error)<.02 else .0008
                delta=J.T@np.linalg.solve(J@J.T+damping*np.eye(6),error)
                q=np.clip(q+np.clip(delta,-.13,.13),self.lower+1e-6,self.upper-1e-6)
        return None
