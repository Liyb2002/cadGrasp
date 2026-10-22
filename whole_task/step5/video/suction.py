"""Vacuum-cup grasps for the loader: one small exposed patch, tool along its normal.

A parallel jaw needs two opposed exposed faces and room for both fingers;
inside a fixture that hugs the part, often neither exists. A cup on a 60 mm
rod needs one non-working surface point whose outward normal leaves room for
the rod and the hand. The same tool moves the bunny and the empty fixtures.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .grasps import pose_blend,translated

ROD=.06
REACH=.103+ROD  # Franka hand origin to the cup face along the hand z axis


def unit(v):return np.asarray(v)/np.linalg.norm(v)


class SuctionPool:
    def __init__(self,mesh,seed=713,samples=1500,rolls=8,candidate_limit=2500):
        self.mesh=mesh;self.candidate_limit=candidate_limit
        rng=np.random.default_rng(seed)
        points,faces=trimesh.sample.sample_surface(mesh,samples,seed=rng)
        self.points=points;self.faces=faces;self.normals=mesh.face_normals[faces]
        self.rolls=np.linspace(0,2*np.pi,rolls,endpoint=False)
        self.rod=np.linspace(.004,REACH,14)

    def frames(self,work_faces,source,target):
        work=set([] if work_faces is None else np.asarray(work_faces).tolist());rows=[]
        for p,f,n in zip(self.points,self.faces,self.normals):
            if int(f) in work:continue
            up=min((P[:3,:3]@n)[2] for P in (source,target))
            z=-n;a=np.array([1.,0,0]) if abs(z[0])<.9 else np.array([0,1.,0])
            x0=unit(np.cross(a,z))
            for roll in self.rolls:
                x=Rotation.from_rotvec(z*roll).apply(x0);y=np.cross(z,x)
                H=np.eye(4);H[:3,:3]=np.column_stack([x,y,z]);H[:3,3]=p-REACH*z
                rows.append((up,H,p,int(f)))
        rows.sort(key=lambda r:-r[0])
        return rows[:self.candidate_limit]

    def choose(self,robot,source,target,work_faces=None,fixture=None,transport_target=None):
        reasons={k:0 for k in ['floor','object','fixture','source_ik','source_approach','transport','target_approach']}
        outer=target if transport_target is None else transport_target
        for _,H,p,face in self.frames(work_faces,source,target):
            z=H[:3,2];hand=robot.hand_vertices(H,0.);rod=p[None]-self.rod[:,None]*z
            local=np.vstack([hand,rod])
            if min((local@P[:3,:3].T+P[:3,3])[:,2].min() for P in (source,target,outer))<robot.table_z+.003:
                reasons['floor']+=1;continue
            if self.mesh.contains(local[::max(1,len(local)//200)]).any():
                reasons['object']+=1;continue
            if fixture is not None and any(fixture.contains((local@P[:3,:3].T+P[:3,3])[::max(1,len(local)//200)]).any()
                                           for P in (target,outer)):
                reasons['fixture']+=1;continue
            q=robot.ik(source@H)
            if q is None:reasons['source_ik']+=1;continue
            source_hand=source@H;pre_source=robot.ik(translated(source_hand,-source_hand[:3,2]*.08),q)
            if pre_source is None:reasons['source_approach']+=1;continue
            source_q=q.copy();valid=True;waypoints=[]
            path=[source,translated(source,[0,0,.16]),translated(outer,[0,0,.16]),outer,target]
            for first,last,duration in zip(path[:-1],path[1:],[.7,1.8,.7,.85]):
                for fraction in np.linspace(0,1,int(np.ceil(duration*12))+1)[1:]:
                    u=float(np.clip(fraction**3*(10-15*fraction+6*fraction*fraction),0,1))
                    q=robot.ik(pose_blend(first,last,u)@H,q,multistart=False)
                    if q is None:valid=False;break
                if not valid:break
                waypoints.append(q.copy())
            target_hand=target@H
            pre_target=robot.ik(translated(target_hand,-target_hand[:3,2]*.08),q) if valid else None
            if not valid:reasons['transport']+=1
            if pre_target is None:reasons['target_approach']+=int(valid);valid=False
            if valid:return dict(hand=H,gap=0.,contact_faces=[face],cup_point=p.tolist(),q_waypoints=waypoints,
                source_pose=source,target_pose=target,q_source=source_q,q_target=q.copy(),
                q_lift_source=waypoints[0],q_lift_transport=waypoints[1],
                q_pre_source=pre_source,q_pre_target=pre_target)
        raise ValueError('No reachable vacuum-cup grasp in the bounded pool: '+str(reasons))
