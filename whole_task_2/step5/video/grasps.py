"""Geometry-derived parallel-jaw grasps for a kinematic presentation."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation,Slerp


def unit(v):return np.asarray(v)/np.linalg.norm(v)

def pose_blend(A,B,u):
    u=float(np.clip(u,0,1))
    P=np.eye(4);P[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([u]).as_matrix()[0]
    P[:3,3]=(1-u)*A[:3,3]+u*B[:3,3];return P

def translated(P,v):
    out=P.copy();out[:3,3]+=v;return out


class GraspPool:
    def __init__(self,mesh,seed=713,samples=380,axis_subdivisions=0,candidate_limit=260,expanded_angles=False,
            minimum_width_m=.008,interior_depths_m=(.002,)):
        self.mesh=mesh
        self.candidate_limit=candidate_limit
        self.angles=[0.,.45,-.45,.9,-.9]
        if expanded_angles:self.angles += [1.35,-1.35,1.8,-1.8,2.25,-2.25]
        rng=np.random.default_rng(seed)
        samples,faces=trimesh.sample.sample_surface(mesh,samples,seed=rng)
        # A fixed 2mm inward seed misses round connectors only 3mm wide.
        # Use the first verified interior seed per sampled surface point.
        remaining=np.ones(len(samples),bool);inside=[]
        for depth in interior_depths_m:
            ids=np.flatnonzero(remaining)
            points=samples[ids]-mesh.face_normals[faces[ids]]*depth
            valid=mesh.contains(points);inside.extend(points[valid]);remaining[ids[valid]]=False
        points=np.asarray(inside).reshape(-1,3)
        if not len(points):raise ValueError('No verified interior grasp seeds')
        axes=trimesh.creation.icosphere(subdivisions=axis_subdivisions).vertices
        axes=axes[axes[:,2]>=0]
        axes=np.vstack([axes,np.eye(3)])
        self.candidates=[]
        for axis in axes:
            origins=np.repeat(points,2,axis=0)
            directions=np.tile([axis,-axis],(len(points),1))
            hits,rays,triangles=mesh.ray.intersects_location(origins,directions,multiple_hits=True)
            nearest={}
            for p,r,f in zip(hits,rays,triangles):
                distance=float(np.linalg.norm(p-origins[r]))
                if r not in nearest or distance<nearest[r][0]:nearest[r]=(distance,p,int(f))
            for i in range(len(points)):
                if 2*i not in nearest or 2*i+1 not in nearest:continue
                a,b=nearest[2*i],nearest[2*i+1]
                width=a[0]+b[0]
                if not minimum_width_m<width<.074:continue
                if abs(mesh.face_normals[a[2]]@axis)<.35 or abs(mesh.face_normals[b[2]]@axis)<.35:continue
                self.candidates.append(dict(center=(a[1]+b[1])/2,axis=axis,width=width,faces=[a[2],b[2]]))
        if not self.candidates:raise ValueError('No native parallel-jaw grasp candidates')

    def choose(self,robot,source,target,work_faces=None,fixture=None,transport_target=None):
        preferred=source[:3,:3].T@[0,0,1]+target[:3,:3].T@[0,0,1]
        if np.linalg.norm(preferred)<.05:preferred=target[:3,:3].T@[1,0,1]
        work=set([] if work_faces is None else work_faces.tolist())
        proposals=[]
        for c in self.candidates:
            if any(f in work for f in c['faces']):continue
            y=unit(c['axis']);normal=preferred-y*(preferred@y)
            if np.linalg.norm(normal)<1e-6:continue
            normal=unit(normal)
            for angle in self.angles:
                n=Rotation.from_rotvec(y*angle).apply(normal)
                z=-n;x=unit(np.cross(y,z));H=np.eye(4);H[:3,:3]=np.column_stack([x,y,z]);H[:3,3]=c['center']-.103*z
                heights=[(P[:3,:3]@c['center']+P[:3,3])[2] for P in [source,target]]
                score=min(heights)-.10*c['width']-.006*abs(angle)
                proposals.append((score,H,c))
        reasons={k:0 for k in ['floor','object','fixture','source_ik','source_approach','transport','target_approach']}
        for _,H,c in sorted(proposals,key=lambda p:-p[0])[:self.candidate_limit]:
            width=c['width']
            local=robot.hand_vertices(H,width)
            if min((local@P[:3,:3].T+P[:3,3])[:,2].min() for P in [source,target])<robot.table_z+.0003:
                reasons['floor']+=1;continue
            # Reject palm/finger penetration; jaw contact itself is allowed.
            # Sparse point screening supports the film, not a swept certificate.
            palm=local[:min(190,len(local))]
            if self.mesh.contains(palm).any():
                reasons['object']+=1;continue
            if fixture is not None:
                world=local@target[:3,:3].T+target[:3,3]
                if fixture.contains(world[::max(1,len(world)//180)]).any():
                    reasons['fixture']+=1;continue
            q=robot.ik(source@H)
            if q is None:
                reasons['source_ik']+=1;continue
            source_hand=source@H
            pre_source=robot.ik(translated(source_hand,-source_hand[:3,2]*.08),q)
            if pre_source is None:
                reasons['source_approach']+=1;continue
            source_q=q.copy();valid=True
            outer=target if transport_target is None else transport_target
            A=translated(source,[0,0,.16]);B=translated(outer,[0,0,.16])
            path=[source,A,B,outer,target]
            durations=[.7,1.8,.7,.85]
            waypoint_joints=[]
            for first,last,duration in zip(path[:-1],path[1:],durations):
                for fraction in np.linspace(0,1,int(np.ceil(duration*12))+1)[1:]:
                    u=float(np.clip(fraction**3*(10-15*fraction+6*fraction*fraction),0,1))
                    q=robot.ik(pose_blend(first,last,u)@H,q,multistart=False)
                    if q is None:valid=False;break
                if not valid:break
                waypoint_joints.append(q.copy())
            target_hand=target@H
            pre_target=robot.ik(translated(target_hand,-target_hand[:3,2]*.08),q) if valid else None
            if not valid:reasons['transport']+=1
            if pre_target is None:
                reasons['target_approach']+=int(valid);valid=False
            if valid:return dict(hand=H,gap=float(width),contact_faces=c['faces'],
                source_pose=source,target_pose=target,q_source=source_q,q_target=q.copy(),
                q_lift_source=waypoint_joints[0],q_lift_transport=waypoint_joints[1],
                q_pre_source=pre_source,q_pre_target=pre_target)
        raise ValueError('No floor-clear, reachable parallel-jaw grasp in the bounded pool: '+str(reasons))
