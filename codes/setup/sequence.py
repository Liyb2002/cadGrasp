"""Continuous KUKA-assisted rest -> ten grounded target poses.

The object is free throughout integration. The gripper stays closed at targets;
support design and robot withdrawal belong to the subsequent baseline task.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import trimesh

import grasp as G
import kuka_transfer as K


def seat(mesh, rotation, xy=(0.,0.)):
    T = np.eye(4); T[:3,:3] = rotation
    T[:2,3] = np.asarray(xy)-(rotation@mesh.center_mass)[:2]
    T[2,3] = -(mesh.vertices@rotation.T)[:,2].min()
    return T


def contacts(model,data):
    fingers = set(); floor_force = 0.
    for contact_index, contact in enumerate(data.contact):
        ids = (int(contact.geom1),int(contact.geom2))
        if contact.dist > .0002:
            continue
        names = [model.geom(g).name for g in ids]
        if not any(n.startswith('object_') for n in names):
            continue
        for g in ids:
            name = model.body(model.geom_bodyid[g]).name
            if name in ('left_finger','right_finger'):
                fingers.add(name)
        if 'floor' in names:
            force = np.zeros(6); mujoco.mj_contactForce(model,data,contact_index,force)
            floor_force += max(0.,float(force[0]))
    return len(fingers),floor_force


class Trial:
    def __init__(self,model,mesh,initial,candidate,capture=True,regions=None):
        self.model,self.mesh,self.initial,self.candidate = model,mesh,initial,candidate
        self.floor_vertices=mesh.convex_hull.vertices
        self.data = mujoco.MjData(model)
        self.hand = model.body('hand').id; self.obj = model.body('object').id
        self.fingers = [model.joint(f'finger_joint{i}').qposadr[0] for i in (1,2)]
        self.speeds = [model.joint(f'finger_joint{i}').dofadr[0] for i in (1,2)]
        self.opening = min(model.jnt_range[model.joint('finger_joint1').id,1],candidate.get('opening',np.inf))
        self.regions=regions
        G.object_pose(model,self.data,initial)
        self.data.qpos[self.fingers] = self.opening
        h = candidate['hand'].copy(); h[:3,3] -= .08*h[:3,2]
        G.hand_pose(self.data,h,initialize=True)
        mujoco.mj_forward(model,self.data)
        self.history=[]; self.object_history=[]; self.capture=capture
        self.steps=0; self.offset=np.zeros(6)

    def step(self,h,force=-70.):
        G.hand_pose(self.data,h);self.data.ctrl[0]=force
        mujoco.mj_step(self.model,self.data)
        self.steps+=1
        if self.steps%33==0 and self.capture:
            mujoco.mj_forward(self.model,self.data)
            self.history.append((self.data.qpos.copy(),self.data.mocap_pos.copy(),self.data.mocap_quat.copy()))
            self.object_history.append(G.transform(self.data,self.obj))

    def warmup(self):
        for step in range(5000):
            t=step*.001;h=self.candidate['hand'].copy()
            h[:3,3]-=.08*(1-G.smooth(t/1.5))*h[:3,2]
            h[2,3]+=.025*G.smooth((t-3.)/1.)
            force=(-70*G.smooth((t-1.5)/.7) if t>=1.5 else
                np.clip(3000*(self.opening-np.mean(self.data.qpos[self.fingers]))
                        -20*np.mean(self.data.qvel[self.speeds]),-70,70))
            self.step(h,force)
        mujoco.mj_forward(self.model,self.data)
        obj=G.transform(self.data,self.obj);hand=G.transform(self.data,self.hand)
        self.relative=np.linalg.inv(hand)@obj
        self.relative_inverse=np.linalg.inv(self.relative)
        self.start=obj.copy()
        fingers,_=contacts(self.model,self.data)
        floor_min=float(trimesh.transform_points(self.mesh.vertices,obj)[:,2].min())
        if fingers != 2 or floor_min < .012:
            raise ValueError(f'Pickup failed: fingers={fingers}, clearance={floor_min:.4g}')

    def proposals(self):
        for span in ((6.,33.),(5.,23.),(15.,42.),(25.,52.)):
            for axis in ([.23,1.,.13],[-.23,-1.,.13],[1.,.23,.13],[-1.,-.23,.13]):
                axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
                poses=[]
                for angle in np.linspace(*span,10):
                    R=Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@self.start[:3,:3]
                    pose=seat(self.mesh,R)
                    hand=pose@self.relative_inverse
                    v=trimesh.transform_points(self.mesh.vertices,pose)
                    ids=np.flatnonzero(v[:,2]<1e-8)
                    com=trimesh.transform_points(self.mesh.center_mass[None],pose)[0]
                    if len(ids)!=1 or np.linalg.norm(com[:2]-v[ids[0],:2])<.001:
                        break
                    if self.regions is not None and not self.regions.feasible(pose):break
                    candidate=dict(hand=hand,width=2*np.mean(self.data.qpos[self.fingers]),
                        opening=min(.08,float(np.mean(self.data.qpos[self.fingers]))+.012))
                    if not G.geometry_check(self.model,pose,candidate):
                        break
                    poses.append(pose)
                if len(poses)==10:
                    return poses,dict(axis_world=axis.tolist(),angles_deg=np.linspace(*span,10).tolist())
        raise ValueError('No ten-pose menu with gripper/floor clearance')

    def follow(self,desired):
        mujoco.mj_forward(self.model,self.data)
        actual=G.transform(self.data,self.obj)
        error=np.r_[desired[:3,3]-actual[:3,3],
                    Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec()]
        desired_height=float((self.floor_vertices@desired[2,:3]).min()+desired[2,3])
        actual_height=float((self.floor_vertices@actual[2,:3]).min()+actual[2,3])
        error[2]=desired_height-actual_height
        self.offset += .001*2.*error
        self.offset[:3]=np.clip(self.offset[:3],-.012,.012)
        self.offset[3:]=np.clip(self.offset[3:],-.10,.10)
        corrected=desired.copy()
        corrected[:3,3]+=self.offset[:3]
        corrected[:3,:3]=Rotation.from_rotvec(self.offset[3:]).as_matrix()@desired[:3,:3]
        self.step(corrected@self.relative_inverse)
        if np.linalg.norm(error[:3])>.045 or np.linalg.norm(error[3:])>.5:
            raise ValueError('Lost object tracking during transition')

    def sequence(self,poses):
        rows=[]
        for index,target in enumerate(poses,1):
            source=G.transform(self.data,self.obj)
            # Re-estimate the grasp frame from measured object/tool poses.
            # This updates a controller estimate, never the object's state.
            hand=G.transform(self.data,self.hand)
            self.relative=np.linalg.inv(hand)@source
            self.relative_inverse=np.linalg.inv(self.relative)
            self.offset[:]=0.
            rotations=Slerp([0,1],Rotation.from_matrix(np.array([source[:3,:3],target[:3,:3]])))
            start_height=max(0.,float(trimesh.transform_points(self.mesh.vertices,source)[:,2].min()))
            xy0=trimesh.transform_points(self.mesh.center_mass[None],source)[0,:2]
            begin=self.steps*.001
            for step in range(4000):
                t=step*.001
                fraction=G.smooth((t-.6)/1.4)
                rotation=rotations(fraction).as_matrix()
                desired=seat(self.mesh,rotation,(1-fraction)*xy0)
                if t<.6:
                    height=start_height+(.025-start_height)*G.smooth(t/.6)
                elif t<2.:
                    height=.025
                else:
                    height=.025*(1-G.smooth((t-2.)/.8))-.00015*G.smooth((t-2.)/.8)
                desired[2,3]+=height
                self.follow(desired)
            # Wait for contact to settle instead of declaring a near-contact
            # hover a successful placement at a fixed timer boundary.
            for extra in range(1500):
                mujoco.mj_forward(self.model,self.data)
                fingers,force=contacts(self.model,self.data)
                current=G.transform(self.data,self.obj)
                angle_now=Rotation.from_matrix(current[:3,:3]@target[:3,:3].T).magnitude()
                if fingers==2 and force>.001 and angle_now<np.deg2rad(1.5):break
                if np.linalg.norm(current[:3,3]-target[:3,3])>.005 or angle_now>np.deg2rad(5):break
                desired=target.copy();desired[2,3]-=.00015
                self.follow(desired)
            mujoco.mj_forward(self.model,self.data)
            actual=G.transform(self.data,self.obj)
            vertices=trimesh.transform_points(self.mesh.vertices,actual)
            gap=float(vertices[:,2].min())
            position=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
            angle=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()))
            fingers,force=contacts(self.model,self.data)
            row=dict(pose_id=f'pose_{index}',start_time_s=begin,end_time_s=self.steps*.001,
                target_position_error_m=position,target_rotation_error_deg=angle,
                raw_mesh_floor_gap_m=gap,object_floor_normal_force_N=force,bilateral_grip=fingers==2)
            if position>.003 or angle>3 or not (-.001<gap<.0003) or force<.001 or fingers!=2:
                raise ValueError('Target hold failed: '+json.dumps(row))
            rows.append(row)
        return rows


def find(name,pair_budget=60):
    folder=G.ROOT/'objects'/name
    mesh=trimesh.load(folder/'mesh.stl',force='mesh')
    from sequence_export import WorkRegions
    regions=WorkRegions(mesh)
    saved=json.loads((folder/'poses.json').read_text())
    if saved.get('schema')=='cadgrasp_sequence_v1':
        placements=[dict(index=0,T_world_mesh=saved['rest']['T_world_mesh'])]
    else:
        placements=saved['poses']
    if name=='B':placements=sorted(placements,key=lambda p:p['index']!=4)
    model=G.build(name,exact=name=='B',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
    for place in placements:
        initial=seat(mesh,np.asarray(place['T_world_mesh'])[:3,:3])
        # Saved settled orientations are the preparation input; no pose teleport
        # occurs after the free object's initial state is set.
        depth = (.112,.118,.10) if mesh.extents.max()<.10 else (.10,.08,.112)
        trials=0
        for index,candidate in enumerate(G.candidates(mesh,initial,pair_budget,max_width=.15,depths=depth)):
            if not G.geometry_check(model,initial,candidate):continue
            trials+=1
            try:
                trial=Trial(model,mesh,initial,candidate,regions=regions)
                trial.warmup()
                poses,rule=trial.proposals()
                checks=trial.sequence(poses)
                arm=K.check(trial.history)
                trial.arm_frames,trial.arm_q,arm['scene_checks']=K.check_scene(model,trial.history,name)
            except ValueError as error:
                print(json.dumps(dict(object=name,placement=place['index'],candidate=index,
                    passed=False,reason=str(error))),flush=True)
                continue
            print(json.dumps(dict(object=name,placement=place['index'],candidate=index,
                passed=True,targets=len(poses),checks=checks,arm=arm)),flush=True)
            return model,mesh,initial,candidate,trial,poses,rule,checks,arm
        print(json.dumps(dict(object=name,placement=place['index'],trials=trials,status='menu exhausted')),flush=True)
    raise RuntimeError(f'{name}: no verified continuous sequence in the candidate menu')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--pairs',type=int,default=60)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--replay',action='store_true',help='Render the already exported trajectory without rerunning search')
    args=parser.parse_args()
    if args.replay:
        manifest=json.loads((G.ROOT/'objects'/args.object/'poses.json').read_text())
        if manifest.get('schema') == 'cadgrasp_pose_set_v1':
            parser.error('These are target-only poses; no robot trajectory has been generated for this revision')
        if args.video is None:args.video=G.ROOT/'objects'/args.object/'video.mp4'
        with np.load(G.ROOT/'objects'/args.object/'trajectory.npz') as z:
            history=list(zip(z['qpos'],z['mocap_pos'],z['mocap_quat']))
            robot_track=(z['robot_frame_indices'].copy(),z['robot_q'].copy())
        hand_xml=G.ROOT/manifest['grasp']['model']
        model=G.build(args.object,hand_xml=hand_xml,floor_hull=True)
        rendered_q=K.render(model,history,args.video,args.object,'parallel',robot_track=robot_track,hand_xml=hand_xml)
        # Store the exact full-frame arm motion visible in the delivered video,
        # including any collision-aware redundant-posture correction.
        if manifest.get('trajectory_revision')=='diverse_regrasp_v1':
            path=G.ROOT/'objects'/args.object/'trajectory.npz'
            with np.load(path) as saved:fields=dict(saved)
            fields['robot_frame_indices']=np.arange(len(rendered_q))
            fields['robot_q']=rendered_q
            np.savez_compressed(path,**fields)
            from organize import publish_segments
            publish_segments(path.parent)
        from organize import publish_video
        publish_video(G.ROOT/'objects'/args.object,args.video)
        print(args.video,flush=True)
        return
    model,mesh,initial,candidate,trial,poses,rule,checks,arm=find(args.object,args.pairs)
    if args.export:
        from sequence_export import export
        export(args.object,mesh,initial,candidate,trial,poses,rule,checks,arm)
    if args.video:
        K.render(model,trial.history,args.video,args.object,'parallel',
                 robot_track=(trial.arm_frames,trial.arm_q))
        if args.export:
            from organize import publish_video
            publish_video(G.ROOT/'objects'/args.object,args.video)
        print(args.video,flush=True)


if __name__=='__main__':main()
