"""Generate a diverse robot demo with an explicitly assumed rigid grasp.

Each attachment starts only after bilateral finger contact, and is released
on a stable intermediate rest. This mode does not establish grasp wrench
feasibility. The actual MuJoCo simulation runs at 1 ms; controller waypoints
are held for multiple physics steps to leave room under robot speed limits.
"""
import argparse
import hashlib

import numpy as np

import antipodal_rays as A
import regrasp_sequence as R
import kuka_transfer as K


def find(name,pairs=160,candidate_budget=120,resume=True,physics_steps=2,downward_component=.65,
         comfortable_transit=False,roll_step=None,pairwise_grasp_direction=12.,
         anchor_index=0,anchor_yaw=0.,grasp_method='rays',joint_transit=False,tool='standard',min_grasp_width=.015,
         com_weight=2.5):
    if not np.isfinite(com_weight) or com_weight<0:
        raise ValueError('com_weight must be finite and nonnegative')
    if grasp_method not in ('rays','sampled'):
        raise ValueError('grasp_method must be rays or sampled')
    if grasp_method=='sampled' and roll_step is not None:
        raise ValueError('The sampled candidate generator uses its fixed 15-degree roll menu')
    original_rays=A.candidates;original_sampled=R.G.candidates
    original_candidates=original_rays if grasp_method=='rays' else original_sampled
    original_solve=K.solve;original_export=R.export
    original_step=R.RegraspTrial.step;original_move=R.RegraspTrial.hand_move
    original_pickup=R.RegraspTrial.pickup
    cache={}
    rolls=None if roll_step is None else sorted(np.arange(-45.,45.0001,roll_step).tolist(),
                                               key=lambda value:(abs(value),value))

    def candidates(*args,**kwargs):
        if grasp_method=='rays':kwargs['rolls']=rolls
        for candidate in original_candidates(*args,**kwargs):
            if candidate['hand'][2,2] < -downward_component:
                yield candidate
                # Swapping identical jaw labels leaves the physical grasp
                # unchanged but can avoid a half-turn of the robot wrist.
                swapped=dict(candidate)
                swapped['hand']=candidate['hand']@np.diag([-1.,-1.,1.,1.])
                swapped['contacts']=candidate['contacts'][::-1].copy()
                swapped['face_ids']=candidate['face_ids'][::-1]
                swapped['roll_deg']=-candidate['roll_deg']
                yield swapped

    def solve(arm,target,previous=None):
        key=(arm.base.tobytes(),target.tobytes(),None if previous is None else previous.tobytes())
        if key not in cache:
            try:
                q,error=original_solve(arm,target,previous);cache[key]=(q.copy(),error)
            except ValueError as error:
                cache[key]=str(error)
        value=cache[key]
        if isinstance(value,str):raise ValueError(value)
        return value[0].copy(),value[1]

    def step(trial,hand,force=None):
        for _ in range(physics_steps):
            before=len(trial.history)
            original_step(trial,hand,force)
            if getattr(trial,'_demo_joint_command',None) is not None and len(trial.history)>before:
                trial._demo_joint_frames.append(len(trial.history)-1-trial._demo_joint_start)
                trial._demo_joint_q.append(trial._demo_joint_command.copy())

    def joint_move(trial,target,duration):
        arm=trial._demo_arm
        source_q=trial._demo_empty_q.copy()
        try:
            target_q,_=K.solve(arm,target,source_q)
        except ValueError:
            try:
                target_q,_=K.solve(arm,target,None)
            except ValueError as error:
                raise ValueError(f'Empty joint-transit endpoint IK failed at {target[:3,3].tolist()}') from error
        # Cubic smoothstep has peak slope 1.5. Reserve 25% speed headroom
        # for the finite-stiffness hand follower and later exact-pose IK.
        actual_duration=max(duration*physics_steps,
                            float(np.max(1.5*np.abs(target_q-source_q)/arm.velocity))/.75)
        count=max(1,int(np.ceil(actual_duration/(.001*physics_steps))))
        try:
            for index in range(count):
                q=source_q+R.G.smooth((index+1)/count)*(target_q-source_q)
                position,rotation,_=K.flange(arm,q)
                hand=np.eye(4);hand[:3,:3]=rotation;hand[:3,3]=position
                trial._demo_joint_command=q
                trial.step(hand,trial.open_force())
            trial._demo_empty_q=target_q.copy()
        finally:
            trial._demo_joint_command=None

    def hand_move(trial,target,duration=1.5,opened=True):
        source=R.G.transform(trial.data,trial.hand)
        if joint_transit and getattr(trial,'_demo_empty_transit',False) and opened:
            if (target[2,3]>source[2,3]+.05
                    and np.linalg.norm(target[:2,3]-source[:2,3])<1e-6):
                target=target.copy();target[:2,3]=[-.1,-.1] if comfortable_transit else [0.,0.]
                if comfortable_transit:target[2,3]=max(.4,target[2,3])
            return joint_move(trial,target,duration)
        if comfortable_transit and getattr(trial,'_demo_empty_transit',False) and opened:
            if (target[2,3]>source[2,3]+.05
                    and np.linalg.norm(target[:2,3]-source[:2,3])<1e-6):
                target=target.copy();target[:2,3]=[-.1,-.1];target[2,3]=max(.4,target[2,3])
                return original_move(trial,target,duration,opened)
            if source[2,3]>=.399 and target[2,3]>=.319:
                target=target.copy();target[2,3]=max(.4,target[2,3])
                turn=K.Rotation.from_matrix(target[:3,:3]@source[:3,:3].T).magnitude()
                if turn>.02:
                    rotation_only=source.copy();rotation_only[:3,:3]=target[:3,:3]
                    original_move(trial,rotation_only,duration,opened)
                return original_move(trial,target,duration,opened)
        if (opened and target[2,3]>source[2,3]+.05
                and np.linalg.norm(target[:2,3]-source[:2,3])<1e-6):
            target=target.copy();target[:2,3]=0.
        return original_move(trial,target,duration,opened)

    def pickup(trial,candidate,reference_rest,first=False):
        trial._demo_empty_transit=not first
        trial._demo_direct_joint_transit=joint_transit and not first
        trial.pickup_robot_track=None
        if joint_transit and not first:
            trial._demo_arm=K.Arm('A');trial._demo_arm.base[:]=[-.45,-.35,0.]
            trial._demo_empty_q=trial.empty_arm_q.copy()
            trial._demo_joint_start=len(trial.history)-1
            trial._demo_joint_frames=[0];trial._demo_joint_q=[trial._demo_empty_q.copy()]
        try:
            result=original_pickup(trial,candidate,reference_rest,first)
            if joint_transit and not first:
                trial.pickup_robot_track=(np.array(trial._demo_joint_frames),np.array(trial._demo_joint_q))
            return result
        finally:
            trial._demo_empty_transit=False
            trial._demo_direct_joint_transit=False
            trial._demo_joint_command=None

    def export(*args,**kwargs):
        kwargs['generator']='codes/setup/demo_regrasp.py'
        metadata=dict(kwargs.get('metadata') or {})
        metadata.update(controller_physics_steps_per_waypoint=physics_steps,
            empty_hand_lift_waypoint_xy_m=[-.1,-.1] if comfortable_transit else [0.,0.],
            empty_hand_comfortable_transit=comfortable_transit,
            empty_hand_joint_transit=joint_transit,
            grasp_candidate_filter=dict(maximum_world_approach_z=-downward_component,
                                        equivalent_jaw_label_variants=True),candidate_rolls_deg=rolls,
            grasp_candidate_method=grasp_method,grasp_com_distance_weight=com_weight)
        metadata['implementation_sha256']={path:hashlib.sha256((R.G.ROOT/path).read_bytes()).hexdigest()
            for path in ('codes/setup/regrasp_sequence.py','codes/setup/antipodal_rays.py',
                         'codes/setup/grasp.py','codes/setup/kuka_transfer.py')}
        kwargs['metadata']=metadata
        return original_export(*args,**kwargs)

    if grasp_method=='rays':A.candidates=candidates
    else:R.G.candidates=candidates
    K.solve,R.export=solve,export
    R.RegraspTrial.step,R.RegraspTrial.hand_move=step,hand_move
    R.RegraspTrial.pickup=pickup
    try:
        return R.find(name,pairs,candidate_budget,resume=resume,grasp_method=grasp_method,ideal_grasp=True,
                      pairwise_grasp_direction_deg=pairwise_grasp_direction,
                      anchor_index=anchor_index,anchor_yaw=anchor_yaw,tool=tool,
                      min_grasp_width=min_grasp_width,com_weight=com_weight)
    finally:
        A.candidates,R.G.candidates=original_rays,original_sampled
        K.solve,R.export=original_solve,original_export
        R.RegraspTrial.step,R.RegraspTrial.hand_move=original_step,original_move
        R.RegraspTrial.pickup=original_pickup


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('--pairs',type=int,default=160)
    parser.add_argument('--candidate-budget',type=int,default=120)
    parser.add_argument('--physics-steps',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--downward-component',type=float,default=.65)
    parser.add_argument('--grasp-method',choices=('rays','sampled'),default='rays')
    parser.add_argument('--comfortable-transit',action='store_true')
    parser.add_argument('--joint-transit',action='store_true')
    parser.add_argument('--tool',choices=('standard','compact'),default='standard')
    parser.add_argument('--min-grasp-width',type=float,default=.015)
    parser.add_argument('--com-weight',type=float,default=2.5)
    parser.add_argument('--roll-step',type=float)
    parser.add_argument('--pairwise-grasp-direction',type=float,default=12.)
    parser.add_argument('--anchor-index',type=int,default=0)
    parser.add_argument('--anchor-yaw',type=float,default=0.)
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    if not 0 <= args.downward_component < 1:parser.error('downward-component must be in [0, 1)')
    if args.roll_step is not None and not 0 < args.roll_step <= 90:parser.error('roll-step must be in (0, 90]')
    if args.grasp_method=='sampled' and args.roll_step is not None:parser.error('--roll-step requires --grasp-method rays')
    if not 0 < args.pairwise_grasp_direction <= 30:parser.error('pairwise-grasp-direction must be in (0, 30]')
    if not np.isfinite(args.com_weight) or args.com_weight<0:parser.error('com-weight must be finite and nonnegative')
    find(args.object,args.pairs,args.candidate_budget,not args.fresh,args.physics_steps,args.downward_component,
         args.comfortable_transit,args.roll_step,args.pairwise_grasp_direction,
         args.anchor_index,args.anchor_yaw,args.grasp_method,args.joint_transit,args.tool,args.min_grasp_width,args.com_weight)
