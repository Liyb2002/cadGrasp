"""Diverse cube sequence with exact opposing-face grasp candidates.

The tool is the unchanged generic straight-finger gripper. Only contact point
selection uses the cube's known planar faces; every motion is checked by the
shared contact simulation and KUKA trajectory search.
"""
import argparse
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

import overhead_regrasp as O


def candidates(mesh,rest,limit=120,max_width=.15,depths=(.112,.118,.10)):
    half=mesh.extents/2
    if not np.allclose(half,half[0],rtol=1e-5):
        raise ValueError('Analytic cube grasps require equal side lengths')
    center=mesh.bounds.mean(axis=0)
    vertical=int(np.argmax(abs(rest[2,:3])))
    for closing in (i for i in range(3) if i!=vertical):
        lateral=next(i for i in range(3) if i not in (vertical,closing))
        axis=rest[:3,closing]
        down=np.array([0.,0.,-1.]);down-=axis*np.dot(axis,down);down/=np.linalg.norm(down)
        for height in (0.,-.010,.010):
            for offset in (0.,-.015,.015):
                contact_center=center.copy();contact_center[vertical]+=height;contact_center[lateral]+=offset
                local=np.repeat(contact_center[None],2,axis=0)
                local[:,closing]+=[-half[closing],half[closing]]
                contacts=local@rest[:3,:3].T+rest[:3,3]
                for roll in (0,-15,15,-30,30,-45,45):
                    approach=Rotation.from_rotvec(axis*np.radians(roll)).apply(down)
                    rotation=np.column_stack((np.cross(axis,approach),axis,approach))
                    for depth in depths:
                        hand=np.eye(4);hand[:3,:3]=rotation;hand[:3,3]=contacts.mean(axis=0)-depth*approach
                        yield dict(hand=hand,width=float(2*half[closing]),contacts=contacts,
                            opening=min(max_width/2+.005,float(half[closing])+.012),
                            face_ids=[-1,-1],roll_deg=roll,
                            com_distance_m=float(np.linalg.norm(contact_center-mesh.center_mass)))


def find(resume=True,candidate_budget=120):
    original_candidates=O.R.G.candidates;original_export=O.R.export
    original_step=O.R.RegraspTrial.step;original_hand_move=O.R.RegraspTrial.hand_move

    def slow_step(trial,hand,force=None):
        # Integrate twice at the real 1 ms physics timestep per controller
        # waypoint. Saved states and event times retain actual elapsed time.
        original_step(trial,hand,force)
        original_step(trial,hand,force)

    def hand_move(trial,target,duration=1.5,opened=True):
        source=O.R.G.transform(trial.data,trial.hand)
        if (opened and target[2,3]>source[2,3]+.05
                and np.linalg.norm(target[:2,3]-source[:2,3])<1e-6):
            # A diagonal empty-hand retreat avoids folding the wrist close
            # to the robot base while rising straight above a side grasp.
            target=target.copy();target[:2,3]=0.
        return original_hand_move(trial,target,duration,opened)

    def export(*args,**kwargs):
        kwargs['generator']='codes/setup/cuboid_regrasp.py'
        metadata=dict(kwargs.get('metadata') or {})
        metadata['grasp_candidate_geometry']='Exact opposing cube faces; shifted centers and wrist approach rolls'
        metadata['controller_physics_steps_per_waypoint']=2
        metadata['empty_hand_lift_waypoint_xy_m']=[0.,0.]
        metadata.setdefault('implementation_sha256',{})['codes/setup/overhead_regrasp.py']=hashlib.sha256(
            (O.R.G.ROOT/'codes/setup/overhead_regrasp.py').read_bytes()).hexdigest()
        kwargs['metadata']=metadata
        return original_export(*args,**kwargs)
    O.R.G.candidates=candidates;O.R.export=export
    O.R.RegraspTrial.step=slow_step;O.R.RegraspTrial.hand_move=hand_move
    try:
        return O.find('cuboid_baseline',candidate_budget=candidate_budget,resume=resume)
    finally:
        O.R.G.candidates=original_candidates;O.R.export=original_export
        O.R.RegraspTrial.step=original_step;O.R.RegraspTrial.hand_move=original_hand_move


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--candidate-budget',type=int,default=120)
    args=parser.parse_args()
    find(not args.fresh,args.candidate_budget)
