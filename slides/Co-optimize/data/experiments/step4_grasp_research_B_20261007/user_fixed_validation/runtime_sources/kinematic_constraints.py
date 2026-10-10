"""User-specified rigid grasp as a constraint, independent of object path."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'single_load'))
from co_common import S,material_volume
from robot import Panda
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

class FixedGrasp:
    def __init__(self,T_object_hand,gap_m):
        T=np.array(T_object_hand,dtype=float,copy=True)
        if T.shape!=(4,4) or not np.all(np.isfinite(T)):raise ValueError('Expected finite 4x4 T_object_hand')
        if not np.allclose(T[3],[0,0,0,1],atol=1e-10):raise ValueError('Invalid homogeneous transform')
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-8):raise ValueError('Expected proper rotation')
        if not .003<=gap_m<=.080:raise ValueError('Contact gap outside pilot range [3,80]mm')
        T.setflags(write=False);self.T_object_hand=T;self.gap_m=float(gap_m)

    def hand_pose(self,T_world_object):
        return np.asarray(T_world_object)@self.T_object_hand

    def validate_path(self,robot,mesh,poses,fixture_world=None):
        """Sampled SE3 path check; accepts translations AND rotations.

        Caller generates/resamples poses; no direction tied to hand approach.
        No final fixture acceptance or continuous collision certificate.
        """
        q=robot.home.copy();records=[]
        for index,T in enumerate(poses):
            hand=self.hand_pose(T)
            q=robot.ik(hand,q)
            if q is None:return dict(passed=False,reason='path_ik',sample=index)
            object_world=mesh.copy();object_world.apply_transform(T);solid=S.solid(object_world)
            if fixture_world is not None:
                if material_volume(solid^fixture_world)>1e-11:return dict(passed=False,reason='object_fixture_collision',sample=index)
                if not robot.clear(hand,self.gap_m,fixture_world):return dict(passed=False,reason='hand_fixture_collision',sample=index)
            report=robot.arm_report(q,self.gap_m,solid if fixture_world is None else solid+fixture_world)
            if not report['passed']:return dict(passed=False,reason='robot_collision',sample=index,collision=report)
            records.append(dict(T_world_object=np.asarray(T).tolist(),T_world_hand=hand.tolist(),q=q.tolist()))
        return dict(passed=True,samples=records,grasp_fixed=True,continuous_collision_certificate=False)

def interpolate_path(waypoints,maximum_translation_m=.01,maximum_rotation_deg=5.):
    """Piecewise translation + spherical rotation; no straight-path assumption."""
    if not waypoints:raise ValueError('At least one waypoint is required')
    result=[np.asarray(waypoints[0]).copy()]
    for A,B in zip(waypoints,waypoints[1:]):
        A=np.asarray(A);B=np.asarray(B)
        angle=Rotation.from_matrix(B[:3,:3]@A[:3,:3].T).magnitude()
        steps=max(1,int(np.ceil(np.linalg.norm(B[:3,3]-A[:3,3])/maximum_translation_m)),int(np.ceil(angle/np.radians(maximum_rotation_deg))))
        rotation=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))
        for fraction in np.linspace(0,1,steps+1)[1:]:
            T=np.eye(4);T[:3,:3]=rotation(fraction).as_matrix();T[:3,3]=(1-fraction)*A[:3,3]+fraction*B[:3,3];result.append(T)
    return result
