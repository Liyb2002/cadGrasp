"""Controlled richer whole-family turns for difficult final merges.

Uses the identical core, oracle and full-load gates. Only the proposal seeds
change. The original seed set remains available as a comparison.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from whole_task_2.operations.state import transform
from . import membrane_search as search

original_seeds=search.seed_layouts


def seeds(q,na,host,rng,kind,extent,left=None,right=None):
    values,boundary=original_seeds(q,na,host,rng,kind,extent,left,right)
    if kind!='rotational_reuse':return values,boundary
    # A common seat is only one useful relative orientation. Let the already
    # merged guest turn as one rigid family around that seat, including a
    # side turn. Every turn is subsequently projected and freshly certified.
    center=transform(np.zeros(3),host)
    extra=[]
    axes=rng.normal(size=(3,3));axes/=np.linalg.norm(axes,axis=1)[:,None]
    for axis,angle in zip(axes,(20.,45.,90.)):
        for sign in (-1.,1.):
            R=Rotation.from_rotvec(axis*np.radians(sign*angle)).as_matrix()
            C=np.eye(4);C[:3,:3]=R;C[:3,3]=center-R@center
            trial=q.copy();trial.material[na:]=C@q.material[na:];extra.append(trial)
    return values[:boundary]+extra+values[boundary:],boundary+len(extra)


def proposal(args):
    previous=search.seed_layouts
    try:
        search.seed_layouts=seeds
        result=search.proposal(args)
        result['proposal_seed_family']='common_seat_plus_20_45_90_degree_rigid_family_turns'
        return result
    finally:search.seed_layouts=previous
