"""A whole-family legal work-axis spin expands rotational reuse proposals.

Apply H to every world object pose and exit and inverse(H) to each material
installation. All material seats remain identical. Gravity remains world Z,
so the original loads and all new ground constraints must be checked again.
Parent files and their old proofs are immutable; this is only a proposal.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from whole_task.step3.run_membranes import load_model
from . import membrane_search as search


def spin(layout,angle,axis,center):
    R=Rotation.from_rotvec(np.asarray(axis)*np.radians(angle)).as_matrix()
    H=np.eye(4);H[:3,:3]=R;H[:3,3]=center-R@center
    q=layout.copy();q.poses=H@q.poses;q.material=q.material@np.linalg.inv(H)
    q.exits=q.exits@R.T
    return q


def proposal(args):
    if args[3]!='rotational_reuse':return search.proposal(args)
    model,_,_=load_model(args[0]);rng=np.random.default_rng(args[4]+1537)
    previous=search.read_group;right=args[2];old,mesh=previous(right)
    angles=np.array([-150.,-120.,-90.,-60.,-30.,30.,60.,90.,120.,150.,180.]);rng.shuffle(angles)
    chosen=None
    for angle in angles:
        q=spin(old,angle,model.axis,model.head)
        if (q.exits[:,2]>=.05).all():chosen=q;break
    if chosen is None:return search.proposal(args)

    def proposed_group(group):
        actual,support=previous(group)
        if group['sha256']==right['sha256']:return chosen.copy(),support
        return actual,support

    try:
        search.read_group=proposed_group
        result=search.proposal(args)
        result['world_pose_family_spin_deg']=float(angle)
        result['unchanged_material_seats_under_work_axis_spin']=True
        return result
    finally:search.read_group=previous
