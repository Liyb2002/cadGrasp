"""Current target: completed occupied volume, with absolute-direction features.

No contact count term. Directions are signed world vectors. Finite failed
searches are unknown labels; hard-invalid actions are masked separately.
"""
import numpy as np


def unit(vectors):
    vectors=np.asarray(vectors,float)
    lengths=np.linalg.norm(vectors,axis=-1,keepdims=True)
    if np.any(lengths<=1e-12):raise ValueError('Zero direction')
    return vectors/lengths


def absolute_spread(vectors,weights=None):
    """Weighted pairwise signed dispersion; unchanged by splitting a patch."""
    vectors=unit(vectors)
    weights=np.ones(len(vectors)) if weights is None else np.asarray(weights,float)
    if weights.shape!=(len(vectors),) or np.any(weights<0) or weights.sum()<=0:raise ValueError('Invalid weights')
    weights=weights/weights.sum()
    # 0: same signed direction; 1: opposing direction pair.
    return float(np.sum(weights[:,None]*weights[None,:]*(1-np.clip(vectors@vectors.T,-1,1)))/2)


def alignment(normals,exit_direction,areas):
    """Geometry proxy for force directions, not solved reaction forces."""
    normals=unit(normals);direction=unit(exit_direction)
    return float(np.average(1-np.clip(normals@direction,-1,1),weights=np.asarray(areas,float)))


def terminal_cost(step5,hard_passed):
    if not hard_passed:raise ValueError('Failed terminal states have no feasible volume target')
    metrics=step5.get('metrics',step5)
    supported=metrics['object_and_support_poses']['box_volume_cm3']
    object_only=metrics['object_poses']['box_volume_cm3']
    if object_only<=0:raise ValueError('Invalid reference volume')
    return float(supported/object_only-1)


def action_target(successful_completions):
    """Best observed continuation, not a certified global optimum.

Each item is (Step5 result, full hard-constraint pass). No successful bounded
continuation means unlabeled, never fabricated infinite/impossible ground truth.
"""
    costs=[terminal_cost(r,passed) for r,passed in successful_completions if passed]
    return min(costs) if costs else None
