"""Unified world-coordinate translation descent and symmetric axis samples."""
import numpy as np
from .layout_update import translation_frame,shift,translation_gradient


def translation_proposals(model,layout,owners,*,fine=False):
    rows=[];delta=model.extent*(1/512 if fine else 1/100)
    fractions=[1/1024,1/512,1/256] if fine else [1/128,1/64,1/32,1/16,1/8]
    for k in owners:
        frame=translation_frame(model,layout,k)
        gradient=translation_gradient(model,layout,[k],delta)
        vectors=[(axis*sign,'sample') for axis in np.eye(len(gradient)) for sign in [-1,1]]
        length=float(np.linalg.norm(gradient))
        if length>1e-28:vectors.insert(0,(-gradient/length,'gradient'))
        for fraction in fractions:
            for vector,kind in vectors:
                trial=shift(model,layout,k,model.extent*fraction*frame@vector)
                if trial.key()==layout.key():continue
                rows.append(('translation-'+kind,trial,dict(pose=model.poses[k],
                    step_m=model.extent*fraction,finite_difference_m=delta,gradient=gradient.tolist(),
                    all_poses_in_objective=True,translation_dimensions=len(gradient),
                    coordinate='world_xyz' if len(gradient)==3 else 'world_xy',
                    world_height_m=model.workpiece_height(trial,k),
                    derivative_kind='resolved_contact_secant; one_sided_at_floor')))
    return rows


def coherent_translation_proposals(model,layout,owners,*,fine=False):
    if len(owners)<2:return []
    rows=[];delta=model.extent*(1/512 if fine else 1/100)
    fractions=[1/1024,1/512,1/256,1/128] if fine else [1/128,1/64,1/32,1/16]
    gradient=translation_gradient(model,layout,owners,delta)
    dimensions=len(gradient)
    def move(vector):
        q=layout.copy()
        for k in owners:q=shift(model,q,k,translation_frame(model,layout,k)@vector)
        return q
    length=float(np.linalg.norm(gradient))
    if length>1e-28:
        for fraction in fractions:
            rows.append(('translation-gradient-coherent',move(-model.extent*fraction*gradient/length),
                dict(poses=[model.poses[k] for k in owners],step_m=model.extent*fraction,
                     finite_difference_m=delta,gradient=gradient.tolist(),all_poses_in_objective=True,
                     translation_dimensions=dimensions,
                     coordinate='shared_world_xyz' if dimensions==3 else 'shared_world_xy')))
    for fraction in fractions[:3]:
        for axis in np.eye(dimensions):
            for sign in [-1,1]:
                q=move(sign*model.extent*fraction*axis)
                if q.key()!=layout.key():rows.append(('translation-coherent-sample',q,
                    dict(step_m=model.extent*fraction,axis=axis.tolist(),sign=sign,
                         translation_dimensions=dimensions)))
    return rows
