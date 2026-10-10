"""World XYZ updates, projected to keep the workpiece above the floor."""
import numpy as np


def translation_frame(model, layout, owner):
    """World XYZ axes expressed in the fixture frame; no horizontal priority."""
    dimensions = 3 if getattr(model,'allow_z_translation',False) else 2
    return model.native[layout.hosts[owner], :3, :3].T[:, :dimensions]


def shift(model, layout, owner, vector):
    trial = layout.copy()
    trial.placements[owner, :3, 3] += vector
    if getattr(model,'allow_z_translation',False):
        height = model.workpiece_height(trial, owner)
        if height < 0.:
            trial.placements[owner, :3, 3] -= height*model.floor_normal(trial, owner)
    return trial


def derivative(model, layout, owners, axis, delta, objective=None):
    """Resolved secant; one-sided at the nonpenetration boundary."""
    minus, plus = layout.copy(), layout.copy()
    negative = (min(delta,max(0.,min(model.workpiece_height(layout,k) for k in owners)))
                if axis==2 else delta)
    for k in owners:
        vector = model.native[layout.hosts[k], :3, :3].T @ np.eye(3)[axis]
        minus = shift(model, minus, k, -negative*vector)
        plus = shift(model, plus, k, delta*vector)
    objective = objective or (lambda q: model.proxy(q)['loss'])
    return (objective(plus)-objective(minus))/(delta+negative)


def translation_gradient(model, layout, owners, delta, objective=None):
    """One gradient in world coordinates, projected at the floor boundary."""
    dimensions = 3 if getattr(model,'allow_z_translation',False) else 2
    gradient = np.array([derivative(model,layout,owners,axis,delta,objective)
                         for axis in range(dimensions)])
    if dimensions == 3 and min(model.workpiece_height(layout,k) for k in owners)<=1e-9:
        gradient[2] = min(gradient[2],0.)
    return gradient
