"""World XYZ updates, projected to keep the workpiece above the floor."""
import numpy as np
from whole_search.common import tangent_frame


def translation_frame(model, layout, owner):
    normal = model.floor_normal(layout, owner)
    horizontal = tangent_frame(normal)
    return (np.column_stack([horizontal, normal])
            if getattr(model,'allow_z_translation',False) else horizontal)


def shift(model, layout, owner, vector):
    trial = layout.copy()
    trial.placements[owner, :3, 3] += vector
    if getattr(model,'allow_z_translation',False):
        height = model.workpiece_height(trial, owner)
        if height < 0.:
            trial.placements[owner, :3, 3] -= height*model.floor_normal(trial, owner)
    return trial


def derivative(model, layout, owners, axis, delta):
    """Resolved secant; one-sided at the nonpenetration boundary."""
    minus, plus = layout.copy(), layout.copy()
    negative = (min(delta,max(0.,min(model.workpiece_height(layout,k) for k in owners)))
                if axis==2 else delta)
    for k in owners:
        vector = model.native[layout.hosts[k], :3, :3].T @ np.eye(3)[axis]
        minus = shift(model, minus, k, -negative*vector)
        plus = shift(model, plus, k, delta*vector)
    return (model.proxy(plus)['loss']-model.proxy(minus)['loss'])/(delta+negative)
