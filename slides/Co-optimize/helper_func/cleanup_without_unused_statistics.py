"""Original deletion and guards, omitting statistics discarded by Step4."""
import numpy as np
from deterministic_base import solid,unpack,moved,union,volume


def remove_object_interiors(mesh,body,placements):
    workpiece=body.copy()
    workpiece.merge_vertices(digits_vertex=15)
    original=solid(mesh)
    excluded=union([moved(solid(workpiece),np.asarray(Q)) for Q in placements])
    result=(original-excluded)^original
    for _ in range(3):
        if volume(result-original)<=1e-10:
            break
        result=result^original
    if volume(result-original)>1e-10:
        sequential=original
        for Q in placements:
            sequential=sequential-moved(solid(workpiece),np.asarray(Q))
        result=sequential^original
    if volume(result-original)>1e-10:
        raise ValueError('Object-interior deletion added above-tolerance material: '+str(volume(result-original)*1e6)+' cm3')
    return unpack(result),dict(operation='remove_actual_object_interiors',
        statistics_computed=False,original_object_coordinates_and_layout_unchanged=True)
