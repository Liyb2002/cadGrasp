"""Remove actual workpiece interiors without changing any saved placement."""
import numpy as np
from deterministic_base import solid,unpack,moved,union,volume


def remove_object_interiors(mesh,body,placements):
    # STL duplicates identical vertices. Merge at 15 decimals, preserving
    # their coordinates, before building the exact closed workpiece solid.
    workpiece=body.copy()
    workpiece.merge_vertices(digits_vertex=15)
    original=solid(mesh)
    excluded=union([moved(solid(workpiece),np.asarray(Q)) for Q in placements])
    result=(original-excluded)^original
    # Retessellated coincident boundaries occasionally need another subset
    # intersection. Keep the original material bound; never waive it.
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
        removed_cm3=volume(original-result)*1e6,
        input_intrusion_cm3=volume(original^excluded)*1e6,
        original_object_coordinates_and_layout_unchanged=True)
