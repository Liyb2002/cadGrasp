"""Conservatively recut a serialized wrap in material and execution frames."""
import numpy as np
import trimesh

from whole_task_2.common.transforms import inverse
from .legacy import kernel as K, unpack_solid
from .work_access import SurfaceSegments


def condition_wrap(mesh, support, states, access, path, tolerance):
    """Keep every component; remove only intersections with declared exclusions.

    Coincident boundaries can behave differently after rigid transforms and
    serialization. Check the actual installed mesh, recut in that frame, and
    rebuild contacts only after every saved-frame check passes.
    """
    original = K.S.solid(support)
    material_forbidden = [K.S.solid(mesh), access.combined_solid()]
    world_forbidden = {pose: SurfaceSegments(task.domain.mesh, task.domain.work_ids,
        access.head_world).solid() for pose, (task, _, _) in states.items()}
    initial_overlap = None
    for cycle in range(6):
        K.D.export_exact_obj(support, path)
        actual = trimesh.load(path, force='mesh', process=False)
        solid = K.S.solid(actual)
        local_overlaps = [K.material_volume(solid ^ excluded) for excluded in material_forbidden]
        world_overlaps = {}
        for pose, (_, transform, _) in states.items():
            installed = actual.copy()
            installed.apply_transform(transform)
            world_overlaps[pose] = K.material_volume(K.S.solid(installed) ^ world_forbidden[pose])
        maximum = max(*local_overlaps, *world_overlaps.values())
        initial_overlap = maximum if initial_overlap is None else initial_overlap
        if maximum <= tolerance:
            added = K.material_volume(solid-original)
            if added > tolerance or not actual.is_watertight or not actual.is_winding_consistent:
                raise RuntimeError('Invalid conditioned wrap boundary or added material')
            return actual, dict(conditioning_cycles=cycle,
                initial_maximum_overlap_m3=initial_overlap, final_maximum_overlap_m3=maximum,
                removed_material_cm3=K.material_volume(original-solid)*1e6,
                added_material_m3=added, original_volume_tolerance_m3=tolerance,
                own_world_work_overlap_m3=world_overlaps,
                object_overlap_m3=local_overlaps[0], continuous_work_overlap_m3=local_overlaps[1],
                all_components_retained=True, actual_contacts_require_recheck=True)
        print(f'Step3: recutting serialized wrap, cycle {cycle+1}; maximum overlap {maximum:.6g} m3', flush=True)
        support = actual
        for pose, (_, transform, _) in states.items():
            if world_overlaps[pose] <= tolerance:
                continue
            installed = support.copy()
            installed.apply_transform(transform)
            support = unpack_solid(K.S.solid(installed)-world_forbidden[pose])
            support.apply_transform(inverse(transform))
        solid = K.S.solid(support)
        support = unpack_solid(solid-material_forbidden[0]-material_forbidden[1])
    raise RuntimeError('Serialized wrap remains unresolved after conservative recuts')
