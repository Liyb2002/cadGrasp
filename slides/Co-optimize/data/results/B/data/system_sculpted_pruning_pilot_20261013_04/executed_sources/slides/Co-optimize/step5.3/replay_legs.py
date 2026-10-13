"""Reconstruct saved legal leg material on the new certified connected core."""
import copy
import numpy as np

from deterministic_base import solid,unpack,volume,enclosure
from prune_support import material_components
from stage5_shared import occupied_bounds


def reconstruct(fixture,previous_core,previous_final,witness,rows,body,record):
    core,anchor,reference=map(solid,[fixture,previous_core,previous_final])
    allowed=solid(witness)
    if len(material_components(core))!=1 or volume(anchor-core)>1e-10:
        raise ValueError('Fixed leg growth requires the certified retained connected core')
    # The old core is already contained in the new one. This union is exactly
    # current core + saved leg material, without a coincident subtraction seam.
    value=core+reference
    mesh=unpack(value);cover=enclosure(mesh,rows)
    target=occupied_bounds(body,previous_final,rows);actual=occupied_bounds(body,mesh,rows)
    if (not mesh.is_watertight or len(material_components(value))!=1 or
            volume(value-allowed)>1e-10 or volume(core-value)>1e-10 or
            not all(r['passed']for r in cover)or not np.allclose(actual,target,atol=1e-8,rtol=0)):
        raise ValueError('Saved leg reconstruction failed geometry, floor, retention or BBX checks')
    construction=copy.deepcopy(record)
    construction.update(passed=True,connected=True,watertight=True,per_pose=cover,
        material_cm3=volume(value)*1e6,added_material_cm3=volume(value-core)*1e6,
        retained_legal_step4_material_loss_cm3=volume(core-value)*1e6,
        outside_saved_legal_component_cm3=volume(value-allowed)*1e6,
        saved_leg_geometry_reconstructed=True,leg_paths_and_sections_unchanged=True,
        locked_final_bbx_bounds_m=target.tolist(),locked_final_bbx_unchanged=True,
        force_solver_run=False)
    construction['history'].append(dict(operation='reconstruct_saved_legal_legs_on_rounded_core',
        reference_core_loss_cm3=volume(anchor-core)*1e6,
        saved_final_material_loss_cm3=volume(reference-value)*1e6,
        added_leg_material_cm3=volume(value-core)*1e6,component_count=1))
    return mesh,construction
