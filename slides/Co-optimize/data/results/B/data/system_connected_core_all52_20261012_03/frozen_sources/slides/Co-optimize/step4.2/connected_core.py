"""Keep one actual contact core before the original force and floor gates.

The material domain is the existing finite wrap. No connector is generated.
Deleted components never contribute contact reactions or Oracle anchors.
"""
from collections import OrderedDict
import numpy as np
from whole_search.common import C, unpack_solid


def largest_material_component(value):
    from prune_support import material_components
    parts = material_components(value)
    if not parts:
        raise ValueError('No positive contact material remains')
    kept = parts[0]
    return kept, dict(input_component_count=len(parts), component_count=1,
        component_volumes_cm3=[C.material_volume(p)*1e6 for p in parts],
        deleted_isolated_material_cm3=C.material_volume(value-kept)*1e6,
        added_material_cm3=C.material_volume(kept-value)*1e6,
        selection='largest material component; original demands rechecked',
        connection_material_added=False)


def prepare(model, layout):
    import sys
    from pathlib import Path
    for folder in ['step5.2', 'step5.3']:
        path=str(Path(__file__).resolve().parents[1]/folder)
        if path not in sys.path:sys.path.insert(0,path)
    cache=getattr(model,'connected_core_cache',None)
    if cache is None:
        cache=model.connected_core_cache=OrderedDict()
        from whole_step4_render import NominalBuilder
        model.connected_core_builder=NominalBuilder(model)
    key=layout.key()
    if key not in cache:
        builder=model.connected_core_builder
        original=builder.solid(layout)
        kept,record=largest_material_component(original)
        mesh=unpack_solid(kept)
        # Certify precisely the OBJ boundary subsequently consumed by Step5.
        mesh=C.trimesh.load(C.trimesh.util.wrap_as_stream(
            mesh.export(file_type='obj',digits=16)),file_type='obj',process=False)
        if not mesh.is_watertight:
            raise ValueError('Connected contact core is not watertight')
        required=[float(model.base_oracle.loads[k][:,2].min()) for k in layout.active]
        record.update(connected=True,watertight=True,
            downward_resultant_passed=all(v>0. for v in required),
            minimum_required_upward_ground_reaction_mg=min(required),
            applied_resultant_convention='opposite of saved need_wrench reaction demand',
            all_pose_floor_clipping_applied=True,
            material_domain='original finite Step4 wrap minus original exclusions')
        cache[key]=(mesh,record)
        while len(cache)>128:cache.popitem(last=False)
        if len(builder.shapes)>128:builder.shapes.clear()
        if len(builder.parts)>2048:builder.parts.clear()
    return cache[key]
