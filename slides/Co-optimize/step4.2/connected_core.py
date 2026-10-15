"""Keep one actual contact core before the original force and floor gates.

The material domain is the existing finite wrap. No connector is generated.
Deleted components never contribute contact reactions or Oracle anchors.
"""
from collections import OrderedDict
import hashlib
import time
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
    repaired=getattr(model,'connected_core_repairs',{}).get(layout.key())
    if repaired is not None:return repaired
    import sys
    from pathlib import Path
    for folder in ['step5.2', 'step5.3']:
        path=str(Path(__file__).resolve().parents[1]/folder)
        if path not in sys.path:sys.path.insert(0,path)
    cache=getattr(model,'connected_core_cache',None)
    if cache is None:
        cache=model.connected_core_cache=OrderedDict()
    if not hasattr(model,'connected_core_builder'):
        from whole_step4_render import NominalBuilder
        model.connected_core_builder=NominalBuilder(model)
    key=layout.key()
    if key not in cache:
        from search_runtime import runtime, layout_key, MISSING
        store=runtime(model)
        saved=store.get('contact_core',layout_key(layout)) if store else MISSING
        if saved is not MISSING:
            cache[key]=saved
            while len(cache)>16:cache.popitem(last=False)
            return saved
        builder=model.connected_core_builder
        began=time.monotonic()
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
        if store:
            store.timings['contact_core']=store.timings.get('contact_core',0.)+time.monotonic()-began
            store.put('contact_core',layout_key(layout),(mesh,record))
        while len(cache)>(16 if store else 128):cache.popitem(last=False)
        if len(builder.shapes)>128:builder.shapes.clear()
        if len(builder.parts)>2048:builder.parts.clear()
    return cache[key]


def legal_clip(model,layout,proposal):
    """Delete a tiny numerical exclusion mismatch; never relax acceptance."""
    original,record=prepare(model,layout)
    kept,components=largest_material_component(C.S.solid(proposal['core']))
    mesh=unpack_solid(kept)
    mesh=C.trimesh.load(C.trimesh.util.wrap_as_stream(mesh.export(file_type='obj',digits=16)),
        file_type='obj',process=False)
    actual=C.S.solid(mesh);allowed=C.S.solid(proposal['witness'])
    outside=C.material_volume(actual-allowed)
    added=C.material_volume(actual-C.S.solid(original))
    if not mesh.is_watertight or outside>1e-10 or added>1e-10:
        raise ValueError('Legal-core clipping failed original subset/space/boundary checks')
    updated=dict(record,connected=True,component_count=1,
        oracle_boundary_cleanup=dict(removed_cm3=C.material_volume(C.S.solid(original)-actual)*1e6,
            outside_after_cm3=outside*1e6,added_cm3=added*1e6,
            acceptance_tolerance_m3_unchanged=1e-10,requires_actual_contact_recertification=True))
    return mesh,updated


def proxy(model,layout,flags):
    """The unchanged fixed/adaptive cone-distance measure on retained rays."""
    from continuous_support.projection import project_demands
    cache=getattr(model,'connected_projection_cache',None)
    if cache is None:cache=model.connected_projection_cache=OrderedDict()
    losses=[];residuals=[]
    for k in layout.active:
        key=(k,hashlib.sha256(flags[k].tobytes()).digest(),model.floor_signature(layout,k),model.quadrature_epoch)
        if key not in cache:
            rays,_=model.supply_at_points(k,flags[k],layout)
            needs,weights,scale=model.demand_screen.demands[k]
            projection=project_demands(rays,needs)
            loss=float(weights@projection['losses']/scale**2)
            residual=float(np.max(np.abs(projection['residuals'])))
            if k in model.adaptive_demands:
                needs,weights,_=model.adaptive_demands[k]
                adaptive=project_demands(rays,needs)
                loss=(loss+float(weights@adaptive['losses']/model.original_scales[k]**2))/2
            cache[key]=(0. if loss<1e-28 else loss,residual)
            while len(cache)>2048:cache.popitem(last=False)
        loss,residual=cache[key];losses.append(loss);residuals.append(residual)
    return dict(loss=float(np.mean(losses)),sum_loss=float(np.sum(losses)),residual_loss=losses,
        maximum_residual=residuals,span_m=model.layout_span(layout),sampled_guidance_only=True,
        quadrature_epoch=model.quadrature_epoch,contact_domain='retained connected core',
        objective='all_pose_integrated_squared_original_cone_distance')
