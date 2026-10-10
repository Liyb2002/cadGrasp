"""Publish the existing force/torque decision; mesh export is presentation only."""
import time
from pathlib import Path
import numpy as np
from whole_search.common import C
from whole_search.fast_search import FastReuseSearch
from whole_search.reuse_first import registered,juxtaposed_count
from whole_search.search import failed_count


def export_nominal_mesh(model,layout,out):
    from whole_step4_render import NominalBuilder
    from whole_search.common import unpack_solid
    builder=NominalBuilder(model)
    mesh=unpack_solid(builder.solid(layout))
    C.D.export_exact_obj(mesh,out/'support.obj')
    model.force_display_builder=builder
    return abs(float(mesh.volume))*1e6


def save_force_result(search,current,mode,began,initialization,**extra):
    """Never call evaluate, exact or classify again for the selected layout."""
    model=search.model;out=search.out;layout=current['layout'];identity=layout.key()
    search.search_only=True
    sampled=FastReuseSearch.save(search,current,mode,began,initialization,**extra)
    active=layout.active
    passed=len(active)==len(model.poses) and failed_count(current)==0
    np.savez_compressed(out/'layout.npz',placements=layout.placements,directions=layout.directions,
                        hosts=layout.hosts,active=np.array(active),native_world=model.native)
    for k in active:
        np.savez_compressed(out/f'{model.poses[k]}_force.npz',mask=current['masks'][k],
                            supply_7d=current['supplies'][k])
    report=dict(sampled,object=model.name,passed=passed,force_passed=passed,
        acceptance='all_original_force_torque_demands_on_search_contacts',
        load_count_per_pose=32768,original_loads_reused=True,
        force_exit_work_passed=False,geometry_verified=False,final_acceptance_run=False,
        full_demand_recheck_run=False,geometry_recovery_run=False,validation_seconds=0.,
        initialization=initialization,estimated_volume_cm3=current['volume_cm3'],
        volume_cm3=current['volume_cm3'],volume_measure='nominal_material_occupancy_estimate',
        translation_axes='world_xyz' if model.allow_z_translation else 'world_xy',
        translation_axis_priority='equal',
        workpiece_heights_m={model.poses[k]:model.workpiece_height(layout,k) for k in active},
        workpiece_floor_contact_allowed={model.poses[k]:model.floor_signature(layout,k) for k in active},
        rotating_reuse_pose_count=sum(registered(layout,k) for k in active),
        juxtaposed_pose_count=juxtaposed_count(layout),
        force_model=C.U.description(),original_floor_model=C.FLOOR.description(),
        exact_evaluations=model.exact_calls,
        provenance=C.provenance(model.inputs,[Path(__file__)]+C.code_sources()),
        executed_search_sources_sha256=model.startup_sources,
        mesh_exported=False,mesh_export_changed_layout=False)
    export_began=time.monotonic()
    if passed:
        try:
            volume=export_nominal_mesh(model,layout,out)
            report.update(mesh_exported=True,display_mesh_volume_cm3=volume,volume_cm3=volume,
                          volume_measure='exported_nominal_mesh_material')
        except Exception as error:
            report['mesh_export_error']=str(error)
            (out/'support.obj').unlink(missing_ok=True)
    assert layout.key()==identity,'Presentation must never alter the accepted layout'
    report.update(mesh_export_seconds=time.monotonic()-export_began,seconds=time.monotonic()-began)
    report['artifacts']={p.name:C.I.sha256(p) for p in out.iterdir()
                         if p.is_file() and p.name!='report.json'}
    C.save(out/'report.json',report)
    return report
