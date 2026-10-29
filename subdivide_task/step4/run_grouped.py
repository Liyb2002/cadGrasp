"""Solve 13 regions with balanced, genuinely shared fixture installations."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from subdivide_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.step3.input_state import FAMILY_ROOT
from subdivide_task.step3.legacy import kernel as K, loaded_sources
from subdivide_task.common.wrenches import endpoint_demands
from subdivide_task.evaluation.metrics import static_bounds
from .angular_model import AngularTaskModel, AngularLayout
from .grouped_search import GroupedSearch, installation_summary
from .family_gate import condition
from .validate_mobile import validate_artifacts
from material_force import certify_material_contacts


def run(input_root=FAMILY_ROOT, output_root=DEFAULT_OUTPUT, object_name='B', tag='grouped13', height=0., resume=None, group_count=5):
    began = time.monotonic()
    input_root = Path(input_root).resolve(); output_root = Path(output_root).resolve()
    out = output_root/object_name/'step4'/tag
    if (out/'report.json').exists():
        raise ValueError('Choose a new tag to preserve a completed run')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'run_status.json', dict(status='loading_13_pose_families'))
    model = AngularTaskModel(name=object_name, output_root=input_root)
    if resume is not None:
        model.inputs.append(Path(resume).resolve())
    if len(model.poses) != 13:
        raise ValueError('Expected the current 13-region input')
    search = GroupedSearch(model, out, group_count=group_count)
    write_json(out/'run_status.json', dict(status='continuous_juxtaposition_search', region_count=len(model.poses)))
    item = search.solve(height=height, resume=resume)
    model.activate(item.layout)
    stage = out/'step4.2'; stage.mkdir(exist_ok=True)
    np.savez_compressed(stage/'layout.npz', **item.layout.arrays(model.native))
    K.D.export_exact_obj(item.witness, stage/'legal_component.obj')
    core, core_record = condition(model, item.layout, item.nominal, stage/'support.obj', False, item.witness)
    full, full_record = condition(model, item.layout, item.solid, stage/'complete_support.obj', True, item.witness)
    execution = [model.execution(item.layout, k) for k in item.layout.active]
    write_json(out/'execution.json', dict(schema='continuous_axial_pose_centered_execution_v1',
        poses=execution, fixed_working_regions=True, needle_direction_fixed=True,
        object_COM_xy_fixed=model.center_xy.tolist(), fixture_translation_allowed=True))
    proof = certify_material_contacts(model, item.layout, full)
    arrays, loads = {}, {}
    for k in item.layout.active:
        targets = endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale
        passed, _ = K.J.classify(proof['supplies'][k], targets)
        if not passed.all() or not proof['masks'][k].all():
            raise ValueError('Exported final support failed retained or continuous loads')
        arrays[f'sample_mask_{k}'] = proof['masks'][k]
        arrays[f'endpoint_mask_{k}'] = passed
        arrays[f'supply_{k}'] = proof['supplies'][k]
        loads[model.poses[k]] = dict(samples_passed=int(proof['masks'][k].sum()),
            sample_count=len(proof['masks'][k]), endpoints_passed=int(passed.sum()),
            endpoint_count=len(passed), actual_contact_area_cm2=proof['contacts'][k]['area_m2']*1e4,
            axial_angle_deg=float(item.layout.angles_deg[k] % 360.))
    np.savez_compressed(stage/'force_certificate.npz', **arrays)
    validation = validate_artifacts(model, out, item.layout)
    metrics = static_bounds(model.mesh.vertices, item.layout.arrays(model.native),
        step4_vertices=core.vertices, full_vertices=full.vertices,
        pose_names=model.poses, execution=execution)
    dimensions = np.asarray(metrics['with_full_support']['dimensions_m'])
    metrics['occupied_xy_bbox_area_cm2'] = float(np.prod(dimensions[:2])*1e4)
    metrics['full_material_volume_cm3'] = abs(float(full.volume))*1e6
    source_family = json.loads(model.family_source.read_text())
    report = dict(schema='whole_task_continuous_axial_juxtaposition_v1', status='pass',
        search_method='balanced_shared_installations_pose_first',
        support_installations=installation_summary(model, item.layout),
        initial_support_pose_count=13, installation_merge_count=13-group_count,
        requested_support_pose_count=group_count,
        full_fixture_accepted=True, object=object_name, poses=model.poses,
        pose_count=len(model.poses), input_root=str(input_root), output_root=str(output_root),
        selected_surface_union_fraction=source_family['selected_surface_union_fraction'],
        working_area_masks_changed=False, original_samples_per_pose=32768,
        total_original_sample_count=sum(len(task.targets) for task in model.tasks),
        final_axial_angles_deg=item.layout.angles_deg.tolist(),
        continuous_angles_optimized_during_juxtaposition=True,
        individual_contact_pieces_may_be_disconnected=True,
        complete_fixture_must_be_connected=True, physical_acceptance_tolerances_changed=False,
        loads=loads, evaluation=metrics, construction=item.construction,
        export_conditioning=dict(core=core_record, complete=full_record),
        guide_queries=search.guide.queries, guide_seconds=search.guide.seconds,
        resumed_generated_layout=str(Path(resume).resolve()) if resume is not None else None,
        timing_scope='this attempt, including checkpoint loading; earlier insertion trace retained separately',
        angle_state_updates=model.angle_updates, seconds=time.monotonic()-began,
        native_robot_IK_certified=False, global_minimum_occupancy_certified=False,
        input_sha256={str(p): sha256(p) for p in set(model.inputs+loaded_sources())},
        artifact_sha256={str(p.relative_to(out)): sha256(p) for p in [
            stage/'layout.npz', stage/'support.obj', stage/'complete_support.obj',
            stage/'legal_component.obj', stage/'force_certificate.npz', out/'execution.json']})
    write_json(out/'report.json', report)
    validation.update(report_sha256=sha256(out/'report.json'), continuous_angles_preserved=True,
                      actual_new_pose_loads_recomputed=True, working_area_count=13)
    write_json(out/'validation.json', validation)
    # Expose the new Step3 evidence next to the existing four-region run.
    shutil.copytree(input_root/object_name/'step3', output_root/object_name/'step3'/tag, dirs_exist_ok=True)
    from .render_families import render
    render(model, item.layout, full, out)
    write_json(out/'run_status.json', dict(status='pass', seconds=time.monotonic()-began))
    print('GROUPED13 PASS', dict(seconds=round(time.monotonic()-began, 2),
        material_cm3=metrics['full_material_volume_cm3'],
        occupied_xy_cm2=metrics['occupied_xy_bbox_area_cm2'],
        angles_deg=item.layout.angles_deg.tolist()), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=FAMILY_ROOT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--object', default='B'); p.add_argument('--tag', default='grouped13')
    p.add_argument('--height', type=float, default=0.)
    p.add_argument('--resume', type=Path)
    p.add_argument('--groups', type=int, choices=[3,4,5], default=5)
    a = p.parse_args()
    try:
        run(a.input_root, a.output_root, a.object, a.tag, a.height, a.resume, a.groups)
    except Exception as error:
        out = a.output_root/a.object/'step4'/a.tag; out.mkdir(parents=True, exist_ok=True)
        write_json(out/'run_status.json', dict(status='unresolved', error=f'{type(error).__name__}: {error}'))
        raise
