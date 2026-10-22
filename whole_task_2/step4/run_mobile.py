"""Certify movable seats, with bounded proposals then original population search."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .mobile import MobileTaskModel
from .search import context
from .export_geometry import condition_core
from .complete_gate import MobileEvaluator
from .station_seed import precheck
from whole_task_2.step1.common import DEFAULT_OUTPUT, write_json, sha256
from whole_task_2.step3.legacy import kernel as K, loaded_sources
from whole_task_2.common.wrenches import endpoint_demands
from whole_task_2.common.force_cone import certificate_description
from population_soft import PopulationSoftSearch
from material_force import certify_material_contacts
from whole_search.common import result_mesh
from search_runtime import SearchRuntime, execution_scope


def run(output_root=DEFAULT_OUTPUT, name='B', tag='movable_station', rounds=12, population=32, seed=42,
        station_precheck=True):
    began = time.monotonic()
    out = Path(output_root).resolve()/name/'step4'/tag
    if (out/'report.json').exists():
        raise ValueError('Run already exists; choose a new --tag to preserve its evidence: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'run_status.json', dict(status='loading'))
    print('Loading movable-station tasks', flush=True)
    model = MobileTaskModel(name=name, output_root=output_root)
    loading_seconds = time.monotonic()-began
    q, metadata = model.initial()
    options = dict(rounds=rounds, population=population, seed=seed, tag=tag,
        station_precheck=station_precheck, station_precheck_maximum_trials=25,
        execution='fixed_work_direction_mobile_window_COM_xy_recenter',
        floor_pressure_envelope='conservative_local_force_cone_outer_hull',
        continuous_load_cover=certificate_description(model.tasks[0].domain),
        floor_projection_input_count=sum(len(x) for x in model.base_oracle.loads),
        full_original_force_sample_count=sum(len(t.targets) for t in model.tasks),
        object_floor_model=K.FLOOR.description())
    runtime = SearchRuntime(out/'runtime', execution_scope(model, options), memory_bytes=64*1024**2)
    model.search_runtime = runtime
    second = out/'step4.2'
    s = context(model, second/'search', iterations=rounds, population=population, seed=seed)
    s.allow_joint_reseating = True
    s.allow_height_repair = True
    search = PopulationSoftSearch(s)
    search.evaluator = MobileEvaluator(s)
    write_json(out/'run_status.json', dict(status='searching', options=options, loading_seconds=loading_seconds))
    print('Loaded in', round(loading_seconds, 2), 's; evaluating grounded seed', flush=True)
    initial = model.evaluate(q)
    write_json(out/'initialization.json', dict(metadata=metadata, counts=initial['counts'],
        construction='original PopulationSoftSearch connected nominal core',
        extra_one_percent_padded_initialization_used=False))
    print('Initial counts', initial['counts'], flush=True)
    selected, precheck_rows = precheck(search, initial) if station_precheck else (None, [])
    if selected is None:
        search.solve(initial)
        selected = s.material_selected
    else:
        s.material_selected = selected
        s.frontier_statistics = dict(policy='bounded_shared_installation_precheck',
            precheck_trials=len(precheck_rows), original_population_search_run=False,
            configured_rounds=rounds, configured_population=population,
            stopped_on_first_complete_solid=True, statistics=dict(search.evaluator.stats))
    search_seconds=time.monotonic()-began-loading_seconds
    print('Search finished; complete=', selected.completed, 'counts=', selected.result['counts'], flush=True)
    np.savez_compressed(second/'layout.npz', **selected.layout.arrays(model.native))
    core = selected.result.get('remaining_mesh')
    if core is None:
        from connected_core import prepare
        core, _ = prepare(model, selected.layout)
    K.D.export_exact_obj(core, second/'support.obj')
    if selected.witness is not None:
        K.D.export_exact_obj(selected.witness, second/'legal_component.obj')
    if selected.solid is not None:
        K.D.export_exact_obj(selected.solid, second/'complete_support.obj')
    conditioning={}
    for label,filename in [('core','support.obj'),('complete','complete_support.obj')]:
        path=second/filename
        if path.exists():
            mesh=K.trimesh.load(path,force='mesh',process=False)
            _,conditioning[label]=condition_core(model,selected.layout,mesh,path)
    write_json(second/'oracle.json', selected.oracle)
    write_json(second/'construction.json', selected.construction or {})
    write_json(second/'search_summary.json', s.frontier_statistics)
    execution = [model.execution(selected.layout, k) for k in selected.layout.active]
    write_json(out/'execution.json', dict(schema='centered_mobile_station_v1', poses=execution,
        centering='Object COM XY fixed at station center (0,-0.5); height chosen by search',
        equivalent_under_horizontal_rigid_translation=True,
        per_pose_support_translations_allowed=True, ground_model=K.FLOOR.description(),
        ik_certified=False, upstream_geometry_exceptions_retained=True))
    # Reopen exported geometry, re-extract its real contact polygons and classify
    # both the original sample family and every continuous-domain endpoint.
    checks = {}
    for label, filename in [('core','support.obj'), ('complete','complete_support.obj')]:
        path = second/filename
        if not path.exists(): continue
        mesh = K.trimesh.load(path, force='mesh', process=False)
        proof = certify_material_contacts(model, selected.layout, mesh)
        endpoints = {}
        arrays = {}
        for k in selected.layout.active:
            target = endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale
            mask, _ = K.J.classify(proof['supplies'][k], target)
            endpoints[model.poses[k]] = dict(passed=int(mask.sum()), total=len(mask))
            arrays.update({f'mask_{k}':proof['masks'][k], f'supply_{k}':proof['supplies'][k], f'endpoint_mask_{k}':mask})
        np.savez_compressed(second/f'{label}_force_certificate.npz', **arrays)
        checks[label] = dict(counts={model.poses[k]:int(v.sum()) for k,v in proof['masks'].items()},
            endpoints=endpoints, watertight=bool(mesh.is_watertight), volume_cm3=abs(float(mesh.volume))*1e6)
    completed = bool(selected.completed and all(v==len(model.tasks[k].targets) for k,v in enumerate(checks['complete']['counts'].values()))
        and all(r['passed']==r['total'] for r in checks['complete']['endpoints'].values())) if 'complete' in checks else False
    validation=None
    if completed:
        from .validate_mobile import validate_artifacts
        print('Replaying complete solids, finite work access, full exits and centered execution',flush=True)
        validation=validate_artifacts(model,out,selected.layout)
    write_json(out/'runtime_statistics.json', runtime.report())
    report = dict(schema='whole_task_mobile_station_step4_v1', status='pass' if completed else 'search_budget_exhausted',
        full_fixture_accepted=completed, poses=model.poses, object=name, output_root=str(Path(output_root).resolve()),
        policy=options, original_search='PopulationSoftSearch with all current reference proposal families',
        original_population_search_run=not bool(precheck_rows and precheck_rows[-1]['completed']),
        selection_stage='station_precheck' if precheck_rows and precheck_rows[-1]['completed'] else 'population_search',
        precheck_trials=len(precheck_rows),
        checks=checks, construction=selected.construction, export_conditioning=conditioning, evaluation=selected.evaluation,
        seconds=time.monotonic()-began, loading_seconds=loading_seconds, search_seconds=search_seconds,
        extra_one_percent_padded_initialization_used=False,
        geometry_contract='current original Step4.2: actual connected complete solid; nominal complete exits; finite work exclusions',
        source_geometry_review_required=model.upstream_report['source_geometry_review_required'],
        input_sha256={str(p):sha256(p) for p in set(model.inputs+loaded_sources())})
    write_json(out/'report.json', report)
    if validation is not None:
        validation['report_sha256']=sha256(out/'report.json')
        write_json(out/'validation.json',validation)
    write_json(out/'run_status.json', dict(status=report['status'], seconds=report['seconds']))
    if completed:runtime.cleanup()
    print('RESULT', report['status'], checks, 'seconds', report['seconds'], flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--object', default='B'); p.add_argument('--tag', default='movable_station')
    p.add_argument('--rounds', type=int, default=12); p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-station-precheck', action='store_true',
                   help='Run the original cold population directly, without bounded seat proposals')
    a=p.parse_args()
    try:
        run(a.output_root, a.object, a.tag, a.rounds, a.population, a.seed, not a.no_station_precheck)
    except Exception as error:
        folder=a.output_root/a.object/'step4'/a.tag
        if folder.exists() and not (folder/'report.json').exists():
            write_json(folder/'run_status.json',dict(status='unresolved',error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':main()
