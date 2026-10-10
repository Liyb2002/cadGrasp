"""Recheck/export the least-material complete feasible checkpoint of a search.

This does not construct geometry or change a layout. It recovers results from
an interrupted/export-failed run and distinguishes selection from new search.
"""
import argparse
import hashlib
import json
import shutil
from common import *
from model import Model, Layout
from classify import classify
from case_sets import CASES
from reuse_first import registered, juxtaposed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--mode', choices=['joint', 'incremental'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.run.resolve()
    case = source/args.case
    history = case/args.mode
    manifest = json.loads((source/'runtime_sources/manifest.json').read_text())
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for relative, expected in manifest['startup_sources'].items():
        assert digest(source/'runtime_sources'/relative) == expected
    model = Model([f'pose_{k}' for k in CASES[args.case]])
    # Independent existing B evidence anchors the immutable original inputs.
    original = json.loads((HERE/'output/ten_v10/pose1-10/joint/report.json').read_text())
    expected_inputs = original['provenance']['inputs']
    for path in model.inputs:
        relative = str(Path(path).resolve().relative_to(ROOT))
        assert relative in expected_inputs and digest(path) == expected_inputs[relative]
    stages = []
    if args.mode == 'incremental':
        stages = json.loads((history/'insertion_stages.json').read_text())
        assert len(stages) == len(model.poses) and all(row['passed'] for row in stages)
    feasible = []
    for path in (case/'exact_states').glob('feasible_*.npz'):
        with np.load(path) as data:
            if tuple(data['active']) != tuple(range(len(model.poses))):
                continue
            if not all(data[f'mask_{k}'].all() for k in range(len(model.poses))):
                continue
            solid = C.S.solid(C.trimesh.Trimesh(data['vertices'], data['faces'], process=False))
            feasible.append((C.material_volume(solid)*1e6, path))
    assert feasible, 'No complete, fully feasible checkpoint exists'
    feasible.sort(key=lambda row:(row[0], int(row[1].stem.split('_')[1])))
    volume, path = feasible[0]
    with np.load(path) as data:
        layout = Layout(data['placements'].copy(), data['directions'].copy(), data['hosts'].copy(),
                        tuple(map(int, data['active'])))
        actual = C.trimesh.Trimesh(data['vertices'].copy(), data['faces'].copy(), process=False)
    material = C.S.solid(actual)
    masks = {};supplies = {};contacts = {};classifiers = {};footprints = []
    for k in layout.active:
        q = layout.placements[k]
        world = model.native[layout.hosts[k]] @ q
        np.testing.assert_allclose(world[:3, :3], model.native[k, :3, :3], atol=1e-10, rtol=0)
        assert abs(world[2, 3]-model.native[k, 2, 3]) < 1e-9
        direction = q[:3, :3].T @ layout.directions[k]
        allowed = model.allowed[k][model.mesh.face_normals[model.allowed[k]] @ direction <= 1e-9]
        tri, src = C.contact_boundary(transform_mesh(model.mesh, q), actual, allowed)
        supply = C.supply(model.tasks[k], model.native[k] @ np.linalg.inv(q), tri, src)
        mask, info = classify(supply, model.tasks[k].targets)
        assert mask.all(), 'Checkpoint lost an original load upon contact regeneration'
        masks[k] = mask;supplies[k] = supply;classifiers[k] = info
        contacts[k] = dict(triangles=tri, sources=src, triangle_count=len(tri),
                           area_m2=float(C.trimesh.triangles.area(tri).sum()))
        xy = C.transform_points(actual.vertices, model.native[layout.hosts[k]])[:, :2]
        footprints.append(float(ConvexHull(xy).volume))
        print('REGENERATED CONTACTS', model.poses[k], int(mask.sum()), flush=True)
    receipt = json.loads(path.with_suffix('.json').read_text())
    result = dict(layout=layout, serial=int(path.stem.split('_')[1]), remaining=material,
        masks=masks, supplies=supplies, contacts=contacts, classifiers=classifiers,
        counts={str(k):int(mask.sum()) for k, mask in masks.items()}, diagnostics=receipt['diagnostics'],
        volume_cm3=volume, maximum_projected_footprint_m2=max(footprints))
    result['actual_work_surface_checks'] = model.verify_work(result)
    assert all(row['passed'] for row in result['actual_work_surface_checks'])
    events = json.loads((history/'trace.json').read_text()) if (history/'trace.json').exists() else []
    own_sources = {relative:sha for relative,sha in manifest['startup_sources'].items()
                   if relative.startswith('slides/pose_set_search/code/')}
    finalizer_sources = model.startup_sources
    model.startup_sources = own_sources
    initial = json.loads((history/'initial_metrics.json').read_text())
    _, initialization = model.initial()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ['trace.json', 'initial_metrics.json', 'insertion_stages.json']:
        if (history/name).exists():
            shutil.copy2(history/name, args.out/name)
    report = model.save(result, args.out, dict(
        strategy=args.mode, policy='rotate-first-selective-juxtapose', initialization=initialization,
        events=events, initial_counts=initial['counts'],
        rotating_reuse_pose_count=sum(registered(layout, k) for k in layout.active),
        juxtaposed_pose_count=juxtaposed_count(layout),
        reuse_modes={model.poses[k]:('single_use_rotating_fixture' if registered(layout, k) else 'juxtaposed')
                     for k in layout.active},
        fixture_placement_count_is_not_cost=True, all_pose_juxtapose_disabled=True,
        objective='all original force/torque/exit/work constraints first, then actual material volume',
        final_selection='smallest actual material volume among complete feasible checkpoints',
        source_run=str(source.relative_to(ROOT)), source_checkpoint=str(path.relative_to(ROOT)),
        source_checkpoint_sha256=digest(path), checkpoint_contact_and_work_regenerated=True,
        feasible_checkpoints_examined=len(feasible),
        feasible_checkpoint_volumes_cm3=[dict(serial=int(p.stem.split('_')[1]), volume_cm3=v) for v,p in feasible],
        search_runtime_manifest=str((source/'runtime_sources/manifest.json').relative_to(ROOT)),
        finalizer_sources_sha256=finalizer_sources,
        resumed_layout=manifest['args'].get('start_layout'),
        source_construction=manifest['args'].get('start_result'),
        volume_descent_events=[row for row in events if row.get('phase') == 'feasible_volume_descent'],
        requested_pose_count=len(model.poses), inserted_pose_count=len(stages) if stages else len(model.poses),
        every_insertion_passed=all(row['passed'] for row in stages), insertion_stages=stages,
        separated_fallback_used=False, failure_scope='Finite search failure is not an infeasibility proof'))
    print('FINALIZED', args.case, args.mode, report['volume_cm3'], 'rotating',
          report['rotating_reuse_pose_count'], 'juxtaposed', report['juxtaposed_pose_count'], flush=True)


if __name__ == '__main__':
    main()
