"""Prune saved material before Step5.3 constructs the final connected support."""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np

import stage5_common as C
from stage5_batch import run_cases
from prune_support import ForceCertificate, Pruner
from whole_search.model import Layout

spec = importlib.util.spec_from_file_location('pruning_stage_render', C.COOPT/'step5.3/render.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def worker(case, args):
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    data = inputs.load(case)
    out = inputs.base/case['id']/'step5/step5.2'/args.experiment
    out.mkdir(parents=True, exist_ok=False)
    visual = C.A.VISUALS/args.object/case['id']/'step5.2'
    visual.mkdir(parents=True, exist_ok=True)
    witness_path = data['selected']/'oracle_connected_witness.obj'
    inputs.protected[C.relative(witness_path)] = C.digest(witness_path)
    witness = C.trimesh.load(witness_path, force='mesh', process=False)
    with np.load(data['selected']/'layout.npz') as arrays:
        layout = Layout.from_arrays(arrays)
    certificate = ForceCertificate(data['record']['poses'], layout, data['body'], data['rows'], args.object)
    source = data['selected']/'support.obj'
    fixture = data['fixture']
    reference_root=inputs.base/case['id']/'step5'/args.reference_stage/args.reference_experiment
    reference_report_path=reference_root/'data/report.json'
    reference_report=json.loads(reference_report_path.read_text())
    constructed=reference_root/reference_report['support']
    inputs.protected[C.relative(reference_report_path)]=C.digest(reference_report_path)
    if args.source_contact_material=='constructed':
        source=constructed
        fixture=C.trimesh.load(source,force='mesh',process=False)
    parent_pruning = None
    if args.resume_pruning_experiment:
        previous_root=inputs.base/case['id']/'step5/step5.2'/args.resume_pruning_experiment
        parent_path=previous_root/'data/report.json'
        parent_pruning=json.loads(parent_path.read_text())
        if (not parent_pruning['passed'] or
                parent_pruning['selected_step4_source']!=C.relative(data['selected'])):
            raise ValueError('Pruning continuation belongs to a different or failed layout')
        source=previous_root/parent_pruning['support']
        if C.digest(source)!=parent_pruning['artifacts'][parent_pruning['support']]:
            raise ValueError('Pruning continuation geometry changed')
        inputs.protected[C.relative(parent_path)]=C.digest(parent_path)
        fixture=C.trimesh.load(source,force='mesh',process=False)
    from deterministic_base import solid,unpack,volume
    allowed=solid(witness)
    def legal_baseline(value):
        original=solid(value)
        return value if volume(original-allowed)<=1e-10 else unpack(original^allowed)
    requested_cleanup=None
    if args.clean_object_interiors:
        from contact_regularization import remove_object_interiors
        fixture,requested_cleanup=remove_object_interiors(fixture,data['body'],
            layout.placements[list(layout.active)])
    baseline = certificate.evaluate(legal_baseline(fixture))
    # Some saved searches certified their real constructed contact polygons,
    # rather than the grid/nominal shell. Use that actual accepted geometry
    # when necessary; never convert its one-load near miss into a nominal PASS.
    if not baseline['passed']:
        source = constructed
        fixture = C.trimesh.load(source, force='mesh', process=False)
        baseline=certificate.evaluate(legal_baseline(fixture))
        if not baseline['passed']:
            from contact_regularization import remove_object_interiors
            fixture,interior_cleanup=remove_object_interiors(fixture,data['body'],
                layout.placements[list(layout.active)])
            baseline=certificate.evaluate(legal_baseline(fixture))
            if not baseline['passed']:
                raise ValueError('Accepted constructed material fails full contact certification: '+str(baseline['counts']))
        else:
            interior_cleanup=None
    else:
        interior_cleanup=None
    inputs.protected[C.relative(source)] = C.digest(source)
    mesh, pruning, proof = Pruner(fixture, witness, certificate, budget=args.candidate_budget).run()
    if interior_cleanup:
        pruning['initial_actual_object_interior_cleanup']=interior_cleanup
    if requested_cleanup:
        pruning['requested_actual_object_interior_cleanup']=requested_cleanup
    if parent_pruning:
        previous=parent_pruning['pruning']
        pruning['cumulative_input_material_cm3']=previous.get('cumulative_input_material_cm3',previous['input_material_cm3'])
        pruning['cumulative_removed_material_cm3']=pruning['cumulative_input_material_cm3']-pruning['retained_material_cm3']
    mesh.export(out/'pruned_support.obj', digits=16)
    np.savez_compressed(out/'force_certificate.npz', **{
        **{f'mask_{k}':value for k,value in proof['masks'].items()},
        **{f'supply_{k}':value for k,value in proof['supplies'].items()}})
    renderer = C.V.Renderer()
    R.reference(renderer, data, visual/'overview.png', fixture=mesh)
    removed = C.trimesh.load(source, force='mesh', process=False)
    from deterministic_base import solid, unpack
    removed = unpack(solid(removed)-solid(mesh))
    T = data['states'][0]
    kept_world, removed_world = C.posed(mesh, T), C.posed(removed, T)
    camera = C.V.camera_for(np.vstack([C.posed(fixture, T).vertices, kept_world.vertices]))
    parts = [C.V.piece(kept_world, C.V.BLUE)]
    if len(removed_world.faces):
        parts.append(C.V.piece(removed_world, '#CB5B64', .48))
    renderer.render([C.V.CPU.placed(piece) for piece in parts], camera, 900).save(visual/'removed_material.png')
    tiles = []
    for row in data['rows']:
        tile, _ = C.H.render_pose(renderer,
            C.posed(data['body'], row['object_transform']), C.posed(mesh, row['fixture_transform']),
            row['cloud'], row['color'], visual/f'{row["pose"]}.png', hull=row['hull'], size=700)
        tiles.append(tile)
    C.V.sheet(tiles, visual/'all_poses.png')
    inputs.verify()
    images = {path.name:C.digest(path) for path in visual.glob('*.png')}
    report = dict(complete=True, passed=pruning['passed'], stage_type='support_pruning',
        object=args.object, id=case['id'], poses=data['record']['poses'], pose_count=case['pose_count'],
        selected_step4_experiment=case['experiment'], selected_step4_source=C.relative(data['selected']),
        step51_source=C.relative(inputs.base/case['id']/'step5/step5.1'/args.step51_experiment),
        source_contact_material=C.relative(source), support='pruned_support.obj', pruning=pruning,
        pruning_continuation=args.resume_pruning_experiment,
        requested_source_contact_material=args.source_contact_material,
        force_passed=True, force_solver_run=True, original_demand_count_per_pose=32768,
        step4_rerun=False, step51_rerun=False, layout_changed=False,
        final_base_constructed=False, published_images=images,
        provenance=dict(inputs=inputs.protected, code={C.relative(path):C.digest(path) for path in
            [Path(__file__), Path(__file__).with_name('prune_support.py'),
             C.COOPT/'helper_func/contact_regularization.py']}),
        artifacts={name:C.digest(out/name) for name in ['pruned_support.obj','force_certificate.npz']})
    C.save(out/'data/report.json', report)
    print('STEP5.2',case['id'],pruning['retained_material_cm3'],'cm3',pruning['component_count'],'components',flush=True)
    return dict(id=case['id'], passed=True, pose_count=case['pose_count'],
        retained_material_cm3=pruning['retained_material_cm3'], removed_material_cm3=pruning['removed_material_cm3'],
        component_count=pruning['component_count'], output=C.relative(out), protected=inputs.protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--step51-experiment', default='material_output_all52_20261012_01')
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--batch-name')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--candidate-budget', type=int, default=64)
    parser.add_argument('--resume-pruning-experiment')
    parser.add_argument('--source-contact-material',choices=['auto','constructed'],default='auto')
    parser.add_argument('--reference-experiment',default='material_output_all52_20261012_final')
    parser.add_argument('--reference-stage',choices=['step5.2','step5.3'],default='step5.2')
    parser.add_argument('--clean-object-interiors',action='store_true')
    args = parser.parse_args()
    args.selection_summary = args.selection_summary or C.A.result_root(args.object)/'data/material_output_all52_20261012_01/batch.json'
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    cases = inputs.cases
    if args.sets:
        missing = set(args.sets)-{case['id'] for case in cases}
        if missing:
            parser.error('Unknown sets: '+str(sorted(missing)))
        cases = [case for case in cases if case['id'] in args.sets]
    directory = inputs.base/'data'/(args.batch_name or 'step52_'+args.experiment)
    directory.mkdir(parents=True, exist_ok=False)
    results = run_cases(cases,args,worker,directory,inputs.base,'step5.2','support_pruning','results')
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
