"""Prune saved material before Step5.3 constructs the final connected support."""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np

import stage5_common as C
from stage5_batch import run_cases
from prune_support import ForceCertificate, Pruner,material_components
from whole_search.model import Layout

spec = importlib.util.spec_from_file_location('pruning_stage_render', C.COOPT/'step5.3/render.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def worker(case, args):
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    data = inputs.load(case)
    connected_policy=C.connected_core_policy(data)
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
    constructed, reference_report = C.constructed_reference(inputs, data, args)
    if args.source_contact_material=='constructed':
        if connected_policy:
            raise ValueError('Connected-core pruning must start from the certified contact core')
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
        from export_stability import stabilize_export
        fixture,_,initial_export_stability=stabilize_export(fixture,witness,data['body'],
            layout.placements[list(layout.active)],data['rows'],certificate,final_support=False)
    baseline = certificate.evaluate(legal_baseline(fixture))
    if connected_policy and not baseline['passed']:
        raise ValueError('Saved connected Step4 core fails its actual original demands: '+str(baseline['counts']))
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
    shape_reference=None
    reference_root=getattr(args,'shape_reference',None)
    if reference_root:
        reference_root=Path(reference_root)
        reference_path=reference_root/'data/report.json'
        reference_record=json.loads(reference_path.read_text())
        if not reference_record['passed']or reference_record['selected_step4_source']!=C.relative(data['selected']):
            raise ValueError('Rounded shape reference belongs to another or failed layout')
        support_path=reference_root/reference_record['support']
        if C.digest(support_path)!=reference_record['artifacts'][reference_record['support']]:
            raise ValueError('Rounded shape reference changed')
        for path in [reference_path,support_path]:inputs.protected[C.relative(path)]=C.digest(path)
        shape_reference=C.trimesh.load(support_path,force='mesh',process=False)
    pruning_style=getattr(args,'pruning_style','rounded')
    pruner_class=Pruner;extra={}
    if pruning_style=='rounded':
        from rounded_pruning import RoundedPruner
        pruner_class=RoundedPruner
        extra=dict(reference=shape_reference,rounding_radius=getattr(args,'rounding_radius',.006))
    pruner=pruner_class(fixture,witness,certificate,budget=args.candidate_budget,
        rows=data['rows'],object_vertices=data['body'].vertices,require_connected=connected_policy,**extra)
    mesh, pruning, proof = pruner.run()
    if interior_cleanup:
        pruning['initial_actual_object_interior_cleanup']=interior_cleanup
    if requested_cleanup:
        pruning['requested_actual_object_interior_cleanup']=requested_cleanup
    from export_stability import stabilize_export
    mesh,proof,export_check=stabilize_export(mesh,witness,data['body'],
        layout.placements[list(layout.active)],data['rows'],certificate,final_support=False)
    pruning.update(retained_material_cm3=volume(solid(mesh))*1e6,
        serialized_export_verification=export_check,
        outside_saved_legal_component_cm3=export_check['history'][-1]['outside_saved_witness_cm3'],
        actual_object_overlap_cm3=export_check['history'][-1]['actual_object_overlap_cm3'],
        component_count=len(material_components(solid(mesh))),
        connected=len(material_components(solid(mesh)))==1,
        force_required_isolated_components=max(0,len(material_components(solid(mesh)))-1))
    pruning['retained_bbx_l']=pruner.bbx(solid(mesh))
    if connected_policy and (not pruning['connected'] or
            volume(solid(mesh)-solid(fixture))>1e-10 or
            pruning['retained_bbx_l']>pruning['input_bbx_l']+1e-8):
        raise ValueError('Serialized deletion must preserve connectivity, subset and nonincreasing BBX')
    if shape_reference is not None:
        actual_bounds=C.occupied_bounds(data['body'],mesh,data['rows'])
        expected_bounds=C.occupied_bounds(data['body'],shape_reference,data['rows'])
        if not np.allclose(actual_bounds,expected_bounds,atol=1e-8,rtol=0):
            raise ValueError('Final rounded export changed the locked pruning BBX')
        if abs(C.volume(actual_bounds)-C.volume(expected_bounds))*1000>1e-8:
            raise ValueError('Final rounded export changed the locked pruning BBX volume')
        if volume(solid(shape_reference)-solid(mesh))>1e-10:
            raise ValueError('Rounded export lost certified reference material')
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
    report = dict(complete=True, passed=pruning['passed'], stage_type='support_pruning',connected_core_policy=connected_policy,
        object=args.object, id=case['id'], poses=data['record']['poses'], pose_count=case['pose_count'],
        selected_step4_experiment=case['experiment'], selected_step4_source=C.relative(data['selected']),
        step51_source=C.relative(inputs.base/case['id']/'step5/step5.1'/args.step51_experiment),
        source_contact_material=C.relative(source), support='pruned_support.obj', pruning=pruning,
        pruning_style=pruning_style,shape_reference=C.relative(reference_root)if reference_root else None,
        pruning_continuation=args.resume_pruning_experiment,
        requested_source_contact_material=args.source_contact_material,
        force_passed=True, force_solver_run=True, original_demand_count_per_pose=32768,
        step4_rerun=False, step51_rerun=False, layout_changed=False,
        final_base_constructed=False, published_images=images,
        provenance=dict(inputs=inputs.protected, code={C.relative(path):C.digest(path) for path in
            [Path(__file__), Path(__file__).with_name('prune_support.py'),
             C.COOPT/'helper_func/contact_regularization.py',
             C.COOPT/'helper_func/export_stability.py',Path(__file__).with_name('rounded_pruning.py')]}),
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
    parser.add_argument('--reference-stage',choices=['step4','step5.2','step5.3'],default='step5.2')
    parser.add_argument('--clean-object-interiors',action='store_true')
    parser.add_argument('--pruning-style',choices=['rounded','legacy'],default='rounded')
    parser.add_argument('--shape-reference',type=Path)
    parser.add_argument('--rounding-radius',type=float,default=.006)
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
