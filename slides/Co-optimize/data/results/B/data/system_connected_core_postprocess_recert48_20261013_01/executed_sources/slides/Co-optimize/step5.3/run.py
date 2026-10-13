"""Construct/render the final support from certified Step5.2-pruned material."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import stage5_common as C
from stage5_batch import run_cases
spec = importlib.util.spec_from_file_location('final_step53_render', Path(__file__).with_name('render.py'))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def worker(case, args):
    inputs = C.Inputs(args.object, args.selection_summary, args.step51_experiment)
    data = inputs.load(case)
    connected_policy=C.connected_core_policy(data)
    pruned_root = inputs.base/case['id']/'step5/step5.2'/(args.prune_experiment or args.experiment)
    prune_path = pruned_root/'data/report.json'
    prune = json.loads(prune_path.read_text())
    if (not prune['complete'] or not prune['passed'] or prune.get('stage_type')!='support_pruning'
            or prune['selected_step4_source']!=C.relative(data['selected'])):
        raise ValueError('Certified pruning for this exact saved layout is required')
    inputs.protected[C.relative(prune_path)] = C.digest(prune_path)
    for name, expected in prune['artifacts'].items():
        path = pruned_root/name
        if C.digest(path)!=expected:
            raise ValueError('Pruned input changed: '+name)
        inputs.protected[C.relative(path)] = expected
    witness_path = data['selected']/'oracle_connected_witness.obj'
    inputs.protected[C.relative(witness_path)] = C.digest(witness_path)
    witness = C.trimesh.load(witness_path, force='mesh', process=False)
    data['fixture'] = C.trimesh.load(pruned_root/prune['support'], force='mesh', process=False)
    out = inputs.base/case['id']/'step5/step5.3'/args.experiment
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    sys.path.insert(0,str(C.COOPT/'step5.2'))
    from prune_support import ForceCertificate
    from whole_search.model import Layout
    with C.np.load(data['selected']/'layout.npz') as arrays:
        layout=Layout.from_arrays(arrays)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],args.object)
    if args.leg_profile=='flat':
        from flat_tapered_base import Builder
        reference_path, reference_report = C.constructed_reference(inputs, data, args)
        reference = C.trimesh.load(reference_path, force='mesh', process=False)
        builder=Builder(data['fixture'], witness, data['rows'], reference,reference_report['construction'],
            object_mesh=data['body'],object_placements=layout.placements[list(layout.active)],
            require_connected_core=connected_policy)
        mesh, construction = builder.run()
    else:
        sys.path.insert(0, str(C.COOPT/'step4.2'))
        from material_builder_bounded import Builder
        rows = [dict(row,object_vertices=C.posed(data['body'],row['object_transform']).vertices)
                for row in data['rows']]
        builder=Builder(data['fixture'], witness, rows)
        mesh, construction = builder.run()
    if not construction['passed'] or not construction['watertight']:
        raise ValueError('Final enclosure, connectivity or closed-boundary construction failed')
    construction['connected_core_policy']=connected_policy
    force=checker.evaluate(mesh)
    if not force['passed']:
        # Coincident unions can bury a required contact under a tiny intrusion
        # into the actual workpiece. Delete that material, exposing the true
        # object boundary, and certify the resulting contacts from scratch.
        from contact_regularization import remove_object_interiors
        from deterministic_base import solid,unpack,volume,components,enclosure
        from material_builder_closed import regularize_external_boundaries
        cleaned,cleanup=remove_object_interiors(mesh,data['body'],layout.placements[list(layout.active)])
        value,discarded=regularize_external_boundaries(solid(cleaned))
        cleaned=unpack(value)
        cover=enclosure(cleaned,data['rows'])
        outside=volume(value-builder.allowed)
        retained=volume(builder.legal_nominal-value)
        if (not cleaned.is_watertight or len(components(value))!=1 or outside>1e-10
                or retained>1e-10 or not all(row['passed']for row in cover)):
            raise ValueError('Object-interior cleanup failed original geometry/retention checks')
        force=checker.evaluate(cleaned)
        if force['passed']:
            mesh=cleaned
            construction.update(per_pose=cover,material_cm3=volume(value)*1e6,
                added_material_cm3=volume(value-builder.original)*1e6,
                outside_saved_legal_component_cm3=outside*1e6,
                retained_legal_step4_material_loss_cm3=retained*1e6,
                actual_object_interior_cleanup=cleanup,
                final_boundary_regularizations=discarded)
    if not force['passed']:
        raise ValueError('Constructed contacts do not preserve all original demands: '+str(force['counts']))
    from export_stability import stabilize_export
    from deterministic_base import solid,volume
    mesh,force,export_check=stabilize_export(mesh,witness,data['body'],
        layout.placements[list(layout.active)],data['rows'],checker,final_support=True)
    construction.update(material_cm3=volume(solid(mesh))*1e6,
        per_pose=export_check['pressure_enclosure'],connected=True,watertight=True,
        serialized_export_verification=export_check,
        outside_saved_legal_component_cm3=export_check['history'][-1]['outside_saved_witness_cm3'],
        actual_object_overlap_cm3=export_check['history'][-1]['actual_object_overlap_cm3'])
    mesh.export(out/'support_with_base.obj', digits=16)
    visual = C.A.VISUALS/args.object/case['id']/'step5.3'
    visual.mkdir(parents=True, exist_ok=True)
    renderer = C.V.Renderer()
    R.reference(renderer, data, visual/'overview.png', fixture=mesh)
    tiles = []
    for row in data['rows']:
        tile, _ = C.H.render_pose(renderer,
            C.posed(data['body'],row['object_transform']), C.posed(mesh,row['fixture_transform']),
            row['cloud'],row['color'],visual/f'{row["pose"]}.png',hull=row['hull'],size=700)
        tiles.append(tile)
    C.V.sheet(tiles, visual/'all_poses.png')
    inputs.verify()
    report = dict(complete=True, passed=True, stage_type='final_support_construction',
        object=args.object, id=case['id'], poses=data['record']['poses'], pose_count=case['pose_count'],
        selected_step4_experiment=case['experiment'], selected_step4_source=C.relative(data['selected']),
        step52_source=C.relative(pruned_root), support='support_with_base.obj', construction=construction,
        pruned_material_cm3=prune['pruning']['retained_material_cm3'], leg_profile=args.leg_profile,
        force_passed_reused_from_step52=False,force_solver_run=True,force_counts=force['counts'],
        step4_rerun=False,
        step51_rerun=False, full_ground_friction_yaw_solved=False, fixture_weight_ratio=0.,
        published_images={path.name:C.digest(path) for path in visual.glob('*.png')},
        seconds=time.monotonic()-began, provenance=dict(inputs=inputs.protected,
            code={C.relative(path):C.digest(path) for path in [Path(__file__),
                Path(__file__).with_name('deterministic_base.py'),
                Path(__file__).with_name('flat_tapered_base.py'),
                C.COOPT/'helper_func/contact_regularization.py',
                C.COOPT/'helper_func/export_stability.py',
                C.COOPT/'step5.2/prune_support.py'] } ))
    C.save(out/'data/report.json',report)
    print('STEP5.3',case['id'],construction['material_cm3'],'cm3',flush=True)
    return dict(id=case['id'], passed=True, pose_count=case['pose_count'],
        material_cm3=construction['material_cm3'], output=C.relative(out), protected=inputs.protected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prune-experiment')
    parser.add_argument('--step51-experiment', default='material_output_all52_20261012_01')
    parser.add_argument('--reference-experiment', default='material_output_all52_20261012_final')
    parser.add_argument('--reference-stage', choices=['step4','step5.2','step5.3'], default='step5.2')
    parser.add_argument('--leg-profile', choices=['flat','original'], default='flat')
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--batch-name')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    args.selection_summary = args.selection_summary or C.A.result_root(args.object)/'data/material_output_all52_20261012_01/batch.json'
    inputs = C.Inputs(args.object,args.selection_summary,args.step51_experiment)
    cases = inputs.cases
    if args.sets:
        missing = set(args.sets)-{case['id'] for case in cases}
        if missing:parser.error('Unknown sets: '+str(sorted(missing)))
        cases = [case for case in cases if case['id'] in args.sets]
    directory = inputs.base/'data'/(args.batch_name or 'step53_'+args.experiment)
    directory.mkdir(parents=True,exist_ok=False)
    results = run_cases(cases,args,worker,directory,inputs.base,'step5.3','final_support_construction','cases')
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


if __name__=='__main__':main()
