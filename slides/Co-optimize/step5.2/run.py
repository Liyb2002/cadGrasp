"""Construct final connected material from Step5.1 and Step4's legal witness."""
from pathlib import Path
import argparse, json, time, traceback, multiprocessing, importlib.util, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import stage5_common as C
from deterministic_base import Builder
spec=importlib.util.spec_from_file_location('coopt_final_base_render',Path(__file__).with_name('render.py'))
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def worker(job):
    case,options=job
    inputs=C.Inputs(options['object'],Path(options['selection']),options['step51_experiment'])
    out=inputs.base/case['id']/'step5/step5.2'/options['experiment']
    out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    began=time.monotonic()
    try:
        data=inputs.load(case)
        oracle_path=data['selected']/'base_oracle.json'
        witness_path=data['selected']/'oracle_connected_witness.obj'
        oracle=json.loads(oracle_path.read_text())
        if not oracle['passed'] or not oracle['connected_passed']:raise ValueError('Step4 legal connection witness missing')
        for p in [oracle_path,witness_path]:inputs.protected[C.relative(p)]=C.digest(p)
        witness=C.trimesh.load(witness_path,force='mesh',process=False)
        prebuilt_path=data['selected']/'preconstructed_base.json'
        construction_reused=False
        material_rebuilt=case['id'] in options.get('rebuild_material_for',[])
        if prebuilt_path.exists() and not material_rebuilt:
            prebuilt=json.loads(prebuilt_path.read_text())
            if (not prebuilt['complete'] or not prebuilt['construction']['passed'] or
                    prebuilt['fixture_weight_ratio']!=data['record']['fixture_weight_ratio']):
                raise ValueError('Preconstructed base acceptance/weight model differs')
            inputs.protected[C.relative(prebuilt_path)]=C.digest(prebuilt_path)
            for name,expected in prebuilt['source_sha256'].items():
                path=data['selected']/name
                if C.digest(path)!=expected:
                    raise ValueError('Preconstructed base input changed: '+name)
                inputs.protected[C.relative(path)]=expected
            mesh=C.trimesh.load(data['selected']/prebuilt['support'],force='mesh',process=False)
            construction=prebuilt['construction'];construction_reused=True
        elif material_rebuilt:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
            from material_builder_bounded import Builder as MaterialBuilder
            rows=[dict(row,object_vertices=C.posed(data['body'],row['object_transform']).vertices)
                  for row in data['rows']]
            mesh,construction=MaterialBuilder(data['fixture'],witness,rows).run()
        else:
            mesh,construction=Builder(data['fixture'],witness,data['rows']).run()
        mesh.export(out/'support_with_base.obj',digits=16)
        if not construction['passed']:raise ValueError('Deterministic construction did not satisfy enclosure/connectivity')
        renderer=C.V.Renderer()
        R.reference(renderer,data,out/'overview.png',fixture=mesh)
        tiles=[]
        for row in data['rows']:
            T=row['fixture_transform']
            tile,_=C.H.render_pose(renderer,C.posed(data['body'],row['object_transform']),C.posed(mesh,T),
                row['cloud'],row['color'],out/f'{row["pose"]}.png',hull=row['hull'],size=700)
            tiles.append(tile)
        C.V.sheet(tiles,out/'all_poses.png')
        inputs.verify()
        report=dict(complete=True,object=options['object'],id=case['id'],poses=data['record']['poses'],
            pose_count=case['pose_count'],selected_step4_experiment=case['experiment'],
            selected_step4_source=C.relative(data['selected']),
            step51_source=C.relative(inputs.base/case['id']/'step5/step5.1'/options['step51_experiment']),
            original_step4_material_cm3=case['volume_cm3'],construction=construction,
            passed=construction['passed'],support='support_with_base.obj',
            force_passed_reused=True,force_solver_run=False,step4_rerun=False,
            full_ground_friction_yaw_solved=False,fixture_weight_ratio=data['record']['fixture_weight_ratio'],
            construction_reused_from_step4=construction_reused,
            material_reconstructed_in_saved_witness=material_rebuilt,
            seconds=time.monotonic()-began,provenance=dict(inputs=inputs.protected),
            artifacts={str(p.relative_to(out)):C.digest(p) for p in out.iterdir() if p.is_file()})
        C.save(out/'data/report.json',report)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1250px;margin:24px auto}img{width:100%}</style>'
            f'<h1>{case["id"]} · Step5.2</h1><p>Connected material: {construction["material_cm3"]:.2f} cm³; '
            f'added {construction["added_material_cm3"]:.2f} cm³. Blue: final solid; grey: object; colored polygons: ground demand hulls.</p>'
            '<p><a href="support_with_base.obj">Final OBJ</a> · <a href="data/report.json">Construction record</a></p>'
            '<img src="overview.png"><img src="all_poses.png">\n')
        result=dict(id=case['id'],passed=True,pose_count=case['pose_count'],
            material_cm3=construction['material_cm3'],added_material_cm3=construction['added_material_cm3'],
            removed_illegal_nominal_material_cm3=construction['removed_illegal_nominal_material_cm3'],
            seconds=report['seconds'],output=C.relative(out),protected=inputs.protected)
    except Exception as error:
        traceback.print_exc();result=dict(id=case['id'],passed=False,error=str(error),seconds=time.monotonic()-began)
    C.A.publish_visuals(out)
    if result['passed'] and options['object']=='B':
        from publish_output_images import publish_case_images
        publish_case_images(case['id'],options['experiment'])
    C.A.cleanup_stage(out)
    C.save(out/'run_result.json',result)
    print('STEP5.2',case['id'],result['passed'],round(result['seconds'],2),result.get('material_cm3',result.get('error')),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--selection-summary',type=Path)
    parser.add_argument('--experiment',default='connected_base_v1')
    parser.add_argument('--step51-experiment')
    parser.add_argument('--batch-name',default='step52_connected_base_v1')
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--rebuild-material-for',nargs='+',default=[],
                        help='Reconstruct these selected solids inside their saved legal witness; retain the layout and force masks')
    args=parser.parse_args()
    selection=C.A.resolve_artifact(args.selection_summary or C.A.result_root(args.object)/'data/base_oracle_results/summary.json')
    step51_experiment=args.step51_experiment or args.experiment
    inputs=C.Inputs(args.object,selection,step51_experiment)
    cases=inputs.cases
    if set(args.rebuild_material_for)-{r['id']for r in cases}:parser.error('Unknown reconstruction set')
    if args.sets:
        if set(args.sets)-{r['id'] for r in cases}:parser.error('Unknown set')
        cases=[r for r in cases if r['id'] in args.sets]
    for case in cases:
        out=inputs.base/case['id']/'step5/step5.2'/args.experiment
        if out.exists():parser.error(f'Existing output is protected: {out}')
    directory=inputs.base/'data'/args.batch_name;directory.mkdir(parents=True,exist_ok=False)
    code=C.begin(directory,'step5.2',inputs,cases)
    options=dict(object=args.object,selection=str(selection),experiment=args.experiment,
                 step51_experiment=step51_experiment,rebuild_material_for=args.rebuild_material_for)
    results=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(case,options)) for case in cases]
        for future in as_completed(futures):
            result=future.result();results.append(result);inputs.protected.update(result.get('protected',{}))
            C.save(directory/'progress.json',dict(complete=False,completed=len(results),passed=sum(r['passed'] for r in results),cases=results))
    summary=dict(cases=results,group_count=len(results),passed=sum(r['passed'] for r in results),
        pose_instance_count=sum(r['pose_count'] for r in results if r['passed']),
        material_cm3=sum(r.get('material_cm3',0) for r in results),
        added_material_cm3=sum(r.get('added_material_cm3',0) for r in results),
        seconds=time.monotonic()-began,force_solver_run=False,step4_rerun=False)
    C.finish(directory,inputs,code,summary)
    cards=''.join(f'<section><h2>{r["id"]}</h2><p><a href="{r["id"]}/step5/step5.2/{args.experiment}/index.html">Per-pose views and OBJ</a></p>'
        f'<img src="{r["id"]}/step5/step5.2/{args.experiment}/overview.png"></section>' for r in sorted(results,key=lambda r:(-r.get('pose_count',0),r['id'])) if r['passed'])
    (inputs.base/f'step5.2_{args.experiment}_index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1250px;margin:24px auto}img{width:100%}section{margin:30px 0}</style>'
        f'<h1>Step5.2 · {summary["passed"]}/{len(results)} connected supports</h1>'+cards+'\n')
    C.A.publish_gallery(inputs.base/f'step5.2_{args.experiment}_index.html')
    for case in cases:
        (inputs.base/case['id']/'step5/step5.2'/args.experiment/'run_result.json').unlink(missing_ok=True)
    print('STEP5.2 ALL',summary['passed'],'/',len(results),'seconds',round(summary['seconds'],2),flush=True)
    if summary['passed']!=len(results):raise SystemExit(1)


if __name__=='__main__':main()
