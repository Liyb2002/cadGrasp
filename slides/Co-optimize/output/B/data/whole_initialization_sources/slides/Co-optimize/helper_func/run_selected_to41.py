"""Explicit downstream run from the merged whole-set initialization.

Old progress caches belong to the pre-merge geometry and are never resumed.
This command also runs Step3.3/Step4.1; use step3.1/run.py for initialization only.
"""
import _bootstrap
import argparse
import contextlib
import json
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from co_common import *
from codes.precompute_objects.dataset import read_selected_pose_groups
import run_all
import step33
import step41


def step3_case(args):
    name, group, thickness = args
    started = time.monotonic()
    row = run_all.run_case(args)
    if not row.get('step4_ready'):
        return dict(row, step33_status='initialization_unresolved')
    root = HERE/'output'/name if name == 'B' else HERE/'data/object_inputs'/name
    try:
        with (root/group['id']/'step3/data/pipeline.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = step33.run(name, group)
        return dict(row, step33_status='pass' if result['passed'] else 'unresolved', through_step33_seconds=time.monotonic()-started,
            initialization_schema=run_all.step31.SCHEMA,
            initialization_report_sha256=I.sha256(root/group['id']/'step3/step3.1/data/report.json'))
    except Exception as error:
        return dict(row, step33_status='unresolved', step33_error=f'{type(error).__name__}: {error}')


def step41_case(args):
    name, group, initialization = args
    root = HERE/'output'/name if name == 'B' else HERE/'data/object_inputs'/name
    try:
        with (root/group['id']/'step3/data/pipeline.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return step41.run(name, group, initialization)
    except Exception as error:
        traceback.print_exc()
        return dict(id=group['id'], status='unresolved', error=f'{type(error).__name__}: {error}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--objects',nargs='+');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args()
    names=args.objects or [p.parent.name for p in sorted((ROOT/'objects').glob('*/selected_pose_sets.json')) if p.parent.name!='B']
    for name in names:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            started=time.monotonic();groups=read_selected_pose_groups(name);rows=[]
            root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
            progress=root/'data/selected_stage3_whole_progress.json'
            if progress.exists():rows=json.loads(progress.read_text())
            rows=[r for r in rows if r.get('initialization_schema')==run_all.step31.SCHEMA
                  and (root/r['id']/'step3/step3.1/data/report.json').exists()
                  and r.get('initialization_report_sha256')==I.sha256(root/r['id']/'step3/step3.1/data/report.json')]
            completed={r['id'] for r in rows}
            for group in groups:(root/group['id']/'step3/data').mkdir(parents=True,exist_ok=True)
            for future in as_completed([pool.submit(step3_case,(name,group,.005)) for group in groups if group['id'] not in completed]):
                row=future.result();rows.append(row);save(progress,rows);print('STEP3 COMPLETE',name,row['id'],row['status'],row['step33_status'],round(row['seconds'],2),flush=True)
            save(root/'data/selected_step3_batch.json',dict(complete=True,object=name,results=rows,sets=len(rows),step33_passed=sum(r['step33_status']=='pass' for r in rows)))
            ready={r['id'] for r in rows if r['step33_status']=='pass'};selected=[g for g in groups if g['id'] in ready]
            prior=root/'data/selected_stage41_whole_progress.json'
            outputs=json.loads(prior.read_text()) if prior.exists() else []
            current=[]
            for row in outputs:
                path=root/row['id']/'step4/step4.1/data/report.json'
                try:
                    checked=I.check_report(path)
                    inputs=checked['provenance']['inputs'];contact=initialized_contacts(root/row['id'])
                    if inputs.get(str(contact.relative_to(ROOT)))==I.sha256(contact):current.append(row)
                except (OSError,RuntimeError,AssertionError,KeyError):pass
            outputs=current
            completed41={r['id'] for r in outputs};pending=[g for g in selected if g['id'] not in completed41]
            cached_path=root/'data/step41/initialization.json'
            cached=json.loads(cached_path.read_text()) if cached_path.exists() else None
            cache_valid=bool(cached and cached.get('initialization_schema')==run_all.step31.SCHEMA and all(g['id'] in cached.get('groups',{}) and all((root/'data/step41'/g['id']/p/f'{kind}_sweep.obj').exists() for p in g['poses'] for kind in ['full','display']) for g in pending))
            if cache_valid:
                cache_valid=all(I.sha256(root/'data/step41'/g['id']/p/f'{kind}_sweep.obj')==cached['artifacts'].get(f"{g['id']}/{p}/{kind}_sweep.obj") for g in pending for p in g['poses'] for kind in ['full','display'])
            initialization=cached if pending and cache_valid else step41.prepare(name,selected,jobs=args.jobs) if pending else None
            if initialization:
                initialization['initialization_schema']=run_all.step31.SCHEMA
                save(cached_path,initialization)
                failed=[g for g in pending if initialization['groups'][g['id']].get('preparation_error')]
                for group in failed:
                    outputs.append(dict(id=group['id'],status='geometry_unresolved',error=initialization['groups'][group['id']]['preparation_error'],stage='step4.1_prepare'))
                pending=[g for g in pending if g not in failed]
                save(prior,outputs)
            for future in as_completed([pool.submit(step41_case,(name,g,initialization)) for g in pending]):
                row=future.result();outputs.append(row);save(prior,outputs);print('STEP4.1 COMPLETE',name,row,flush=True)
            save(root/'data/step41_batch.json',dict(complete=True,object=name,results=outputs,sets=len(outputs),gate_stopped_ids=[g['id'] for g in groups if g['id'] not in ready],optimized=False,seconds=time.monotonic()-started))
            print('OBJECT COMPLETE',name,'selected',len(groups),'step41',len(outputs),'seconds',round(time.monotonic()-started,2),flush=True)
            from inspect_selected_stage_status import collect
            collect()


if __name__=='__main__':main()
