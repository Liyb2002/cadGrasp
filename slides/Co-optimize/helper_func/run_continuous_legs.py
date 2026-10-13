"""Run independent Step5.3 solid-leg experiments with private staged pictures."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from run_dynamic_postprocess import ROOT,COOPT,BASE,sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args()
    if args.worker:
        payload=json.loads(args.worker.read_text())
        try:
            from continuous_legs_case import run_case
            result=run_case(payload['case'],payload['previous'],payload['selection'],
                args.experiment,render=not args.no_render)
        except Exception as error:
            result=dict(id=payload['case']['id'],pose_count=payload['case']['pose_count'],
                experiment=payload['case']['experiment'],passed=False,status='CONTINUOUS_LEGS_FAIL',
                error=str(error),traceback=traceback.format_exc())
        save(Path(payload['output']),result)
        print('CASE RESULT',result['id'],result['status'],result.get('error',''),flush=True)
        return
    baseline=BASE/'data/system_sculpted_pruning_all52_20261013_01'
    previous=json.loads((baseline/'evaluation_results.json').read_text())
    old={r['id']:r for r in previous['cases']}
    selected=json.loads((baseline/'selection.json').read_text())
    cases=[c for c in selected['results']if args.sets is None or c['id']in args.sets]
    assert len(cases)==52 if args.sets is None else len(cases)==len(args.sets)
    directory=BASE/'data'/args.experiment
    directory.mkdir(exist_ok=False)
    system=BASE/'data'/('system_'+args.experiment)
    system.mkdir(exist_ok=False)
    save(system/'selection.json',selected)
    save(system/'previous_evaluation.json',previous)
    selection=system/'selection.json'
    code=[p for folder in ['helper_func','vis_func','step4.2','step5.1','step5.2','step5.3','evaluation']
          for p in (COOPT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.h']]
    import shutil
    for path in code:
        target=system/'executed_sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    protected=json.loads((baseline/'protected_originals.json').read_text())
    for previous_case in previous['cases']:
        for key in ['step52_source','step53_source']:
            folder=ROOT/previous_case[key]
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix != '.png':protected[str(p.relative_to(ROOT))]=sha(p)
    save(system/'protected_originals.json',protected)
    archived=[]
    for case in cases:
        public=COOPT/'output/B'/case['id']/'step5.3'
        for picture in public.glob('*.png'):
            target=system/'previous_images'/picture.relative_to(COOPT/'output')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(picture,target)
            archived.append(dict(path=str(picture.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(picture)))
    save(system/'previous_public_images.json',archived)
    save(system/'execution.json',dict(complete=False,groups=[c['id']for c in cases],
        step4_rerun=False,step51_rerun=False,step52_rerun=False,
        source_baseline='sculpted_pruning_all52_20261013_01',
        code={str(p.relative_to(ROOT)):sha(p)for p in code}))
    rows=[];began=time.monotonic()
    def record():
        save(directory/'batch.json',dict(complete=len(rows)==len(cases),group_count=len(cases),
            passed=sum(r['passed']for r in rows),results=rows,seconds=time.monotonic()-began,
            step4_rerun=False,step51_rerun=False,step52_rerun=False))
    def one(case):
        label=case['id'].replace('/','__')
        output=directory/'workers'/(label+'.json')
        payload=directory/'payloads'/(label+'.json')
        save(payload,dict(case=case,previous=old[case['id']],selection=str(selection.resolve()),output=str(output.resolve())))
        log=directory/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--experiment',args.experiment,'--worker',str(payload)]
        if args.no_render:command.append('--no-render')
        with log.open('w')as stream:
            code=subprocess.call(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads(output.read_text())if output.exists()else dict(id=case['id'],pose_count=case['pose_count'],
            experiment=case['experiment'],passed=False,status='RUNTIME_FAIL',error='exit '+str(code))
        result['execution_log']=str(log.relative_to(ROOT));return result
    record()
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        for future in as_completed([pool.submit(one,c)for c in cases]):
            result=future.result();rows.append(result);record()
            print('CONTINUOUS',len(rows),'/',len(cases),'PASS',sum(r['passed']for r in rows),result['id'],result['status'],flush=True)
    changed=[p for p,h in protected.items()if not(ROOT/p).is_file()or sha(ROOT/p)!=h]
    source_changes=[str(p.relative_to(ROOT))for p in code
        if sha(p)!=sha(system/'executed_sources'/p.relative_to(ROOT))]
    save(system/'input_verification.json',dict(passed=not changed and not source_changes,
        protected_files=len(protected),changed_originals=changed,changed_sources=source_changes))
    save(system/'work_state.json',dict(complete=len(rows)==len(cases),groups=len(cases),passed=sum(r['passed']for r in rows),
        private_images=True,public_images_published=False,input_verification_passed=not changed and not source_changes))


if __name__=='__main__':main()
