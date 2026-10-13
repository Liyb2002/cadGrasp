"""Bounded saved-layout retries; retain each attempt under its own experiment."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import argparse, functools, json, os, subprocess, sys
from run_dynamic_postprocess import ROOT, COOPT, BASE, load_stage, save, sha


def execute(case, options, params, output, stage):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    module=load_stage(stage)
    if stage=='step5.3' and params:
        import flat_tapered_base
        original=flat_tapered_base.Builder
        flat_tapered_base.Builder=functools.partial(original,**params)
    from stage5_batch import safe_worker
    row=safe_worker(module.worker,case,SimpleNamespace(**options),BASE,stage)
    row.update(stage=stage,retry_parameters=params,experiment=options['experiment'])
    save(Path(output),row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True);p.add_argument('--stage',choices=['step5.2','step5.3'],required=True)
    p.add_argument('--step51-experiment',required=True);p.add_argument('--prune-experiment',required=True)
    p.add_argument('--selection-summary',type=Path,required=True);p.add_argument('--sets',nargs='+',required=True)
    p.add_argument('--candidate-budget',type=int,default=0)
    p.add_argument('--source-contact-material',choices=['auto','constructed'],default='auto')
    p.add_argument('--clean-object-interiors',action='store_true')
    p.add_argument('--parameters',default='{}');p.add_argument('--worker-payload',type=Path)
    args=p.parse_args()
    if args.worker_payload:
        execute(*json.loads(args.worker_payload.read_text()));return
    directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=False)
    cases=[r for r in json.loads(args.selection_summary.read_text())['results']if r['id']in args.sets]
    assert len(cases)==len(args.sets)
    params=json.loads(args.parameters)
    options=dict(object='B',experiment=args.experiment,step51_experiment=args.step51_experiment,
        selection_summary=str(args.selection_summary.resolve()),reference_stage='step4',reference_experiment=args.experiment,
        source_contact_material=args.source_contact_material,clean_object_interiors=args.clean_object_interiors,
        resume_pruning_experiment=None,candidate_budget=args.candidate_budget,prune_experiment=args.prune_experiment,leg_profile='flat')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def one(case):
        name=case['id'].replace('/','__');payload=directory/(name+'_payload.json');out=directory/(name+'.json')
        save(payload,[case,options,params,str(out.resolve()),args.stage])
        with(directory/(name+'.log')).open('w')as stream:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                '--experiment',args.experiment,'--stage',args.stage,'--step51-experiment',args.step51_experiment,
                '--prune-experiment',args.prune_experiment,'--selection-summary',str(args.selection_summary),
                '--sets',case['id'],'--worker-payload',str(payload)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return json.loads(out.read_text())if out.exists()else dict(id=case['id'],passed=False,error='Worker exited '+str(code))
    rows=[]
    with ThreadPoolExecutor(max_workers=min(4,len(cases)))as pool:
        for future in as_completed([pool.submit(one,c)for c in cases]):
            row=future.result();rows.append(row)
            save(directory/'batch.json',dict(complete=len(rows)==len(cases),results=rows,options=options,
                parameters=params,source_sha256=sha(Path(__file__))))
            print('POSTPROCESS RETRY',args.experiment,row['id'],row['passed'],row.get('error',''),flush=True)


if __name__=='__main__':main()
