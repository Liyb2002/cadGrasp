"""Add the local surface finish to completed results without rerunning pruning."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from wrap_pipeline import ROOT, COOPT, DEFAULT_EXPERIMENT, freeze, protect_originals, read, save, sha

DEFAULT_REINFORCEMENT='wrap_surface_reinforced_all52_20261015_03'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',default='wrap_balanced_all52_20261015_02')
    parser.add_argument('--experiment',default=DEFAULT_REINFORCEMENT)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--numeric-only',action='store_true')
    parser.add_argument('--source-snapshot',type=Path)
    args=parser.parse_args()
    base=COOPT/'data/results/B'
    old=base/'data'/('step5_'+args.source_experiment)
    source=read(old/'summary.json')
    assert source['complete'] and not source['step5_failed']
    selection=ROOT/read(old/'execution.json')['selection']
    cases={r['id']:r for r in read(selection)['results']}
    chosen=[r for r in source['results'] if not args.sets or r['id'] in args.sets]
    if args.sets and set(args.sets)-{r['id'] for r in chosen}:parser.error('Unknown pose sets')
    directory=base/'data'/('step5_'+args.experiment)
    directory.mkdir(parents=True,exist_ok=True)
    archive=(args.source_snapshot or directory/'executed_sources').resolve()
    if not archive.exists():freeze(archive)
    selected=[cases[r['id']] for r in chosen]
    protected_path=directory/'protected_originals.json'
    protected=read(protected_path) if protected_path.exists() else {}
    protected.update({p:h for p,h in protect_originals(selected,base).items() if p not in protected})
    save(protected_path,protected)
    execution=dict(complete=False,experiment=args.experiment,selection=str(selection.relative_to(ROOT)),
        source_snapshot=str(archive.relative_to(ROOT)),source_experiment=args.source_experiment,
        original_pruning_reused=True,step4_rerun=False,started_unix=time.time())
    save(directory/'execution.json',execution)
    cpus=sorted(os.sched_getaffinity(0))
    eligible=[r for r in chosen if r['passed']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TBB_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')

    def queue(slot):
        results=[]
        assigned=cpus[max(0,len(cpus)-2*args.jobs)+2*slot:max(0,len(cpus)-2*args.jobs)+2*slot+2] or cpus[-2:]
        for row in eligible[slot::args.jobs]:
            out=base/row['id']/'step5'/args.experiment
            if (out/'data/report.json').exists():
                report=read(out/'data/report.json')
                assert report['passed'] and sha(out/'support.obj')==report['support_sha256']
            else:
                if out.exists() and any(out.iterdir()):
                    previous=out/'failed_attempts'/str(time.time_ns());previous.mkdir(parents=True)
                    for item in list(out.iterdir()):
                        if item.name!='failed_attempts':shutil.move(str(item),previous/item.name)
                    label=row['id'].replace('/','__')
                    for folder,suffix in [('logs','.log'),('payloads','.json')]:
                        old_path=directory/folder/(label+suffix)
                        if old_path.exists():shutil.copy2(old_path,previous/('worker'+suffix))
                payload=dict(action='reinforce',object='B',case=cases[row['id']],
                    source=str(ROOT/row['output']),output=str(out),experiment=args.experiment,
                    selection_summary=str(selection),source_snapshot=str(archive),
                    repository_root=str(ROOT),cpus=assigned)
                label=row['id'].replace('/','__')
                path=directory/'payloads'/(label+'.json');save(path,payload)
                log=directory/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('w') as handle:
                    process=subprocess.run([sys.executable,str(COOPT/'step5/wrap_frozen_worker.py'),str(path)],
                        cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                if process.returncode:raise RuntimeError('Reinforcement failed: '+row['id']+'; '+str(log))
                report=read(out/'data/report.json')
            result=dict(row,output=str(out.relative_to(ROOT)),report=report,reused_pruning=True)
            save(directory/'progress'/(row['id'].replace('/','__')+'.json'),result)
            results.append(result)
            print('REINFORCED',row['id'],report['local_surface_reinforcement']['added_material_cm3'],flush=True)
        return results

    results=[r for r in chosen if not r['passed']]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(queue,k) for k in range(args.jobs)]):
            results.extend(future.result())
    order={r['id']:i for i,r in enumerate(chosen)}
    results.sort(key=lambda r:order[r['id']])
    summary=dict(source,experiment=args.experiment,results=results,group_count=len(chosen),
        eligible=len(eligible),passed=len(eligible),step5_failed=0,
        upstream_step4_fail=len(chosen)-len(eligible),original_pruning_reused=True,
        pipeline='connected_hull_wrap_then_progressive_deletion_then_local_surface_expansion')
    save(directory/'summary.json',summary)
    execution.update(complete=True,finished_unix=time.time());save(directory/'execution.json',execution)
    if not args.numeric_only:
        subprocess.run([sys.executable,str(COOPT/'step5/wrap_finish.py'),
                        '--experiment',args.experiment,'--jobs',str(args.jobs)],cwd=ROOT,check=True)
    print('LOCAL SURFACE FINISH COMPLETE',len(eligible),'groups; pruning unchanged',flush=True)


if __name__=='__main__':main()
