"""Prepare original-demand contact guides for every current eligible set."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os,subprocess,sys,time
from feasible_geometry import C


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args()
    source=C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json'
    inputs=C.Inputs('B',source,'pso_all52_20261013_01_floor')
    cases=[c for c in inputs.cases if c['passed']]
    root=C.COOPT/'data/results/B/data/contact_subset_cache_20261015';root.mkdir(parents=True,exist_ok=True)
    code=C.freeze(root/'preparation_sources')
    for case in cases:
        data=inputs.load(case)
        full=C.COOPT/'data/results/B'/case['id']/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
        inputs.protected[C.relative(full)]=C.digest(full)
    started=time.monotonic();results=[]
    def save():
        C.save(root/'preparation.json',dict(complete=len(results)==len(cases),
            eligible_groups=len(cases),completed=len(results),passed=sum(r['passed']for r in results),
            results=results,seconds=time.monotonic()-started,
            original_sources=inputs.protected,executed_sources=code))
    save()
    def one(case):
        log=root/'logs'/(case['id'].replace('/','__')+'.log');log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('w')as output:
            status=subprocess.run([sys.executable,str(C.COOPT/'step5/contact_subset_batch.py'),case['id']],
                cwd=C.ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        report=root/case['id']/'report.json'
        record=json.loads(report.read_text())if report.exists()else{}
        return dict(id=case['id'],passed=status.returncode==0 and record.get('complete',False),
            input_faces=sum(r['input_faces']for r in record.get('records',[])),
            retained_faces=sum(len(r['retained_faces'])for r in record.get('records',[])),
            poses=len(record.get('records',[])),log=C.relative(log),returncode=status.returncode)
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        pending={pool.submit(one,case):case for case in sorted(cases,key=lambda c:(c['pose_count'],c['id']))}
        for future in as_completed(pending):
            row=future.result();results.append(row);save()
            print('CONTACT_GUIDES',len(results),'/',len(cases),row['id'],row['passed'],
                row['input_faces'],'->',row['retained_faces'],flush=True)
    inputs.verify()
    if not all(r['passed']for r in results):raise SystemExit(1)


if __name__=='__main__':main()
