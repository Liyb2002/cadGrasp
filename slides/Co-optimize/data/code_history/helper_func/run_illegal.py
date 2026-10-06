"""Run saved illegal B sets without changing the accepted-set registry or outputs."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import argparse,time,multiprocessing,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_all

SOURCE=ROOT/'objects/B/illegal_pose_sets.json'

def run_gate(original):
    group=dict(original,id='illegal/'+original['id'])
    (HERE/'output/B'/group['id']/'step3/data').mkdir(parents=True,exist_ok=True)
    result=run_all.run_case(('B',group,.005))
    result.update(id=original['id'],output_id=group['id'],original_floor_violation_count=original['total_directed_violating_sample_count'])
    base=HERE/'output/B'/group['id']
    for stage in ['step3.1','step3.2']:
        path=base/'step3'/stage/'data/report.json'
        if path.exists():
            report=json.loads(path.read_text());report['provenance']['inputs'].update(I.hashes([SOURCE]));save(path,report)
    # Upstream report hash changed when the registry source was added.
    path=base/'step3/step3.2/data/report.json'
    if path.exists():
        report=json.loads(path.read_text());report['provenance']['inputs'].update(I.hashes([base/'step3/step3.1/data/report.json']));save(path,report);I.check_report(path)
    path=base/'step3/data/report.json'
    if path.exists():
        report=json.loads(path.read_text());report['provenance']['inputs'].update(I.hashes([base/'step3/step3.1/data/report.json',base/'step3/step3.2/data/report.json',SOURCE]));save(path,report);I.check_report(path)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args()
    groups=json.loads(SOURCE.read_text())['sets'];began=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_gate,g) for g in groups]
        for future in as_completed(futures):
            row=future.result();rows.append(row);print('ILLEGAL GATE',row,flush=True)
    order={g['id']:i for i,g in enumerate(groups)};rows.sort(key=lambda r:order[r['id']])
    out=HERE/'output/B/illegal';save(out/'data/gate_batch.json',dict(complete=True,sets=len(rows),results=rows,seconds=time.monotonic()-began,provenance=provenance([SOURCE],[HERE/'helper_func/run_illegal.py'])))
    print('GATES',sum(r['status']=='pass' for r in rows),'PASS',sum(r['status']=='fail' for r in rows),'FAIL',sum(r['status']=='unresolved' for r in rows),'UNRESOLVED',flush=True)

if __name__=='__main__':main()
