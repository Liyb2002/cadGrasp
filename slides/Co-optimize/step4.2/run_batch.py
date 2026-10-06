"""Run the selected sampling and local-gradient solver on all 30 B sets."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import argparse

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE/'data/experiments/selected_hybrid_new_batch/B'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--workers',type=int,default=2)
    parser.set_defaults(algorithm='clearance_fast')
    parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();OUT=args.out.resolve()
    assert args.iterations>0 and args.workers>0
    OUT.mkdir(parents=True, exist_ok=True)
    groups = json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g, id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    assert len(groups) == 30 and len({g['id'] for g in groups}) == 30
    jobs = []
    for group in groups:
        folder = OUT/group['id']
        folder.mkdir(parents=True, exist_ok=True)
        data = folder/'data'
        data.mkdir(exist_ok=True)
        if (data/'report.json').exists():raise RuntimeError(f'Preserve existing result: choose a fresh --out, {folder}')
        source = HERE/'output/B'/group['id']/'step4/step4.1/data/report.json'
        raw = source.read_bytes()
        initial = json.loads(raw)
        (data/'batch_initialization_snapshot.json').write_bytes(raw)
        directions = np.asarray([row['direction_fixture'] for row in initial['state_results']])
        assert len(directions) == len(group['poses'])
        np.savez_compressed(data/'batch_start_directions.npz', directions=directions)
        jobs.append((group, folder, hashlib.sha256(raw).hexdigest()))
    rows = {g['id']: dict(pose_set=g['id'], status='pending', initialization_sha256=digest) for g, folder, digest in jobs}
    def ledger():
        values = list(rows.values())
        write(OUT/'batch.json', dict(algorithm=args.algorithm, initialization='Step4.1 only', iterations_limit=args.iterations,
            proposal_budget=args.max_proposals,
            connectivity_required=False, total=len(values), completed=sum(r['status'] not in ['pending','running','queued'] for r in values),
            force_exit_passed=sum(r.get('force_exit_passed', False) for r in values),
            recovered=sum(r.get('recovered', False) for r in values),
            initially_feasible=sum(r.get('initially_feasible', False) for r in values), results=values))
    def run(job):
        group, folder, digest = job
        entry='solver.py'
        command=[sys.executable,str(HERE/'step4.2'/entry),'--set',group['id'],
            '--directions',str(folder/'data/batch_start_directions.npz'),'--out',str(folder),'--iterations',str(args.iterations)]
        command+=['--max-proposals',str(args.max_proposals)]
        started=time.monotonic()
        with (folder/'data/run.log').open('w') as log:
            process=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        row=dict(pose_set=group['id'], initialization_sha256=digest, returncode=process.returncode, seconds=time.monotonic()-started)
        report=folder/'data/report.json'
        if process.returncode == 0 and report.exists():
            r=json.loads(report.read_text())
            force_exit=bool(r['force_passed'] and r['nominal_sweep_overlap_m3']<1e-10 and r['partition_error_m3']<1e-10 and float(np.max(r['endpoint_overlap_m3']))<1e-10 and r['clearance_diagnostics']['geometry_resolved'])
            initial=r['initial_counts'] is not None and all(a==b for a,b in zip(r['initial_counts'],r['load_count_per_pose']))
            row.update(status='complete',force_exit_passed=force_exit,initially_feasible=initial,
                recovered=force_exit and not initial,initial_counts=r['initial_counts'],final_counts=r['final_counts'],
                recovered_from_unresolved_initialization=force_exit and r['initial_counts'] is None,
                geometric_progress_steps=sum(any(t.get('accepted') and t.get('acceptance_reason')=='geometry' for t in it['trials']) for it in r['iterations']),
                single_component_passed=r['single_component_passed'],stop_reason=r['stop_reason'],report=str(report.relative_to(HERE)))
        else:
            row.update(status='numerically_unresolved',force_exit_passed=False, error_log=str((folder/'data/run.log').relative_to(HERE)))
        return row
    ledger()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        for job in jobs:
            rows[job[0]['id']]['status']='queued'
            futures[pool.submit(run,job)]=job[0]['id']
        ledger()
        for future in concurrent.futures.as_completed(futures):
            key=futures[future]
            try: rows[key]=future.result()
            except Exception as error: rows[key].update(status='runner_error',error=str(error))
            ledger()
            print(json.dumps(rows[key]),flush=True)
    print('BATCH COMPLETE',flush=True)

if __name__=='__main__':
    main()
