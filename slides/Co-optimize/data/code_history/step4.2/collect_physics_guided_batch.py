"""Verify batch records and classify force/exit acceptance independently of connectivity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
from co_common import I
import hashlib
import json
import numpy as np
import argparse

HERE=Path(__file__).resolve().parents[1]
BASE=HERE/'data/experiments/history/comparisons/physics_guided_batch/B'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE)
    args=parser.parse_args();base=args.out.resolve()
    path=base/'batch.json'
    raw=path.read_bytes()
    batch=json.loads(raw)
    runner_times={}
    runner_log=base.parent.with_suffix('.log')
    if runner_log.exists():
        for line in runner_log.read_text().splitlines():
            try:record=json.loads(line)
            except (ValueError,TypeError):continue
            if isinstance(record,dict) and 'pose_set' in record and 'seconds' in record:
                runner_times[record['pose_set']]=record['seconds']
    assert batch['completed']==30, 'Batch must finish before final collection'
    (base/'batch_runner_original.json').write_bytes(raw)
    for row in batch['results']:
        folder=base/row['pose_set']
        assert hashlib.sha256((folder/'batch_initialization_snapshot.json').read_bytes()).hexdigest()==row['initialization_sha256']
        p=folder/'report.json'
        if not p.exists():
            wall=folder/'wall_budget.json'
            if wall.exists():
                data=json.loads(wall.read_text())
                row.update(status='wall_budget_exhausted',stop_reason='per_case_wall_budget_exhausted',
                    wall_budget_seconds=data['budget_seconds'],force_exit_passed=False)
            continue
        I.check_report(p)
        r=json.loads(p.read_text())
        initial=bool(np.array_equal(r['initial_counts'],r['load_count_per_pose']))
        diagnostics=r['clearance_diagnostics']
        passed=bool(r['force_passed'] and diagnostics['geometry_resolved'] and diagnostics['contact_check_performed']
            and r['nominal_sweep_overlap_m3']<1e-10 and r['partition_error_m3']<1e-10
            and float(np.max(r['endpoint_overlap_m3']))<1e-10)
        if row['status']=='runner_error':
            row['summary_correction']=row.pop('error')+'; endpoint list is reduced by maximum; solver output unchanged'
        wall=runner_times.get(row['pose_set'],row.get('runner_wall_seconds',row['seconds']))
        row.update(status='complete',returncode=0,seconds=wall,runner_wall_seconds=wall,
            optimization_seconds=r['seconds'],constructor_seconds=r.get('constructor_seconds'),force_exit_passed=passed,
            initially_feasible=initial,recovered=passed and not initial,initial_counts=r['initial_counts'],
            final_counts=r['final_counts'],single_component_passed=r['single_component_passed'],
            stop_reason=r['stop_reason'],report=str(p.relative_to(HERE)),provenance_verified=True)
    rows=batch['results']
    batch.update(force_exit_passed=sum(r.get('force_exit_passed',False) for r in rows),
        recovered=sum(r.get('recovered',False) for r in rows),initially_feasible=sum(r.get('initially_feasible',False) for r in rows),
        numerically_unresolved=sum(r['status']=='numerically_unresolved' for r in rows),
        wall_budget_exhausted=sum(r['status']=='wall_budget_exhausted' for r in rows),
        completed=30,complete=True)
    batch['by_category']={category:dict(total=len(selected),passed=sum(r.get('force_exit_passed',False) for r in selected))
        for category,selected in [('normal',[r for r in rows if not r['pose_set'].startswith('illegal/')]),
                                  ('illegal',[r for r in rows if r['pose_set'].startswith('illegal/')])]}
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(batch,indent=2)+'\n');temporary.replace(path)
    print(json.dumps({k:v for k,v in batch.items() if k!='results'},indent=2))

if __name__=='__main__':main()
