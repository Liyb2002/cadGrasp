"""Bound exploratory B subprocesses and report time exhaustion honestly."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def write(path,data):
    temporary=path.with_suffix('.budget.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    parser.add_argument('--seconds',type=int,default=3600);args=parser.parse_args()
    roots=[p.resolve() for p in args.roots];done=set()
    while len(done)<len(roots):
        processes=subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True)
        for line in processes.splitlines()[1:]:
            fields=line.strip().split(None,2)
            if len(fields)<3:continue
            pid,elapsed,command=fields
            try:words=shlex.split(command)
            except ValueError:continue
            if '--out' not in words or '--set' not in words:continue
            out=Path(words[words.index('--out')+1]).resolve()
            root=next((r for r in roots if r==out or r in out.parents),None)
            if root is None or int(elapsed)<args.seconds or (out/'report.json').exists():continue
            if (out/'wall_budget.json').exists():continue
            record=dict(status='wall_budget_exhausted',passed=False,pose_set=words[words.index('--set')+1],
                        budget_seconds=args.seconds,elapsed_seconds=int(elapsed),pid=int(pid),
                        policy='Exploratory wall-time cap introduced after observing excessive cost; no infeasibility claim',
                        source_command=words,termination_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write(out/'wall_budget.json',record)
            try:os.kill(int(pid),signal.SIGTERM)
            except ProcessLookupError:pass
            print('WALL BUDGET',out,int(elapsed),flush=True)
        for root in roots:
            if root in done:continue
            path=root/'batch.json'
            if not path.exists():continue
            batch=json.loads(path.read_text())
            if batch['completed']!=batch['total']:continue
            for row in batch['results']:
                budget=root/row['pose_set']/'wall_budget.json'
                if budget.exists() and not (budget.parent/'report.json').exists():
                    row.update(status='wall_budget_exhausted',force_exit_passed=False,
                               stop_reason='per_case_wall_budget_exhausted',wall_budget_seconds=args.seconds)
            batch.update(per_case_wall_budget_seconds=args.seconds,
                         wall_budget_exhausted=sum(r['status']=='wall_budget_exhausted' for r in batch['results']))
            write(path,batch);done.add(root)
        if len(done)<len(roots):time.sleep(10)

if __name__=='__main__':main()
