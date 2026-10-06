"""External wall budget for gated, fixed-five-pose cases; never classify feasibility."""
import argparse,json,os,shlex,signal,subprocess,time
from pathlib import Path


def write(path,data):
    tmp=path.with_suffix('.budget.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--b-gate',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seconds',type=int,default=1800)
    a=p.parse_args();root=a.root.resolve()
    rows=json.loads(a.manifest.read_text())['cases'];groups={r['object']:r['group']['id'] for r in rows}
    opened=None
    while True:
        gate=json.loads(a.b_gate.read_text())
        if gate['force_exit_passed']>=28:
            opened=opened or time.time()
        if opened:
            listing=subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True)
            for line in listing.splitlines()[1:]:
                fields=line.strip().split(None,2)
                if len(fields)!=3:continue
                pid,elapsed,command=fields
                try:words=shlex.split(command)
                except ValueError:continue
                if '--case' not in words or '--out' not in words:continue
                if Path(words[words.index('--out')+1]).resolve()!=root:continue
                name=words[words.index('--case')+1]
                if name not in groups:continue
                folder=root/name/groups[name]
                marker=folder/'evaluation_timing.json'
                if not marker.exists():
                    birth=time.time()-int(elapsed);start=max(birth,opened)
                    write(marker,dict(evaluation_start_unix=start,b_gate_open_observed_unix=opened,
                         gate_wait_seconds=max(0.,start-birth),wall_budget_seconds=a.seconds,
                         timing_policy='external process age excludes observed B-selection wait; one-second process-clock precision'))
                timing=json.loads(marker.read_text());age=time.time()-timing['evaluation_start_unix']
                if age<a.seconds or (folder/'report.json').exists() or (folder/'wall_budget.json').exists():continue
                write(folder/'wall_budget.json',dict(status='wall_budget_exhausted',passed=False,
                      object=name,pose_set=groups[name],elapsed_seconds=age,budget_seconds=a.seconds,
                      policy='bounded generalization evaluation; no infeasibility claim'))
                try:os.kill(int(pid),signal.SIGTERM)
                except ProcessLookupError:pass
                print('GENERALIZATION WALL BUDGET',name,round(age,1),flush=True)
        batch_path=root/'batch.json'
        if batch_path.exists():
            batch=json.loads(batch_path.read_text())
            if batch['completed']==batch['total']:
                for row in batch['results']:
                    folder=root/row['object']/row['pose_set'];marker=folder/'evaluation_timing.json'
                    if marker.exists():
                        timing=json.loads(marker.read_text())
                        row['gate_wait_seconds']=timing['gate_wait_seconds']
                        row['evaluation_wall_seconds']=max(0.,row['seconds']-timing['gate_wait_seconds'])
                    if (folder/'wall_budget.json').exists() and not (folder/'report.json').exists():
                        row.update(status='wall_budget_exhausted',passed=False)
                        write(folder/'case.json',row)
                batch['per_case_evaluation_wall_budget_seconds']=a.seconds
                batch['wall_budget_exhausted']=sum(r['status']=='wall_budget_exhausted' for r in batch['results'])
                write(batch_path,batch);return
        time.sleep(10)

if __name__=='__main__':main()
