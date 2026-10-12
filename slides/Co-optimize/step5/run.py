"""Finish every selected saved Step4 layout: demand hulls, solid base, evaluation."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import artifact_store as A
sys.path.insert(0,str(COOPT/'evaluation'))
from metrics import METRIC_VERSION, SUPPORT_POSE_METRIC_VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--experiment',default='connected_base_v1')
    parser.add_argument('--step51-experiment')
    parser.add_argument('--step52-batch',help='Completed construction batch, including a saved partial-run continuation')
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--selection-summary',type=Path)
    args=parser.parse_args()
    base=A.result_root(args.object)
    selection=A.resolve_artifact(args.selection_summary or base/'data/base_oracle_results/summary.json')
    step51_experiment=args.step51_experiment or args.experiment
    stages=[('step5.1','step51_'+step51_experiment),
            ('step5.2',args.step52_batch or 'step52_'+args.experiment),
            ('evaluation','evaluation_'+args.experiment)]
    records=[];began=time.monotonic()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/cadgrasp-final-system')
    for stage,batch in stages:
        summary=base/'data'/batch/'summary.json'
        reused=summary.exists()
        if reused:
            report=json.loads(summary.read_text())
            if not report['complete'] or (stage=='step5.2' and report['passed']!=report['group_count']):
                raise ValueError(f'Existing batch has unfinished cases: {summary}')
            if stage=='evaluation' and report.get('metric_version')!=METRIC_VERSION:
                raise ValueError('Obsolete evaluation: regenerate evaluation/run.py with --replace; reuse physical stages unchanged')
            if stage=='evaluation' and report.get('support_pose_metric_version')!=SUPPORT_POSE_METRIC_VERSION:
                raise ValueError('Evaluation lacks support-pose membership: regenerate evaluation/run.py with --replace; reuse physical stages unchanged')
            if stage=='step5.1':
                verified=json.loads((summary.parent/'verification.json').read_text())
                inputs=verified['protected_inputs']
            elif stage=='step5.2':
                inputs=json.loads((summary.parent/'verification.json').read_text())['protected_inputs']
            else:
                inputs=json.loads((summary.parent/'verification.json').read_text())['protected']
            for path,expected in inputs.items():
                if A.digest(ROOT/path)!=expected:raise ValueError(f'Saved upstream changed: {path}')
            print('REUSE completed immutable stage',stage,flush=True)
        else:
            cmd=[sys.executable,str(COOPT/stage/'run.py'),args.object,'--experiment',args.experiment,
                 '--batch-name',batch,'--selection-summary',str(selection)]
            if stage=='step5.2':cmd.extend(['--jobs',str(args.jobs),'--step51-experiment',step51_experiment])
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            report=json.loads(summary.read_text())
        records.append(dict(stage=stage,summary=str(summary.relative_to(ROOT)),
            summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),reused=reused,
            group_count=report['group_count']))
    destination=base/'data'/('system_'+args.experiment);destination.mkdir(parents=True,exist_ok=False)
    report=dict(complete=True,object=args.object,experiment=args.experiment,selection=str(selection.relative_to(ROOT)),
        stages=records,group_count=records[-1]['group_count'],seconds=time.monotonic()-began,
        step4_saved_results_reused=True,force_solver_run=False,
        entry_source=str(Path(__file__).relative_to(ROOT)),entry_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination/'pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FULL SYSTEM COMPLETE',report['group_count'],'sets',flush=True)


if __name__=='__main__':main()
