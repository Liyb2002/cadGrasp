"""Default: connect and wrap ground hulls, then progressively delete material."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import artifact_store as A


def legacy_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--step51-experiment',default='material_output_all52_20261012_01')
    parser.add_argument('--step52-batch')
    parser.add_argument('--step53-batch')
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--selection-summary',type=Path)
    parser.add_argument('--skip-evaluation',action='store_true')
    parser.add_argument('--leg-profile',choices=['flat','original'],default='flat')
    args=parser.parse_args()
    base=A.result_root(args.object)
    selection=A.resolve_artifact(args.selection_summary or base/'data/material_output_all52_20261012_01/batch.json')
    demand_summary=base/'data'/('step51_'+args.step51_experiment)/'summary.json'
    if not demand_summary.exists():
        raise FileNotFoundError('Run Step5.1 first, or select its saved experiment: '+str(demand_summary))
    demand=json.loads(demand_summary.read_text())
    if not demand['complete']:raise ValueError('Saved Step5.1 batch is incomplete')
    records=[dict(stage='step5.1',summary=str(demand_summary.relative_to(ROOT)),
        summary_sha256=A.digest(demand_summary),reused=True)]
    stages=[('step5.2',args.step52_batch or 'step52_'+args.experiment,'support_pruning'),
            ('step5.3',args.step53_batch or 'step53_'+args.experiment,'final_support_construction')]
    if not args.skip_evaluation:stages.append(('evaluation','evaluation_'+args.experiment,None))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for stage,batch,kind in stages:
        summary=base/'data'/('evaluation'if stage=='evaluation'else batch)/'summary.json'
        reused=summary.exists()and stage!='evaluation'
        if reused:
            report=json.loads(summary.read_text())
            if (not report['complete'] or (kind and report.get('stage_type')!=kind)
                    or (kind and report.get('passed')!=report['group_count'])):
                raise ValueError('Existing batch belongs to a different or unfinished stage: '+str(summary))
            checks=json.loads((summary.parent/'verification.json').read_text())
            for path,sha in checks.get('protected_inputs',checks.get('protected',{})).items():
                if A.digest(ROOT/path)!=sha:raise ValueError('Saved input changed: '+path)
        else:
            cmd=[sys.executable,str(COOPT/stage/'run.py'),args.object,'--experiment',args.experiment,
                 '--batch-name',batch,'--selection-summary',str(selection)]
            if stage!='evaluation':cmd+=['--step51-experiment',args.step51_experiment,'--jobs',str(args.jobs)]
            if stage=='step5.3':cmd+=['--leg-profile',args.leg_profile]
            if args.sets:cmd+=['--sets',*args.sets]
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            report=json.loads(summary.read_text())
        records.append(dict(stage=stage,summary=str(summary.relative_to(ROOT)),
            summary_sha256=A.digest(summary),reused=reused,group_count=report['group_count']))
    directory=base/'data'/('system_'+args.experiment)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'pipeline.json').write_text(json.dumps(dict(complete=True,stages=records,
        object=args.object,experiment=args.experiment,sets=args.sets,
        contact_pruning_uses_original_demands=True,step4_rerun=False,
        step51_rerun=False,stage_order=[record['stage'] for record in records]),indent=2)+'\n')


def main():
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        return legacy_main()
    elif '--legacy-feasible' in sys.argv:
        sys.argv.remove('--legacy-feasible')
        from feasible_pipeline import main as run_feasible
        return run_feasible()
    elif '--legacy-periphery' in sys.argv:
        sys.argv.remove('--legacy-periphery')
        from periphery_pipeline import main as run_periphery
        return run_periphery()
    else:
        from wrap_pipeline import main as run_wrap
        return run_wrap()


if __name__=='__main__':main()
