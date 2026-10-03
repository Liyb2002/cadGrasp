"""Run all saved B sets, recover finite routing limits, audit, publish Step4."""
from pathlib import Path
import sys,json,argparse,time,traceback,multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
from step3_scheculer import contacts as I,operation_dsl as F
from step3_scheculer.run_operation_dsl import run


def execute(args):
    group,config,resume=args;started=time.monotonic()
    if resume:
        try:
            record=I.check_report(group/'step3_scheculer'/F.STAGE/'report.json')
            if record.get('passed'):
                body=I.check_report(I.ROOT/record['witness'])
                assert body['passed'] and body['construction']['minimum_branch_thickness']['all_complete_cores_preserved']
                return dict(group=group.name,passed=True,physical_heads=record['physical_head_count'],shared_heads=record['shared_head_count'],initial_heads=record['initial_physical_heads'],coverage=record['covered_counts'],reused_completed_run=True)
        except (OSError,RuntimeError,AssertionError,KeyError):pass
    row=run((group,config))
    if not row['passed']:
        from step3_scheculer.operation_navigation_recovery import activate
        from step3_scheculer.run_dsl import saved_task
        activate();poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
        try:
            optimizer=F.Optimizer(group,[saved_task(group,p) for p in poses],config);optimizer.optimize()
            row=dict(group=group.name,passed=True,physical_heads=optimizer.incumbent.physical_count,shared_heads=optimizer.incumbent.shared_count,initial_heads=optimizer.initial_state.physical_count,coverage=[32768]*len(poses),navigation_recovery=True)
        except Exception as error:
            row=dict(group=group.name,passed=False,error=str(error),traceback=traceback.format_exc())
    row['seconds']=time.monotonic()-started
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+');parser.add_argument('--jobs',type=int,default=3);parser.add_argument('--resume',action='store_true');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();config=dict(device=args.device,steps=3,samples=24,init_seeds=96)
    groups=[I.OUTPUTS/'B'/g for g in args.groups] if args.groups else sorted((I.OUTPUTS/'B').glob('pose*+*'))
    inputs=[(g,config,args.resume) for g in groups]
    if args.jobs>1:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(execute,inputs))
    else:rows=[execute(v) for v in inputs]
    if all(r['passed'] for r in rows):
        from step3_scheculer.correct_operation_metadata import correct_all
        correct_all(groups)
    out=I.OUTPUTS/'B/pose2+9+13+15+17/step3_scheculer'/F.STAGE
    I.save(out/'batch_report.json',dict(complete=True,passed=all(r['passed'] for r in rows),results=rows,passed_count=sum(r['passed'] for r in rows),config=config,resume=args.resume,provenance=dict(code=I.hashes([Path(__file__)]),inputs=I.hashes([g/'step3_scheculer'/F.STAGE/'report.json' for g in groups if (g/'step3_scheculer'/F.STAGE/'report.json').exists()]))))
    if not all(r['passed'] for r in rows):raise SystemExit('Qualified incumbents remain preserved; some new constructions require further recovery')
    from step3_scheculer.review_operation_dsl import main as audit
    from step4_connect_support.publish_operation_step4 import publish_all
    audit();publish_all()

if __name__=='__main__':main()
