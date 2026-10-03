"""Run the feasible-incumbent head/exit DSL on saved B groups."""
import argparse,json,time,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import operation_dsl as F,contacts as I
from step3_scheculer.run_dsl import saved_task


def run(args):
    from step3_scheculer.operation_growth_recovery import activate
    activate()
    group,config=args;started=time.monotonic()
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    out=group/'step3_scheculer'/F.STAGE
    I.save(out/'report.json',dict(complete=False,passed=False,schema=F.SCHEMA,status='initializing'))
    try:
        optimizer=F.Optimizer(group,[saved_task(group,p) for p in poses],config)
        optimizer.optimize()
        result=I.check_report(out/'report.json')
        result=dict(group=group.name,passed=True,physical_heads=result['physical_head_count'],
            shared_heads=result['shared_head_count'],initial_heads=result['initial_physical_heads'],
            coverage=result['covered_counts'],initial_angle_loss=result['initial_angle_loss'],final_angle_loss=result['final_angle_loss'])
    except Exception as error:
        result=dict(complete=True,group=group.name,passed=False,error=str(error),traceback=traceback.format_exc())
        I.save(out/'report.json',result)
    result['seconds']=time.monotonic()-started
    print('FEASIBLE CASE DONE',result,flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+');p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=3);p.add_argument('--samples',type=int,default=24)
    p.add_argument('--init-seeds',type=int,default=96);a=p.parse_args();config=vars(a).copy();config.pop('groups');config.pop('jobs')
    root=I.OUTPUTS/'B';groups=[root/g for g in a.groups] if a.groups else sorted(root.glob('pose*+*'))
    if a.jobs>1:
        with ProcessPoolExecutor(max_workers=a.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(run,[(g,config) for g in groups]))
    else:rows=[run((g,config)) for g in groups]
    out=root/'pose2+9+13+15+17/step3_scheculer'/F.STAGE
    I.save(out/'batch_report.json',dict(complete=True,results=rows,passed_count=sum(r['passed'] for r in rows),config=config))
    if not all(r['passed'] for r in rows):raise SystemExit('Some cases have not reached a fully feasible initialization; inspect batch_report.json')
    from step3_scheculer.review_operation_dsl import main as audit_all
    from step4_connect_support.publish_operation_step4 import publish_all
    audit_all()
    publish_all()

if __name__=='__main__':main()
