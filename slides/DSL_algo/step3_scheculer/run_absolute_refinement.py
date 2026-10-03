"""Second measured branch: locally compact qualified common-direction layouts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import trimesh
from step3_scheculer import absolute_dsl as A,absolute_local_descent as L
from step3_scheculer.review_operation_dsl import load
from step3_scheculer.run_dsl import saved_task

class Optimizer(A.Optimizer):
    def __init__(self,group,tasks):
        self.group,self.tasks,self.config=group,tasks,dict(device='cuda')
        self.out=group/'step3_scheculer'/A.F.STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.checker=A.F.Checker(tasks);self.events=[];self.trial=0
        self.source=group/'step3_scheculer/dsl_absolute/final'
        self.seed,_=load(self.source,tasks);self.reference=self.source/'shape.obj';self.reference_report=A.F.I.check_report(self.source/'report.json')
        self.seed_inputs=[self.source/'report.json',self.source/'state.json']+[self.source/f'contacts_{t.pose}.npz' for t in tasks]
        self.incumbent=self.initial_state=self.seed;self.check=self.checker.check(self.seed);assert self.check['passed']
        self.initial_volume=A.M.volume(tasks,self.seed,trimesh.load(self.reference,force='mesh',process=False))
        self.best_volume=self.initial_volume;self.witness=(self.source,self.reference_report,None)
        A.F.save_state(self.out/'initial',tasks,self.seed,self.check)
        self.event('initialize_qualified',volume_cm3=self.initial_volume,directions=A.diagnostic(tasks,self.seed))

    def optimize(self):
        branches=[(self.seed,'incumbent')]
        old=self.group/'step3_scheculer/dsl_absolute/trials'
        ranked=[]
        for metric in old.glob('*/step5/report.json'):
            if 'common_up' not in metric.parent.parent.name:continue
            r=A.F.I.check_report(metric);ranked.append((r['metrics']['object_and_support_poses']['box_volume_cm3'],metric.parent.parent))
        for _,folder in sorted(ranked)[:2]:
            A.F.I.check_report(folder/'report.json');state,_=load(folder,self.tasks);branches.append((state,folder.name))
        for state,name in branches:
            try:
                compact,history=L.compact(self.tasks,state,rounds=4)
                self.event('local_descent',source=name,history=history,steps=len(history))
                if history:self.consider(compact,'local_compact_'+name)
            except Exception as e:self.event('reject_local_descent',source=name,reason=str(e))
        self.consider(self.incumbent,'final_step4_regrowth')
        self.publish()

def main():
    A.activate();source=A.F.sources
    A.F.STAGE='dsl_absolute_refined';A.F.BODY_STAGE='dsl_absolute_refined_support';A.F.SCHEMA='absolute_direction_local_descent_v8'
    A.F.sources=lambda:list(dict.fromkeys(source()+[Path(__file__),Path(L.__file__)]))
    group=A.F.I.OUTPUTS/'B'/sys.argv[1]
    r=A.F.I.check_report(group/'step3_scheculer/dsl_absolute/report.json')
    optimizer=Optimizer(group,[saved_task(group,p) for p in r['poses']]);optimizer.optimize()
    print('ABSOLUTE REFINED DONE',group.name,optimizer.initial_volume,optimizer.best_volume,flush=True)
if __name__=='__main__':main()
