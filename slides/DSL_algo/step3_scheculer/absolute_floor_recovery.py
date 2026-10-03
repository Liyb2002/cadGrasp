"""Reject all-floor vertex penetration during growth, before one acceptance.

The old volume-only predicate allowed tiny sphere caps below the floor; the
constructor acceptance correctly rejected them. No geometry tolerance relaxes.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import trimesh
from step3_scheculer import absolute_dsl as A,absolute_local_descent as L
from step3_scheculer.review_operation_dsl import load
from step3_scheculer.run_dsl import saved_task

class Optimizer(A.Optimizer):
    def __init__(self,group,tasks):
        self.group,self.tasks,self.config=group,tasks,dict(device='cuda')
        self.out=group/'step3_scheculer'/A.F.STAGE;self.out.mkdir(parents=True,exist_ok=True)
        self.checker=A.F.Checker(tasks);self.events=[];self.trial=0
        self.source=group/'step3_scheculer/dsl_absolute_refined/final'
        self.seed,_=load(self.source,tasks);self.reference=self.source/'shape.obj';self.reference_report=A.F.I.check_report(self.source/'report.json')
        self.seed_inputs=[self.source/'report.json',self.source/'state.json']+[self.source/f'contacts_{t.pose}.npz' for t in tasks]
        self.incumbent=self.initial_state=self.seed;self.check=self.checker.check(self.seed);assert self.check['passed']
        self.initial_volume=A.M.volume(tasks,self.seed,trimesh.load(self.reference,force='mesh',process=False))
        self.best_volume=self.initial_volume;self.witness=(self.source,self.reference_report,None)
        A.F.save_state(self.out/'initial',tasks,self.seed,self.check)
        self.event('initialize_qualified',volume_cm3=self.initial_volume,directions=A.diagnostic(tasks,self.seed))

    def optimize(self):
        branches=[]
        for folder in sorted((self.group/'step3_scheculer/dsl_absolute/trials').glob('*common_up_seating*')):
            state,_=load(folder,self.tasks)
            result=self.consider(state,'strict_floor_'+folder.name)
            if result:branches.append(result)
        for state,witness,value in sorted(branches,key=lambda r:r[2])[:2]:
            compact,history=L.compact(self.tasks,state,rounds=5)
            self.event('local_descent',steps=len(history),history=history)
            if history:self.consider(compact,'strict_floor_local_compact_'+witness[0].name)
        self.consider(self.incumbent,'final_step4_regrowth')
        self.publish()

def activate():
    A.activate()
    class FloorGrow(A.F.Grow):
        def legal(self,body):
            points=A.F.S.unpack(body).vertices
            if any(((points-o)@b.T)[:,2].min() < -1e-10 for b,o in zip(self.bases,self.offsets)):return False
            return super().legal(body)
    original=A.F.sources
    A.F.Grow=FloorGrow;A.F.STAGE='dsl_absolute_floor';A.F.BODY_STAGE='dsl_absolute_floor_support';A.F.SCHEMA='absolute_direction_floor_growth_v9'
    A.F.sources=lambda:list(dict.fromkeys(original()+[Path(__file__),Path(L.__file__)]))

def main():
    activate();group=A.F.I.OUTPUTS/'B'/sys.argv[1]
    row=A.F.I.check_report(group/'step3_scheculer/dsl_absolute_refined/report.json')
    optimizer=Optimizer(group,[saved_task(group,p) for p in row['poses']]);optimizer.optimize()
    print('ABSOLUTE FLOOR DONE',group.name,optimizer.initial_volume,optimizer.best_volume,flush=True)
if __name__=='__main__':main()
