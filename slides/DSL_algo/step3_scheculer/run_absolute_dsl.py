"""Run all saved B sets through absolute direction search, Step4 and Step5."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import absolute_dsl as A
from step3_scheculer.run_dsl import saved_task

def main():
    A.activate()
    group=A.F.I.OUTPUTS/'B'/sys.argv[1]
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    optimizer=A.Optimizer(group,[saved_task(group,p) for p in poses],dict(device='cuda',seeds=96))
    optimizer.optimize()
    print('ABSOLUTE DONE',group.name,optimizer.initial_volume,optimizer.best_volume,flush=True)
if __name__=='__main__':main()
