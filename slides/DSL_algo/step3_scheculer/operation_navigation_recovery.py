"""Broaden only the finite navigation search window, not the output envelope."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from step3_scheculer import operation_dsl as F,operation_growth_recovery as R
from step3_scheculer.run_dsl import saved_task


class RoomGrow(R.RecoveryGrow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Baseline's 50 mm routing margin is computational, not a feasibility
        # boundary. Try 100 mm without using a previous support footprint.
        window=dict(min_m=(np.asarray(self.navigation_window['min_m'])-.05).tolist(),max_m=(np.asarray(self.navigation_window['max_m'])+.05).tolist())
        window['min_m'][2]=0.
        self.navigation_window=window
        installed=[F.F.transform(root,b,o) for root,b,o in zip(self.case.root_solids,self.bases,self.offsets)]
        self.mask=(F.F.bounded_space(window,self.bases,self.offsets)-self.forbidden)+F.F.union(installed)
        self.reference=F.S.unpack(self.mask)

    def connect(self):
        full,construction=super().connect()
        construction.update(navigation_margin_m=.1,navigation_recovery=True,previous_support_footprint_used=False)
        return full,construction


def activate():
    R.activate();original_sources=F.sources
    F.Grow=RoomGrow
    F.sources=lambda:list(dict.fromkeys(original_sources()+[Path(__file__)]))


def main():
    activate();group=F.I.OUTPUTS/'B'/sys.argv[1]
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    optimizer=F.Optimizer(group,[saved_task(group,p) for p in poses],dict(device='cuda',steps=3,samples=24,init_seeds=96))
    optimizer.optimize()
    print('NAVIGATION RECOVERY DONE',group.name,optimizer.incumbent.physical_count,flush=True)

if __name__=='__main__':main()
