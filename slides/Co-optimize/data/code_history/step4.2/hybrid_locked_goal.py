"""Concurrent sampling with persistent physics-guided contact unlock goals."""
from hybrid_all_loads import *
from physics_guided_locked_goal import locked_refine
import physics_guided_locked_search as locked_scheduling

class LockedGradientBranch(AllLoadGradientBranch):
    refine=locked_refine

    def run_branch(self,seed,steps):
        self.best=None;self.beam=[];self.remember(seed)
        begin=len(self.trace);result=self.refine(seed,steps,'persistent physical contact goal')
        return result,self.trace[begin:]

class LockedGoalHybridSearch(AllLoadHybridSearch):
    optimize=locked_scheduling.locked_goal_search
    refine=locked_refine

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        locked_scheduling.GradientBranch=LockedGradientBranch
        self.additional_code += [Path(__file__),HERE/'helper_func/physics_guided_locked_goal.py',
            HERE/'helper_func/physics_guided_unlock_filter.py',HERE/'helper_func/physics_guided_locked_search.py',
            HERE/'helper_func/physics_guided_contact_planes.py']

    def remember(self,result,common=None):
        super().remember(result,common)
        if result.get('_unlock_goal') is not None:
            # Keep one justified geometric continuation alongside the two best
            # real physical candidates. It need not immediately improve force.
            others=[r for r in self.beam if r['serial']!=result['serial']]
            self.beam=others[:2]+[result]

    def finish(self,*args,**kwargs):
        self.report_extra.update(unlock_policy='freeze physical contact goal across geometric progress; change only after a new best physical gap; at most four geometric steps per goal; retain one continuation')
        return super().finish(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--set',required=True)
    parser.add_argument('--directions',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();source=ROOT/'objects'/args.object
    groups=json.loads((source/'pose_sets.json').read_text())['sets']
    if (source/'illegal_pose_sets.json').exists():
        groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((source/'illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=LockedGoalHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
