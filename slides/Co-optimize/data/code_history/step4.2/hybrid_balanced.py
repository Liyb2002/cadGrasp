"""Balanced sampling plus gradients: broad branch eligibility and witness reuse."""
from hybrid_fast import *
from physics_guided_balanced_search import balanced_search

class BalancedHybridSearch(FastHybridSearch):
    optimize=balanced_search

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.relative_gap_limit=None
        self.additional_code += [Path(__file__),HERE/'helper_func/physics_guided_balanced_search.py']

    def remember(self,result,common=None):
        # A small passed-load count need not imply a large continuous deficit.
        # Keep the original diverse exploration pool; the launch loop uses the
        # measured coverage guard, without the additional 1% residual gate.
        RefinedHybridSearch.remember(self,result,common)

    def finish(self,*args,**kwargs):
        self.report_extra.update(gradient_eligibility_policy='original coverage/diversity guard; no extra relative-residual cutoff',
            sampling_policy='original 4 common + 4 independent trials every 8 covering proposals; no direction-family truncation',
            final_recovery_policy='at most three diverse actual-contact states, at most configured iterations each')
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
    search=BalancedHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
