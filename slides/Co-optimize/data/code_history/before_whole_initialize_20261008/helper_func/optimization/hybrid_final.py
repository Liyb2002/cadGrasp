"""Shared-trend sampling and short gradient branches with reusable force witnesses."""
from physics_guided_parallel_adaptive import *
from physics_guided_concurrent_search import concurrent_search


class HybridFinalSearch(AdaptiveParallelSearch):
    optimize=concurrent_search

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'helper_func/optimization/physics_guided_concurrent_search.py']

    def finish(self,result,*args,**kwargs):
        save(self.out/'gradient_eligibility.json',self.eligibility)
        aggregate=result['construction']['remaining']
        self.report_extra.setdefault('algorithm','shared-trend sampling + short real-cone-gradient branches with certificate reuse')
        self.report_extra.update(
            gradient_eligibility_relative_loss_limit=self.relative_gap_limit,
            necessary_cone_witness_reuse=self.necessary_cone.stats,
            component_pruning_deferred=True,
            single_component_passed=bool(len(aggregate.decompose())==1 and all(m.all() for m in result['masks'])),
            validation_policy='exact aggregate construction and every original load; component pruning deferred; no exported replay')
        # Keep the exact accepted aggregate and avoid repeated all-load solves
        # on disconnected parts when connectivity is outside the requested scope.
        return ParallelHybridSearch.finish(self,dict(result,components=[]),*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--set',required=True)
    parser.add_argument('--directions',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();source=ROOT/'objects'/args.object
    groups=read_sets(args.object)['sets']
    if (source/'illegal_pose_sets.json').exists():
        groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((source/'illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=HybridFinalSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
