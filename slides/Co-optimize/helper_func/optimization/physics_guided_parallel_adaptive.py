"""Parallel search with residual-based gradient eligibility and witness reuse."""
from physics_guided_parallel import *
from physics_guided_proxy import CachedNecessaryCone


class AdaptiveParallelSearch(ParallelHybridSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.relative_gap_limit=.01;self.eligibility={}
        self.necessary_cone=CachedNecessaryCone(self.rays,self.floors,
            [U.target(task.targets) for task,T in self.states],C.W.solve)
        self.additional_code += [Path(__file__),HERE/'helper_func/optimization/physics_guided_proxy.py']
        self.additional_artifacts += ['gradient_eligibility.json']

    def remember(self,result,common=None):
        super().remember(result,common)
        values=[]
        for k,row in enumerate(self.selection_trace[-len(self.states):]):
            for index in row['indices']:
                target=U.target(self.states[k][0].targets[index])
                values.append(self.projection(result,k,index)['loss']/max(.5*float(target@target),1e-12))
        gap=max(values,default=0.)
        self.eligibility[result['serial']]=dict(relative_working_gap=gap,eligible=gap<=self.relative_gap_limit,counts=result['counts'])
        # Preserve the sampler incumbent even when it is unsuitable for a
        # short gradient branch. The branch pool only retains near-feasible
        # states in actual normalized residual, not merely passed-load count.
        self.beam=[r for r in self.beam if self.eligibility[r['serial']]['eligible']]

    def proxy(self,d):
        keep=np.max(self.ray_normals@d.T,axis=1)<=1e-9
        return self.necessary_cone.check(keep,self.proxy_indices)

    def finish(self,*args,**kwargs):
        save(self.out/'gradient_eligibility.json',self.eligibility)
        self.report_extra.update(algorithm='adaptive concurrent sampling and short real-cone gradient branches',
            gradient_eligibility_relative_loss_limit=self.relative_gap_limit,
            gradient_eligibility_policy='max selected squared real equilibrium residual / squared target norm <= 0.01; inherited coverage and diversity guard also applies',
            necessary_cone_witness_reuse=self.necessary_cone.stats)
        return super().finish(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set',required=True);parser.add_argument('--directions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--max-proposals',type=int,default=1200);args=parser.parse_args()
    groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=AdaptiveParallelSearch('B',group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
