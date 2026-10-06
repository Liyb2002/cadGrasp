"""Balanced hybrid with stable, low-dimensional cone projections."""
from hybrid_balanced import *
import physics_guided_balanced_search as scheduling
from physics_guided_cone_svd import cone_projection as svd_projection
from physics_guided_preserving_gradient import preserving_refine

class StableProjection:
    def projection(self,result,k,index):
        if hasattr(self,'stop_event') and self.stop_event.is_set():
            raise RuntimeError('gradient branch cancelled after another branch solved')
        key=(result['serial'],k,index)
        if key not in self.projection_cache:
            began=time.monotonic()
            value=svd_projection(result['supplies'][k],U.target(self.states[k][0].targets[index]))
            self.cone_projection_seconds=getattr(self,'cone_projection_seconds',0.)+time.monotonic()-began
            self.projection_cache[key]=value
        return self.projection_cache[key]

class PreservingGradientBranch(StableProjection,GradientBranch):
    refine=preserving_refine

    def run_branch(self,seed,steps):
        self.best=None;self.beam=[];self.remember(seed)
        begin=len(self.trace);result=self.refine(seed,steps,'concurrent preserving gradient branch')
        # Return the actual continuation, including legitimate geometric progress.
        # The parent retains its independent best constructed force state.
        return result,self.trace[begin:]

class PreservingHybridSearch(StableProjection,BalancedHybridSearch):
    refine=preserving_refine
    def __init__(self,*args,**kwargs):
        # Process-local scheduler binding; other running scientific versions
        # continue using their original loaded sources and solver.
        scheduling.GradientBranch=PreservingGradientBranch
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'helper_func/physics_guided_cone_svd.py',HERE/'helper_func/physics_guided_preserving_gradient.py',HERE/'helper_func/physics_guided_contact_planes.py']

    def finish(self,*args,**kwargs):
        self.report_extra.update(contact_retention_policy='75% normalized missing-ray value, 25% normalized existing real head-force allocation; geometry-only steps shrink trust radius; repeated states rejected',cone_projection_backend='SVD active-set; original unbounded seven-coordinate cone; all-ray KKT gate',
                                 sampling_cone_projection_seconds=getattr(self,'cone_projection_seconds',0.))
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
    search=PreservingHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
