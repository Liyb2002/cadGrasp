"""Balanced hybrid with stable, low-dimensional cone projections."""
from hybrid_balanced import *
import physics_guided_balanced_search as scheduling
from physics_guided_cone_svd import cone_projection as svd_projection

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

class StableGradientBranch(StableProjection,GradientBranch):
    pass

class StableHybridSearch(StableProjection,BalancedHybridSearch):
    def __init__(self,*args,**kwargs):
        # Process-local scheduler binding; other running scientific versions
        # continue using their original loaded sources and solver.
        scheduling.GradientBranch=StableGradientBranch
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'helper_func/physics_guided_cone_svd.py']

    def finish(self,*args,**kwargs):
        self.report_extra.update(cone_projection_backend='SVD active-set; original unbounded seven-coordinate cone; all-ray KKT gate',
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
    search=StableHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
