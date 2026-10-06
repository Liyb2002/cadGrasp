"""Same concurrent hybrid search with a separately recorded compiled guidance field."""
from hybrid_final import *
import physics_guided_field
from physics_guided_field_fast import FastObjectDistanceField,BACKEND,LIBRARY_VERSION
import igl


class FastHybridSearch(HybridFinalSearch):
    def __init__(self,*args,**kwargs):
        # Sweep models import this factory when instantiated. Scope is this
        # experiment process; canonical source files and other jobs are unchanged.
        physics_guided_field.ObjectDistanceField=FastObjectDistanceField
        began=time.monotonic();super().__init__(*args,**kwargs)
        self.constructor_seconds=time.monotonic()-began
        self.additional_code += [Path(__file__),HERE/'helper_func/optimization/physics_guided_field_fast.py',Path(igl.__file__)]

    def finish(self,*args,**kwargs):
        self.report_extra.update(distance_field_backend=BACKEND,libigl_version=LIBRARY_VERSION,
            constructor_seconds=self.constructor_seconds,
            guidance_backend_policy='separate cache key; unchanged grid, trilinear extension and exact force/exit acceptance')
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
    search=FastHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
