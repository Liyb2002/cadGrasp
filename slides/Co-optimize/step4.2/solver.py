"""Fixed hybrid search with conservative padded-prism repair."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from hybrid_fast import *
import physics_guided
from physics_guided_padded_sweep import ConservativeExitClearance

class ClearanceFastHybridSearch(FastHybridSearch):
    def __init__(self,*args,**kwargs):
        physics_guided.ExitClearance=ConservativeExitClearance
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'helper_func/optimization/physics_guided_padded_sweep.py']
        self.additional_artifacts += ['padded_sweep_repairs.json']

    def finish(self,result,*args,**kwargs):
        records=self.clearance.padded_repairs+result['construction'].get('padded_sweep_repairs',[])
        unique={json.dumps(r,sort_keys=True):r for r in records};records=list(unique.values())
        save(self.out/'padded_sweep_repairs.json',records)
        self.report_extra.update(padded_sweep_numerics='collapsed padded face prisms conservatively enlarged and retained; nominal sweep unchanged',
            padded_sweep_repairs=sum(r.get('repaired_prisms',0) for r in records),
            maximum_padded_prism_extra_cube_half_extent_m=max((r.get('maximum_extra_cube_half_extent_m',0.) for r in records),default=0.))
        return super().finish(result,*args,**kwargs)


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
    search=ClearanceFastHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
