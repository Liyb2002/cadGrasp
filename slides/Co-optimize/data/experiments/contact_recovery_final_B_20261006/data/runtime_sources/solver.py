"""Single-pose chain with joint descent and conservative padded-prism repair."""

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
    parser.add_argument('--directions',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=8);parser.add_argument('--max-proposals',type=int,default=1200)
    parser.add_argument('--search',choices=['contact-recovery','local-descent','shared','single-pose','candidate-chain','fast-candidate','contact-chain','force-descent','proposals-only'],default='contact-recovery')
    parser.add_argument('--exact-finalists',type=int,default=1)
    parser.add_argument('--candidates',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sample-angle',type=float,default=30.)
    args=parser.parse_args()
    if args.search not in ['local-descent','contact-recovery'] and args.directions is None:parser.error('--directions is required for historical search modes')
    if min(args.iterations,args.max_proposals,args.candidates,args.exact_finalists)<1:parser.error('iterations and max-proposals must be positive')
    if not 5.<args.sample_angle<=90.:parser.error('--sample-angle must be greater than 5 and at most 90')
    if (args.out/'data/report.json').exists():parser.error('choose a fresh --out to preserve existing results')
    source=ROOT/'objects'/args.object
    groups=read_pose_groups(args.object,category='all')
    group=next(g for g in groups if g['id']==args.set)
    search=ClearanceFastHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    if args.search in ['local-descent','contact-recovery']:
        from local_descent import local_descent_search
        from types import MethodType
        search.optimize=MethodType(local_descent_search,search)
        search.contact_recovery_enabled=args.search=='contact-recovery'
    if args.search=='single-pose':
        if not 5.<args.sample_angle<=90.:parser.error('--sample-angle must be greater than 5 and at most 90')
        from single_pose_chain import single_pose_search
        from types import MethodType
        search.chain_seed=args.seed;search.sample_angle=args.sample_angle
        search.optimize=MethodType(single_pose_search,search)
        search.additional_code.append(HERE/'helper_func/optimization/single_pose_chain.py')
        search.additional_artifacts.append('chain_trajectory.json')
    if args.search=='candidate-chain':
        from candidate_chain import candidate_chain_search
        from types import MethodType
        search.chain_seed=args.seed;search.sample_angle=args.sample_angle
        search.candidates_per_round=args.candidates
        search.optimize=MethodType(candidate_chain_search,search)
        search.additional_code += [HERE/'helper_func/optimization/candidate_chain.py',HERE/'helper_func/optimization/single_pose_chain.py']
        search.additional_artifacts.append('chain_trajectory.json')
    if args.search=='fast-candidate':
        from fast_candidate_chain import fast_candidate_search
        from types import MethodType
        search.chain_seed=args.seed;search.sample_angle=args.sample_angle
        search.candidates_per_round=args.candidates;search.exact_finalists=args.exact_finalists
        search.optimize=MethodType(fast_candidate_search,search)
        search.additional_code += [HERE/'helper_func/optimization/fast_candidate_chain.py',HERE/'helper_func/optimization/single_pose_chain.py']
        search.additional_artifacts += ['chain_trajectory.json','candidate_timing.json']
    if args.search=='contact-chain':
        from contact_lock_chain import contact_chain_search
        from types import MethodType
        search.chain_seed=args.seed;search.sample_angle=args.sample_angle
        search.candidates_per_round=args.candidates;search.exact_finalists=args.exact_finalists
        search.optimize=MethodType(contact_chain_search,search)
    if args.search in ['force-descent','proposals-only']:
        from force_candidate_chain import force_candidate_search
        from types import MethodType
        search.chain_seed=args.seed;search.sample_angle=args.sample_angle
        search.candidates_per_round=args.candidates
        search.descent_enabled=args.search=='force-descent'
        search.optimize=MethodType(force_candidate_search,search)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
