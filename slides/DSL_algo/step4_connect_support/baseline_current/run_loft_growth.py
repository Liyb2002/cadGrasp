"""Grow reference-style whole bodies in one shared-material greedy loop."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT,OUTPUTS
from step4_connect_support.baseline_current import material_graph as M,material_network as N,loft_candidates as L,loft_growth as G
from step4_connect_support.baseline_current import loft_acceptance as A,loft_refine as P,run_unified_growth as R
from step4_connect_support.baseline_current import loft_joints as J
from step4_connect_support.baseline_current.run_greedy import read_case,plain
from step4_connect_support.baseline_current.run_local_bodies import Tee
from step4_connect_support.baseline_current.publish_loft_growth import publish,SCHEMA


def optimize(case,context,graph,max_feedback=32,checkpoint=None,resume=False):
    seed=None;groups=None;history=[];began=time.monotonic()
    cache_key=dict(graph=graph['key'],solver={Path(m.__file__).name:L.S.I.sha256(Path(m.__file__))
        for m in (G,A,R)},driver=L.S.I.sha256(Path(__file__)))
    resumed=None
    if resume and checkpoint is not None and checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved['key']!=cache_key:raise RuntimeError('Accepted checkpoint belongs to different inputs or solver code')
        seed=np.asarray(saved['selected'],bool);history=saved['history']
        groups=[N.Group(np.asarray(g['members'],int),g['kind'],g['pose'],g['direction'],g['threshold']) for g in saved['groups']]
        resumed=A.inspect(case,context,graph,seed)
        if not resumed['passed']:raise RuntimeError('Accepted checkpoint failed its full replay')
        print('Replayed accepted checkpoint',resumed['mesh'].volume*1e6,flush=True)
    for iteration in range(max_feedback):
        if resumed is not None:
            result=resumed;break
        solutions,groups,separation=G.solve_cover(graph,case.demands,groups,seed)
        chosen=solutions[0][1]
        result=A.inspect(case,context,graph,chosen)
        history.append(dict(iteration=iteration,volume_cm3=result['mesh'].volume*1e6,
            passed=result['passed'],root=solutions[0][2],growth_steps=solutions[0][3],
            geometric_separation=separation,selected_nodes=np.flatnonzero(chosen).tolist()))
        print('LOFT',iteration,'passed',result['passed'],'volume',result['mesh'].volume*1e6,flush=True)
        if result['passed']:break
        new,records=R.reaction_groups(case,graph,result['checks'],result['certificate'])
        history[-1]['original_load_separation']=records
        changed=False
        for group in new:
            if not any(g.kind==group.kind and g.pose==group.pose and np.array_equal(g.members,group.members) for g in groups):
                groups.append(group);changed=True
        if not changed:raise RuntimeError('Repeated original-load failure without a new candidate group')
        seed=chosen
    else:raise RuntimeError('Whole-loft feedback limit reached without full acceptance')
    if checkpoint is not None:
        checkpoint.write_text(json.dumps(plain(dict(key=cache_key,selected=result['selected'].tolist(),history=history,
            groups=[dict(kind=g.kind,pose=g.pose,members=g.members.tolist(),direction=g.direction,threshold=g.threshold) for g in groups])),indent=2))
    # Start again with all learned requirements visible from the first action.
    # This lets one broad useful body replace several early short extensions.
    # Keep the accepted incumbent if the bounded restart fails any real check.
    restart=[]
    try:
        restart_seed=None
        for restart_iteration in range(8):
            solutions,groups,separation=G.solve_cover(graph,case.demands,groups,seed=restart_seed,roots=graph['heads'][:1])
            proposed_volume,chosen,root,steps,coverage=solutions[0]
            row=dict(root=root,proposed_volume_cm3=proposed_volume,growth_steps=steps,
                geometric_separation=separation,iteration=restart_iteration)
            restart.append(row)
            if proposed_volume>=result['mesh'].volume*1e6-1e-7:
                row.update(adopted=False,reason='not_smaller');break
            other=A.inspect(case,context,graph,chosen)
            row.update(passed=other['passed'],adopted=other['passed'])
            if other['passed']:
                result=other;break
            new,records=R.reaction_groups(case,graph,other['checks'],other['certificate'])
            row['original_load_separation']=records
            changed=False
            for group in new:
                if not any(g.kind==group.kind and g.pose==group.pose and np.array_equal(g.members,group.members) for g in groups):
                    groups.append(group);changed=True
            if not changed:break
            restart_seed=chosen
    except RuntimeError as error:
        restart.append(dict(adopted=False,reason=str(error)))
    result,deletions=P.prune(case,context,graph,result)
    design=dict(method=SCHEMA,forced_opposite_floor_projection=False,preassigned_head_floor_roles=False,
        separate_final_connection_phase=False,posthoc_network_convexification=False,
        short_connections_selected_in_same_greedy=True,
        whole_head_floor_bodies_rebuilt_during_each_action=True,
        objective='true_incremental_union_volume',external_surface_area_reported_separately=True,
        shape_family='head_and_8mm_back_section_to_3mm_tangential_foot_lofts',
        finite_candidate_family=True,handcrafted_reference_terminals_used=False,optimality_claim=False,
        shortest_path_cost_is_proposal_proxy=True,each_action_repriced_by_actual_union=True,
        original_load_feedback=True,maximum_feedback_iterations=max_feedback,
        groups=[dict(kind=g.kind,pose=g.pose,members=g.members.tolist(),direction=g.direction,threshold=g.threshold) for g in groups],
        iterations=history,constraint_informed_restart=restart,
        full_acceptance_reverse_deletions=deletions,solve_seconds=time.monotonic()-began)
    return result,design


def run_pair(pair,max_feedback=32,resume=False):
    out=pair/'step4';work=Path(tempfile.gettempdir())/'cadgrasp_loft_growth'/pair.name
    work.mkdir(parents=True,exist_ok=True)
    reference=json.loads((out/'reference_report.json').read_text())
    case=read_case((ROOT/reference['source_schedule']).parent)
    placement=reference['placement']
    context=M.prepare(case,placement,Path(tempfile.gettempdir())/'cadgrasp_unified_growth'/pair.name)
    graph=J.augment(L.build(case,context,work),context,work)
    G.enable_whole_body_growth(graph,context)
    result,design=optimize(case,context,graph,max_feedback,work/'accepted_checkpoint.json',resume)
    report=publish(out,work,case,context,placement,graph,result,design,reference)
    print('RESULT',pair.name,report['volume_cm3'],report['external_area_cm2'],report['comparison'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',nargs='+',default=['pose1+3'])
    parser.add_argument('--max-feedback',type=int,default=32)
    parser.add_argument('--resume',action='store_true',help='Replay a code/input-matched private accepted checkpoint before refinement')
    args=parser.parse_args();results=[]
    for name in args.pair:
        pair=OUTPUTS/'B'/name
        with (pair/'step4/loft_growth.log').open('w') as log,redirect_stdout(Tee(sys.stdout,log)):
            report=run_pair(pair,args.max_feedback,args.resume)
            results.append(dict(pair=name,passed=report['passed'],volume_cm3=report['volume_cm3'],
                external_area_cm2=report['external_area_cm2']))
    print(json.dumps(plain(results),indent=2))


if __name__=='__main__':main()
