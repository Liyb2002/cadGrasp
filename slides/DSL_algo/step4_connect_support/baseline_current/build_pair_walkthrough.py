"""Search a new pair seating and construct the current sparse co-design feet."""
import argparse
import itertools
from pathlib import Path
import sys
import time

import manifold3d as md
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import reseating as R,zero_thickness_heads as Z
from step4_connect_support.baseline_current import build_coupled_saddle as S,outer_feet as O
from step4_connect_support.baseline_current.run_independent import read_case
from step4_connect_support.baseline_current.run_outer_feet import build
from step4_connect_support.baseline_current.finish_outer_feet import finish


def initialize(group):
    out=group/'step4';data=out/'data';data.mkdir(parents=True,exist_ok=True)
    poses=['pose_'+p for p in group.name.removeprefix('pose').split('+')]
    case=read_case(group.parent.name,poses,data/'input',
        independent_root=group/'step3_scheculer/independent_poses_floor2mm')
    if not case.schedule['passed']:raise ValueError('Walkthrough requires both original Step3 results to pass')
    Z.prepare(case,S.RELIEF)
    paths=[p for p in case.paths if not p.is_relative_to(case.source)]
    plan=dict(complete=True,object=case.name,poses=poses,
        generated_support_roots=case.support_seed_records,
        maximum_spatial_span_m=None,max_span_ratio=None,size_limit_enforced=False,
        previous_volume_cm3=None,
        provenance=dict(inputs=I.hashes(paths),code=I.hashes([Path(__file__),Path(Z.__file__)])))
    I.save(data/'design_inputs.json',plan)
    return R.load_case(out),plan


def construction_cache(case,placement):
    cache=case.output/'data/construction_cache';cache.mkdir(exist_ok=True)
    parts=[]
    box=md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    for k,(b,o,ident) in enumerate(zip(placement['bases'],placement['offsets'],placement['direction_ids'])):
        sweep=R.transform_solid(case.sweeps[k][ident],b,o)
        mesh=S.unpack(sweep)
        np.savez_compressed(cache/f'sweep{k}.npz',v=mesh.vertices,f=mesh.faces)
        parts.append(sweep.minkowski_sum(box))
    mesh=S.unpack(O.union(parts))
    np.savez_compressed(cache/'forbidden.npz',v=mesh.vertices,f=mesh.faces)
    I.save(cache/'sweep_cache.json',dict(placement=placement,sweep_length_m=S.SWEEP_LENGTH,
        relief_m=S.RELIEF,inputs=I.hashes(case.paths),
        sweep_code=I.sha256(Path(S.swept_solid.__code__.co_filename))))


def run(group):
    began=time.perf_counter();case,plan=initialize(group)
    R.precompute(case)
    if not all(case.menus):raise RuntimeError('One task has no retained exit direction')
    # Compactness ranks candidates; it never rejects them by a dimension cap.
    pool,summary=R.candidate_pool(case,limit_ratio=np.inf,count=6000)
    selected=[];tests=[]
    for rank,row in enumerate(pool[:3000]):
        menus,checks=R.screen(case,row)
        if menus is None:continue
        b=np.asarray(row['bases']);o=np.asarray(row['offsets'])
        try:R.root_records(case,b,o)
        except AssertionError:continue
        if any(np.linalg.norm(o-np.asarray(r['offsets']))<.018 and
               np.linalg.norm(b-np.asarray(r['bases']))<.4 for r in selected):continue
        row.update(direction_menus=menus,screen=checks,rank=rank);selected.append(row)
        print('WALKTHROUGH LAYOUT',len(selected),'rank',rank,'span mm',row['metrics']['maximum_spatial_span_m']*1000,flush=True)
        if len(selected)>=8:break
    ledger=dict(complete=False,selected=selected,candidate_summary=summary,
        size_limit_enforced=False,attempts=tests,global_infeasibility_claim=False)
    ledger_path=case.output/'data/walkthrough_search.json'
    I.save(ledger_path,ledger)
    for index,row in enumerate(selected):
        choices=itertools.islice(itertools.product(*row['direction_menus']),2)
        for ids in choices:
            placement=dict(bases=row['bases'],offsets=row['offsets'],direction_ids=list(ids),
                directions=[case.catalogues[k][i].tolist() for k,i in enumerate(ids)])
            plan.update(placement=placement)
            I.save(case.output/'data/design_inputs.json',plan)
            construction_cache(case,placement)
            attempt=dict(layout_index=index,direction_ids=list(ids));tests.append(attempt)
            try:
                candidate=build(case.name,group.name)
            except (ValueError,RuntimeError) as error:
                attempt.update(passed=False,error=str(error))
                print('WALKTHROUGH RETRY',index,str(error),flush=True)
                I.save(ledger_path,ledger);continue
            finish(group.name,int(candidate.name.rsplit('_',1)[1]))
            attempt.update(passed=True)
            ledger.update(complete=True,passed=True,seconds=time.perf_counter()-began)
            I.save(ledger_path,ledger)
            print('WALKTHROUGH BODY PASSED',group.name,flush=True)
            return
    ledger.update(complete=True,passed=False,seconds=time.perf_counter()-began)
    I.save(ledger_path,ledger)
    raise RuntimeError('No body passed the finite pair seating/foot search')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',default='pose1+3')
    args=p.parse_args();run(I.OUTPUTS/'B'/args.group)
