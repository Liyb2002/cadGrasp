"""Search the last merge of two certified families, with unchanged acceptance.

The 32-sample population stops proposing once every sample has used its
rounds, and its seat-aligned seeds are never floor-feasible for the final
9+4 pair.  This driver works on one pair from the certified family bank:

1. sample guest rotations in SO(3); keep floor-feasible translations that
   touch the host (placement_search.py);
2. rank them by sampled merged volume, connectivity and a force screen of the
   retained parent contact rays, then refine the best by (1+lambda) search;
3. build the actual parent union exactly as membrane_search.proposal does,
   apply the unchanged 75% gate, full paired loads, continuous source-face
   exits and the ground oracle, and save an accepted node into the bank.

Placements failing only the 75% gate may be fully audited as diagnostics.
They are saved separately and are never registered as accepted families.
"""
import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from whole_task_2.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task_2.step3.run_membranes import load_model
from .membrane_search import (read_group,parent_union_candidate,saves_volume,save_group,checkpoint,
    complete_ground_proof,freeze,MERGE_VOLUME_RATIO)
from .family_bank import register,best_cover
from .ground_oracle import GroundOracle,clip_core
from .placement_search import PlacementScreen,IncrementalUnion,parent_data,merged_layout,score

SOURCE=ROOT/'whole_task_2/data/B/step3_membranes'
SCREEN=None


def pair_model(left,right):
    base,_,_=load_model(SOURCE);model=base.subset(left['members']+right['members'])
    model.oracle_required=True
    return model


def load_screen(folder,left,right):
    """Parent rays and pressure hulls are bound to both actual fixture files."""
    path=Path(folder)/'parents.pkl';binding=[left['sha256'],right['sha256'],
        sha256(Path(__file__).with_name('placement_search.py'))]
    model=pair_model(left,right)
    if path.exists():
        saved=pickle.loads(path.read_bytes())
        if saved['sha256']==binding:
            return PlacementScreen(model,saved['host'],saved['guest'],IncrementalUnion(model,saved['host'],saved['guest']))
    base,_,_=load_model(SOURCE)
    host=parent_data(base,left,read_group);guest=parent_data(base,right,read_group)
    path.write_bytes(pickle.dumps(dict(sha256=binding,host=host,guest=guest)))
    return PlacementScreen(model,host,guest,IncrementalUnion(model,host,guest))


def _init(folder,left,right):
    global SCREEN
    SCREEN=load_screen(folder,left,right)


def _screen(args):
    seed,deadline=args
    return [] if time.time()>deadline else SCREEN.screen(seed,deadline=deadline)
def _refine(args):return SCREEN.refine(*args)


def certify(args):
    """Actual construction and complete acceptance, as in proposal()."""
    left,right,C,destination,diagnose,budget=args
    began=time.monotonic();model=pair_model(left,right);model.proposal_deadline=began+budget
    qa,_=read_group(left);qb,_=read_group(right);na=len(qa.poses)
    q=merged_layout(dict(layout=qa),dict(layout=qb),np.asarray(C))
    info=dict(destination=str(destination),placement=np.asarray(C).tolist(),operation='rotational_reuse',
        placement_model='so3_floor_feasible_whole_family',parent_sha256=[left['sha256'],right['sha256']],
        merge_volume_ratio_limit=MERGE_VOLUME_RATIO)
    try:
        candidate=clip_core(model,parent_union_candidate(model,q,left,right,na))
        info['volume_cm3']=candidate['volume_cm3']
        info['parent_volume_ratio']=candidate['volume_cm3']/(left['volume_cm3']+right['volume_cm3'])
        gate=saves_volume(candidate['volume_cm3'],left['volume_cm3'],right['volume_cm3'])
        if not gate and not diagnose:return dict(info,status='volume_rejected',seconds=time.monotonic()-began)
        model.refine_exits(candidate);model.audit(candidate)
        info['volume_cm3']=candidate['volume_cm3']
        info['parent_volume_ratio']=candidate['volume_cm3']/(left['volume_cm3']+right['volume_cm3'])
        gate=saves_volume(candidate['volume_cm3'],left['volume_cm3'],right['volume_cm3'])
        if not gate and not diagnose:return dict(info,status='volume_rejected',seconds=time.monotonic()-began)
        witness,oracle=GroundOracle(model).query(candidate)
        candidate.update(oracle_required=True,oracle=oracle,oracle_witness_mesh=witness)
        path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
        if gate:
            info['group']=save_group(path,left['members']+right['members'],candidate,'rotational_reuse',[left,right])
            status='accepted'
        else:
            # Same files and proofs, but outside the bank: it exceeds the gate.
            np.savez_compressed(path,**candidate['layout'].arrays(),vertices=candidate['support'].vertices,
                faces=candidate['support'].faces,members=np.asarray(left['members']+right['members']))
            np.savez_compressed(path.with_name(path.stem+'_witness.npz'),vertices=witness.vertices,faces=witness.faces)
            info.update(file=str(path),sha256=sha256(path),audit=candidate['audit'],ground_oracle=oracle,
                ground_oracle_complete=complete_ground_proof(oracle,len(q.poses)),accepted_family=False)
            status='over_volume_limit_but_fully_certified'
        info.update(status=status,seconds=time.monotonic()-began)
        write_json(path.with_name(path.stem+'_endgame.json'),{k:v for k,v in info.items() if k!='audit'})
        return info
    except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as error:
        return dict(info,status='physics_rejected',reason=str(error)[:300],seconds=time.monotonic()-began)


def _child(send,args):
    try:send.send(certify(args))
    except BaseException as error:send.send(dict(status='worker_failed',reason=str(error)[:300],destination=str(args[3])))
    finally:send.close()


def isolated(args,timeout):
    """Bounded native work, as membrane_worker.isolated_proposal."""
    import multiprocessing
    ctx=multiprocessing.get_context('spawn');receive,send=ctx.Pipe(duplex=False)
    worker=ctx.Process(target=_child,args=(send,args));worker.start();send.close()
    try:
        if receive.poll(timeout):
            try:result=receive.recv()
            except EOFError:result=dict(status='worker_failed',reason='exited without a result',destination=str(args[3]))
        else:result=dict(status='timed_out',destination=str(args[3]))
    finally:
        if worker.is_alive():worker.terminate()
        worker.join(3.)
        if worker.is_alive():worker.kill();worker.join()
        receive.close()
    return result


def certify_batch(left,right,rows,folder,tag,record,count,workers,diagnose,budget,limit):
    """Full acceptance of the best distinct placements not certified before."""
    from concurrent.futures import ThreadPoolExecutor
    done={np.round(np.asarray(c['placement']),6).tobytes() for c in record['certified']}
    unique=[];seen=set(done)
    for r in sorted([r for r in rows if r.get('ok')],key=lambda r:score(r,limit)):
        key=np.round(r['C'],6).tobytes()
        if key not in seen and (diagnose or r['volume_cm3']<=limit):seen.add(key);unique.append(r)
    jobs=[(left,right,r['C'],folder/f'candidate_{tag}_{k:03}.npz',diagnose,budget) for k,r in enumerate(unique[:count])]
    with ThreadPoolExecutor(workers) as threads:
        for result in threads.map(lambda a:isolated(a,budget+60.),jobs):
            record['certified'].append(dict({k:v for k,v in result.items() if k not in ('audit','ground_oracle')},round=tag))
            write_json(folder/'search.json',record)
            print('ENDGAME CERTIFY',tag,result['status'],round(result.get('volume_cm3',float('nan')),3),
                  round(result.get('parent_volume_ratio',float('nan')),4),flush=True)


def search_pair(folder,left,right,*,rotations=240,starts=40,iterations=24,certify_count=60,workers=20,
                diagnose=True,budget=900.,seed=42,screen_seconds=3600.,refine_seconds=1800.,rounds=3):
    """The same rigid-placement merge as a proposal, with the full parallel budget.

    Sample SO(3) placements on the exact construction, then iterate local
    refinement from the best known placements and complete certification.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);record_path=folder/'search.json'
    limit=MERGE_VOLUME_RATIO*(left['volume_cm3']+right['volume_cm3'])
    record=json.loads(record_path.read_text()) if record_path.exists() else dict(parents=[left,right],
        limit_cm3=limit,certified=[],settings=dict(rotations=rotations,starts=starts,iterations=iterations,
        certify=certify_count,seed=seed,screen_seconds=screen_seconds,refine_seconds=refine_seconds,rounds=rounds,
        placement_search_sha256=sha256(Path(__file__).with_name('placement_search.py'))))
    for c in record['certified']:c.setdefault('round','r0')
    began=time.monotonic();_init(folder,left,right);SCREEN.limit=limit
    ctx=multiprocessing.get_context('fork');screened=folder/'screen.pkl'
    if screened.exists():rows=pickle.loads(screened.read_bytes())
    else:
        deadline=time.time()+screen_seconds
        with ProcessPoolExecutor(workers,mp_context=ctx) as pool:
            rows=[r for part in pool.map(_screen,[(k,deadline) for k in range(seed,seed+rotations)]) for r in part]
        screened.write_bytes(pickle.dumps(rows))
    key=lambda r:score(r,limit)
    ok=sorted([r for r in rows if r['ok']],key=key)
    record.update(screened_placements=len(rows),connected_placements=len(ok),
        screened_force_pass_within_gate=sum(1 for r in ok if all(r['proxy']) and r['volume_cm3']<=limit))
    print('ENDGAME SCREEN',len(rows),'connected',len(ok),'pass within gate',record['screened_force_pass_within_gate'],flush=True)
    for turn in range(rounds):
        # Iterated local search: refine from the best known placements, where
        # fully certified ones (even above the gate) are the best starts, then
        # certify the best new placements. The certificate stays the arbiter.
        if any(c['status']=='accepted' for c in record['certified']):break
        refined_path=folder/f'refined_{turn}.pkl'
        if refined_path.exists():refined=pickle.loads(refined_path.read_bytes())
        else:
            legal=sorted([c for c in record['certified'] if c['status']=='over_volume_limit_but_fully_certified'],
                key=lambda c:c['volume_cm3'])
            chosen=[SCREEN.evaluate(np.asarray(c['placement'])) for c in legal[:starts//2]]
            chosen+=[r for r in ok if r['volume_cm3']<=1.2*limit][:starts-len(chosen)]
            with ProcessPoolExecutor(workers,mp_context=ctx) as pool:
                deadline=time.monotonic()+refine_seconds
                refined=list(pool.map(_refine,[(r,seed+97*turn+k,iterations,8,4.,deadline) for k,r in enumerate(chosen)]))
            refined_path.write_bytes(pickle.dumps(refined));ok=sorted(ok+[r for r in refined if r.get('ok')],key=key)
        if not any(c['round']==f'r{turn}' for c in record['certified']):
            certify_batch(left,right,refined+ok,folder,f'r{turn}',record,certify_count,workers,diagnose,budget,limit)
    accepted=[c for c in record['certified'] if c['status']=='accepted']
    legal=[c for c in record['certified'] if c['status'] in ('accepted','over_volume_limit_but_fully_certified')]
    record.update(seconds=record.get('seconds',0.)+time.monotonic()-began,accepted_count=len(accepted),
        best_accepted=min(accepted,key=lambda c:c['volume_cm3']) if accepted else None,
        best_fully_certified=min(legal,key=lambda c:c['volume_cm3']) if legal else None)
    write_json(record_path,record)
    return record


def run(workdir=None,source_workdir=None,pair=None,rotations=240,starts=40,iterations=24,certify_count=60,
        workers=20,diagnose=True,budget=900.,publish=True,destination=None,seed=42):
    """Standalone endgame on a copy of a finished population."""
    root=ROOT/'whole_task_2/data/B'
    source_workdir=Path(source_workdir or root/'step4_compact_families')
    workdir=Path(workdir or root/'step4_endgame').resolve();workdir.mkdir(parents=True,exist_ok=True)
    manifest=workdir/'population.json'
    if not manifest.exists():shutil.copy2(source_workdir/'population.json',manifest)
    state=json.loads(manifest.read_text());bank=state['family_bank']
    task_count=sum(len(g['members']) for g in state['branches'][0]['groups'])
    if pair:left,right=bank[pair[0]],bank[pair[1]]
    else:
        cover=best_cover(state,task_count)
        if len(cover)!=2:raise ValueError('Endgame expects a certified two-family exact cover')
        left,right=sorted(cover,key=lambda g:-len(g['members']))
    folder=workdir/f'{left["sha256"][:12]}_{right["sha256"][:12]}'
    record=search_pair(folder,left,right,rotations=rotations,starts=starts,iterations=iterations,
        certify_count=certify_count,workers=workers,diagnose=diagnose,budget=budget,seed=seed)
    if not record['accepted_count']:
        print('ENDGAME: no merge satisfies the',MERGE_VOLUME_RATIO,'gate; best legal',
              None if not record['best_fully_certified'] else round(record['best_fully_certified']['volume_cm3'],3),flush=True)
        return record
    best=record['best_accepted']
    register(state,best['group']);state['best_cover']=[best['group']]
    state.setdefault('endgame',[]).append(dict(group=best['group'],search=str(folder/'search.json')))
    checkpoint(manifest,state)
    report=freeze(workdir,destination or root/'step4_one_set_best',publish=publish)
    record['frozen']=report['final_volume_cm3'];write_json(folder/'search.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',type=Path);p.add_argument('--source-workdir',type=Path)
    p.add_argument('--pair',nargs=2);p.add_argument('--rotations',type=int,default=240)
    p.add_argument('--starts',type=int,default=40);p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--certify',type=int,default=60);p.add_argument('--workers',type=int,default=20)
    p.add_argument('--no-diagnose',action='store_true');p.add_argument('--no-publish',action='store_true')
    p.add_argument('--destination',type=Path);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    run(a.workdir,a.source_workdir,a.pair,a.rotations,a.starts,a.iterations,a.certify,a.workers,
        not a.no_diagnose,publish=not a.no_publish,destination=a.destination,seed=a.seed)


if __name__=='__main__':
    main()
