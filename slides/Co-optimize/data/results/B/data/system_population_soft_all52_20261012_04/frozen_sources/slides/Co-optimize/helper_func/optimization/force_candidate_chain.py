"""Compare every raw proposal and force-descent endpoint using all saved loads."""
import time
from pathlib import Path
import numpy as np
from co_common import save,provenance
from contact_lock_chain import ContactLocks,contact_state
from force_descent import force_descent
from single_pose_chain import sample_one


def real_score(state):
    fractions=np.array([m.mean() for m in state['masks']])
    return float(fractions.min()),float(fractions.sum())


def force_candidate_search(search,iterations=1):
    descent_enabled=getattr(search,'descent_enabled',True)
    began=time.perf_counter();rng=np.random.default_rng(search.chain_seed)
    model=ContactLocks(search.mesh,search.points,search.ray_normals,search.length)
    current=contact_state(search,model,search.normals.copy());initial_counts=current['counts']
    search.remember(current);events=[];rounds=0
    np.savez_compressed(search.out/'initial_directions.npz',directions=current['directions'])
    while search.proposals<search.max_proposals and not all(m.all() for m in current['masks']):
        rounds+=1;origin=current;winner=current;winner_score=real_score(current);rows=[]
        count=min(search.candidates_per_round,search.max_proposals-search.proposals)
        for index in range(count):
            t=time.perf_counter();pose=int(rng.integers(len(origin['directions'])))
            d,axis,angle=sample_one(origin['directions'],search.normals,pose,rng,search.sample_angle)
            search.proposals+=1
            row=dict(candidate=index+1,pose=search.group['poses'][pose],axis=axis,angle_degrees=angle,direction_choice=d.tolist())
            try:
                raw=contact_state(search,model,d,origin)
                if descent_enabled:
                    endpoint,stats=force_descent(search,raw,d)
                    after=contact_state(search,model,endpoint,raw)
                    descent_helped=real_score(after)>real_score(raw)
                    chosen=after if descent_helped else raw
                    row.update(raw_counts=raw['counts'],descent_counts=after['counts'],descent_helped=descent_helped,
                               descent_worsened=real_score(after)<real_score(raw),descent=stats,
                               gradient_descent=endpoint.tolist(),kept_counts=chosen['counts'])
                else:
                    chosen=raw
                    row.update(raw_counts=raw['counts'],kept_counts=raw['counts'],descent_enabled=False)
                candidate_score=real_score(chosen)
                if candidate_score>winner_score:
                    winner=chosen;winner_score=candidate_score;row['round_best_so_far']=True
                print('FORCE CANDIDATE' if descent_enabled else 'RAW CANDIDATE',rounds,index+1,row['raw_counts'],'->',row['kept_counts'],flush=True)
            except (RuntimeError,ValueError) as error:row['error']=str(error)
            row['seconds']=time.perf_counter()-t;rows.append(row)
            save(search.out/'round_in_progress.json',dict(round=rounds,candidates=rows))
            if all(m.all() for m in winner['masks']):break
        current=winner;search.remember(current)
        events.append(dict(round=rounds,candidates=rows,accepted=current is not origin,counts=current['counts'],
                           directions=current['directions'].tolist(),real_score=winner_score))
        save(search.out/'chain_trajectory.json',events)
        print('FORCE ROUND',rounds,current['counts'],flush=True)
    np.savez_compressed(search.out/'directions.npz',directions=current['directions'])
    np.savez_compressed(search.out/'contact_locks.npz',directions=current['directions'],locks=current['locks'],active=current['active'])
    report=dict(algorithm='one chain; all proposals and reaction-reoptimized descent endpoints checked on all original loads' if descent_enabled else 'one chain; direction_choice only; all 32 proposals checked on all original loads',
        gradient_descent_enabled=descent_enabled,
        complete=True,contact_model_force_passed=bool(all(m.all() for m in current['masks'])),
        geometry_constructed=False,clearance_certified=False,full_fixture_accepted=False,
        pose_set=search.group['id'],initial_counts=initial_counts,final_counts=current['counts'],
        load_count_per_pose=[len(t.targets) for t,T in search.states],seconds=time.perf_counter()-began,
        rounds=rounds,candidates=search.proposals,random_seed=search.chain_seed,
        ranking='maximize minimum pose feasible fraction, then sum of feasible fractions; ties retain current state',
        descent='candidate-local relaxed force-equilibrium envelope; reoptimized unbounded reactions; 12 SLSQP steps; real rollback' if descent_enabled else 'disabled',
        original_loads_reused=True,load_subsampling_for_final_acceptance=False,directions=current['directions'].tolist(),
        provenance=provenance([search.seed_path,search.contact_path]+[p for task,T in search.states for p in task.inputs],
            [Path(__file__),Path(__file__).with_name('force_descent.py'),Path(__file__).with_name('contact_lock_chain.py')]+search.additional_code))
    save(search.out/'report.json',report)
    print('FORCE FINAL',report['final_counts'],'seconds',report['seconds'],flush=True)
    return report
