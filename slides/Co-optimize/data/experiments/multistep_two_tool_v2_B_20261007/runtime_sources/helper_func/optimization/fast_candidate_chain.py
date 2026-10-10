"""Cheap frozen physics/sweep candidate descent; exact validation of finalists only."""
import time
import numpy as np
from scipy.optimize import minimize
from physics_guided_geometry import tangent_frames, retract, retraction_jacobian, acquisition, distance_cost
from physics_guided_contact_sweep import NominalContactSweep
from single_pose_chain import sample_one
from physics_guided import save


def prepare_guidance(search,current,probe_limit=64):
    weights,info=search.dual_weights(current,acquisition(current['directions'],search.ray_normals)[0])
    selected=np.flatnonzero(weights>0)
    selected=selected[np.argsort(weights[selected])[-probe_limit:]]
    if not len(selected):raise ValueError('No helpful contact probes')
    frozen=weights[selected].copy();frozen/=frozen.sum()
    origin=current['directions'];frames=tangent_frames(origin)
    model=NominalContactSweep(search.clearance,search.points[selected],search.ray_normals[selected],search.length,float(search.mesh.extents.max()))
    values,jac=model.linearize(origin,frames)
    normals=search.ray_normals[selected]
    def objective(flat):
        z=flat.reshape(len(origin),2)
        candidate=retract(origin,frames,z)
        c,g=acquisition(candidate,normals)
        mapping=retraction_jacobian(origin,frames,z)
        gradient=np.einsum('jnk,nki->jni',g,mapping)
        extra,derivative=distance_cost(values,jac,z)
        return float(frozen@(c+extra)),np.einsum('j,jni->ni',frozen,gradient+derivative).ravel()
    return objective,origin,frames,dict(probes=len(selected),physics=info)


def cheap_candidates(search,current,count=32):
    began=time.perf_counter()
    objective,origin,frames,info=prepare_guidance(search,current)
    setup=time.perf_counter()-began
    rng=search.candidate_rng;rows=[];choice_seconds=0.;descent_seconds=0.
    baseline=objective(np.zeros(len(origin)*2))[0]
    for index in range(count):
        began=time.perf_counter();pose=int(rng.integers(len(origin)))
        candidate,axis,angle=sample_one(origin,search.normals,pose,rng,search.sample_angle)
        # Represent the proposed sphere point exactly in the incumbent tangent chart.
        dots=np.sum(candidate*origin,axis=1)
        z=np.einsum('nki,nk->ni',frames,candidate)/dots[:,None]
        choice_seconds+=time.perf_counter()-began
        began=time.perf_counter();start=z.ravel();radius=np.deg2rad(12.)
        opt=minimize(objective,start,jac=True,method='SLSQP',constraints=[
            dict(type='ineq',fun=lambda x:np.sum(retract(origin,frames,x.reshape(z.shape))*search.normals,axis=1)),
            dict(type='ineq',fun=lambda x:radius**2-float((x-start)@(x-start)))],
            options={'maxiter':20,'ftol':1e-8})
        valid=np.isfinite(opt.x).all() and np.linalg.norm(opt.x-start)<=radius*(1+1e-5)
        endpoint=retract(origin,frames,opt.x.reshape(z.shape)) if valid else candidate
        if np.min(np.sum(endpoint*search.normals,axis=1))<-1e-10: endpoint=candidate;valid=False
        value=objective(opt.x if valid else start)[0]
        # A failed optimizer must not turn an improving proposal into a worse endpoint.
        initial_value=objective(start)[0]
        if value>initial_value:endpoint=candidate;value=initial_value
        elapsed=time.perf_counter()-began;descent_seconds+=elapsed
        rows.append(dict(candidate=index+1,pose=search.group['poses'][pose],axis=axis,angle_degrees=angle,
                         direction_choice=candidate.tolist(),gradient_descent=endpoint.tolist(),
                         surrogate_before=baseline,surrogate_after=value,gradient_descent_seconds=elapsed,
                         optimizer_success=bool(opt.success)))
    rows.sort(key=lambda r:r['surrogate_after'])
    return rows,dict(guidance_setup_seconds=setup,direction_choice_total_seconds=choice_seconds,
                     gradient_descent_total_seconds=descent_seconds,gradient_descent_mean_ms=descent_seconds/count*1000,**info)


def fast_candidate_search(search,iterations=1):
    began=time.monotonic();search.candidate_rng=np.random.default_rng(search.chain_seed)
    timings=[];events=[];initial=search.normals.copy()
    np.savez_compressed(search.out/'initial_directions.npz',directions=initial)
    t=time.perf_counter();current=search.exact(initial);initial_time=time.perf_counter()-t
    initial_counts=current['counts'];search.remember(current)
    events.append(dict(stage='initial',counts=current['counts'],directions=initial.tolist()))
    rounds=0
    while search.proposals<search.max_proposals and not all(m.all() for m in current['masks']):
        rounds+=1;count=min(search.candidates_per_round,search.max_proposals-search.proposals)
        rows,timing=cheap_candidates(search,current,count);search.proposals+=count
        timing.update(round=rounds,candidates=count,exact_finalists=[])
        accepted=False
        for row in rows[:search.exact_finalists]:
            t=time.perf_counter()
            try:
                result=search.exact(np.array(row['gradient_descent']))
                search.choose_loads(result);loads=sorted(search.load_bank)
                before,_=search.actual_loss(current,loads);after,_=search.actual_loss(result,loads)
                protected=all(not a.all() or b.all() for a,b in zip(current['masks'],result['masks']))
                feasible=all(m.all() for m in result['masks'])
                accepted=bool(protected and (feasible or after<before-max(1e-12,before*1e-4)))
                row.update(exact_counts=result['counts'],real_loss_before=before,real_loss_after=after,accepted=accepted)
                if accepted:current=result;search.remember(current)
            except (RuntimeError,ValueError) as error:row.update(error=str(error),accepted=False)
            timing['exact_finalists'].append(dict(candidate=row['candidate'],seconds=time.perf_counter()-t))
            if accepted:break
        events.append(dict(stage='round',round=rounds,candidates=rows,accepted=accepted,
                           counts=current['counts'],directions=current['directions'].tolist()))
        timings.append(timing)
        save(search.out/'candidate_timing.json',dict(initial_exact_seconds=initial_time,rounds=timings))
        save(search.out/'chain_trajectory.json',events)
        print('FAST ROUND',rounds,json_safe_timing(timing),current['counts'],'accepted',accepted,flush=True)
    np.savez_compressed(search.out/'continuation_directions.npz',directions=current['directions'])
    save(search.out/'chain_trajectory.json',events);save(search.out/'global_proposals.json',events)
    save(search.out/'critical_load_selection.json',search.selection_trace)
    search.report_extra=dict(algorithm='one chain; 32 cheap single-pose direction_choice / frozen-model gradient_descent candidates; exact finalists only',
        proposal_count=search.proposals,round_count=rounds,candidate_count_per_round=search.candidates_per_round,
        exact_finalists_per_round=search.exact_finalists,random_seed=search.chain_seed,
        guidance_policy='64 physics-valued probes; incumbent full-exit sweep linearization reused across candidates; 20 SLSQP iterations; approximate guidance only',
        connectivity_required=False,passed=bool(all(m.all() for m in current['masks']) and current['overlap']<1e-10 and current['partition']<1e-10 and max(current['endpoint_overlap'])<1e-10))
    return search.finish(current,initial_counts,began,'aggregate_force_feasible' if search.report_extra['passed'] else 'bounded_fast_candidate_chain_unresolved',continuation_counts=current['counts'])


def json_safe_timing(timing):
    return {k:v for k,v in timing.items() if k!='physics'}
