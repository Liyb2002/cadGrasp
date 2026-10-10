"""Continue Step4.1 directions; no random proposals or contact-only acceptance."""
import time
from pathlib import Path
import numpy as np
from co_common import save, provenance, D, S, U
from contact_lock_chain import ContactLocks, contact_state
from force_descent import force_descent
from physics_guided_geometry import tangent_frames, retract
from contact_recovery import RecoveryTarget, plateau_allowed, preserves_loads


def accept_step(before, after, before_loss, after_loss):
    """Protect solved poses; prefer coverage, allow true deficit decrease on ties."""
    protected = all(not a.all() or b.all() for a, b in zip(before['masks'], after['masks']))
    def score(state):
        f = np.array([m.mean() for m in state['masks']])
        return float(f.min()), float(f.sum())
    improved = score(after) > score(before)
    tied = score(after) == score(before)
    return bool(protected and (improved or (tied and after_loss < before_loss-max(1e-12, before_loss*1e-4))))


def step41_state(search, directions):
    masks, supplies = [], []
    for pose in search.group['poses']:
        with np.load(search.base/'step4/step4.1/data'/f'{pose}.npz') as z:
            masks.append(z['force_mask'].copy()); supplies.append(z['supply_7d'].copy())
    search.exact_calls += 1
    return dict(serial=search.exact_calls, directions=directions.copy(), masks=masks,
                supplies=supplies, counts=[int(m.sum()) for m in masks])


def local_descent_search(search, iterations=8):
    began=time.monotonic()
    # The native floor normals constrain directions; they are NOT the start.
    directions=np.asarray([r['direction_fixture'] for r in search.initial['state_results']],float)
    if search.start_directions is not None:
        supplied=np.load(search.start_directions)['directions']
        if supplied.shape!=directions.shape or not np.allclose(supplied,directions,atol=1e-12,rtol=0):
            raise ValueError('local-descent must start from this group saved Step4.1 directions')
    np.savez_compressed(search.out/'initial_directions.npz',directions=directions)
    initial=step41_state(search,directions); initial_counts=initial['counts']
    errors=[]; events=[]; stop='iteration_limit'; constructed=False
    try:
        current=search.exact(directions); constructed=True
    except (RuntimeError,ValueError) as e:
        current=initial; errors.append(str(e))
    exact_initial_counts=current['counts'].copy() if constructed else None
    model=ContactLocks(search.mesh,search.points,search.ray_normals,search.length)
    recovery=getattr(search,'contact_recovery_enabled',False)
    target=None
    avoided=set();target_attempt=0
    for iteration in range(iterations):
        if constructed and all(m.all() for m in current['masks']):
            stop='force_and_exit_feasible'; break
        row=dict(iteration=iteration+1, before_counts=current['counts'], trials=[])
        try:
            if recovery:
                if target is None:
                    target=RecoveryTarget(search,current,directions,avoided=sorted(avoided));target_attempt+=1
                row['target_attempt']=target_attempt
                endpoint,stats=target.step(directions)
                cost_before=target.nonlinear_cost(directions)
            else:
                guidance=contact_state(search,model,directions)
                endpoint,stats=force_descent(search,guidance,directions,radius_degrees=4.,physical_state=current)
            row['descent']=stats
            # Rebuild the local tangent chart after each accepted step.
            frames=tangent_frames(directions)
            dots=np.sum(endpoint*directions,axis=1)
            coordinates=np.einsum('nki,nk->ni',frames,endpoint/np.maximum(dots[:,None],1e-12)-directions)
            if np.linalg.norm(coordinates)<1e-10:
                row['accepted']=False; events.append(row)
                if recovery and target_attempt<3:
                    avoided.update(map(int,target.ids));target=None
                    row['target_switched']=True
                    save(search.out/'optimization_trace.json',events)
                    continue
                stop='relaxed_descent_stalled';break
            accepted=False
            for fraction in [1.,.5,.25,.125,.0625,.03125]:
                trial_direction=retract(directions,frames,fraction*coordinates)
                record=dict(fraction=fraction,accepted=False)
                try:
                    trial=search.exact(trial_direction)
                    # Use the SAME enlarged set for both real cone deficits.
                    search.choose_loads(current); loads=search.choose_loads(trial)
                    before_loss,_=search.actual_loss(current,loads) if loads else (0.,[])
                    after_loss,_=search.actual_loss(trial,loads) if loads else (0.,[])
                    full=all(m.all() for m in trial['masks'])
                    accepted=full or accept_step(current,trial,before_loss,after_loss)
                    if recovery:
                        # Preserve every previously passing load, even in
                        # poses that were only partially feasible.
                        accepted=accepted and preserves_loads(current,trial)
                        cost_after=target.nonlinear_cost(trial_direction)
                        angles=np.degrees(np.arccos(np.clip(np.sum(trial_direction*target.anchor,axis=1),-1,1)))
                        plateau=not accepted and plateau_allowed(current,trial,cost_before,cost_after,float(angles.max()),target.plateau_steps)
                        accepted=accepted or plateau
                        record.update(target_cost_before=cost_before,target_cost_after=cost_after,
                                      target_angle_degrees=float(angles.max()),plateau_progress=plateau)
                    # Resolving geometry without changing coverage is useful
                    # when the Step4.1 baseline has no geometry certificate.
                    if not constructed and all(np.all(b[a]) for a,b in zip(current['masks'],trial['masks'])):
                        accepted=True
                    record.update(counts=trial['counts'],before_loss=before_loss,after_loss=after_loss,accepted=accepted)
                    if accepted:
                        if recovery:
                            if any(np.count_nonzero(b)>np.count_nonzero(a) for a,b in zip(current['masks'],trial['masks'])):
                                target=None;avoided.clear();target_attempt=0
                            else:target.plateau_steps+=1
                        current=trial;directions=trial_direction;constructed=True
                except (RuntimeError,ValueError) as e:
                    record['geometry_error']=str(e)
                row['trials'].append(record)
                if accepted:break
            row['accepted']=accepted;events.append(row)
            save(search.out/'optimization_trace.json',events)
            np.savez_compressed(search.out/'directions.npz',directions=directions)
            print('LOCAL DESCENT',iteration+1,'accepted',accepted,current['counts'],flush=True)
            if not accepted:
                if recovery and target_attempt<3:
                    avoided.update(map(int,target.ids));target=None
                    row['target_switched']=True
                    save(search.out/'optimization_trace.json',events)
                    continue
                stop='real_line_search_stalled';break
        except (RuntimeError,ValueError) as e:
            row.update(error=str(e),accepted=False);events.append(row);stop='guidance_unresolved';break
    force_passed=all(m.all() for m in current['masks'])
    clearance=bool(constructed)
    if constructed:
        D.export_exact_obj(S.unpack(current['construction']['remaining']),search.out/'remaining_support.obj')
    np.savez_compressed(search.out/'directions.npz',directions=directions)
    save(search.out/'optimization_trace.json',events)
    report=dict(complete=True,algorithm='Step4.1 missing-contact recovery gradient with bounded nonregressing plateau steps' if recovery else 'Step4.1 joint local force-envelope descent; exact geometry and all-load acceptance; no random proposals',
        pose_set=search.group['id'],poses=search.group['poses'],initial_counts=initial_counts,
        exact_initial_counts=exact_initial_counts,initial_geometry_errors=errors,
        final_counts=current['counts'],force_passed=bool(force_passed),geometry_constructed=constructed,
        clearance_certified=clearance,force_exit_passed=bool(force_passed and clearance),
        full_fixture_accepted=False,connectivity_required=False,stop_reason=stop,iterations=events,
        seconds=time.monotonic()-began,load_count_per_pose=[len(t.targets) for t,T in search.states],
        original_loads_reused=True,load_subsampling_for_final_acceptance=False,
        no_uplift=U.description(),directions=directions.tolist(),exact_evaluations=search.exact_calls,
        component_count=len(current['construction']['remaining'].decompose()) if constructed else None,
        initialization='saved Step4.1 state_results directions; floor normals used only as constraints',
        guidance='finite contact locks and candidate-local reoptimized reaction envelope; guidance only',
        acceptance='exact rebuilt aggregate material; all original loads; solved poses protected; coverage improvement or tied coverage with common-working-set cone deficit decrease',
        provenance=provenance([search.initial_path,search.seed_path,search.contact_path]+[p for t,T in search.states for p in t.inputs],
            [Path(__file__),Path(__file__).with_name('force_descent.py'),Path(__file__).with_name('contact_lock_chain.py'),Path(__file__).with_name('contact_recovery.py')]+search.additional_code))
    report.update(contact_recovery_enabled=recovery,plateau_step_limit=6,plateau_angle_limit_degrees=12,
                  deterministic_target_attempt_limit=3,
                  acceptance='exact geometry and all original loads; preserve every passing load; real progress or bounded decrease in fixed missing-contact obstruction' if recovery else report['acceptance'])
    if constructed:
        report.update(clearance_diagnostics=current['construction']['diagnostics'],nominal_sweep_overlap_m3=current['overlap'],
                      partition_error_m3=current['partition'],endpoint_overlap_m3=current['endpoint_overlap'])
    save(search.out/'report.json',report)
    print('LOCAL FINAL',search.group['id'],report['final_counts'],stop,'certified',report['force_exit_passed'],flush=True)
    return report
