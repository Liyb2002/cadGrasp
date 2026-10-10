"""Direct real-geometry finite differences of the farthest original wrench."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from co_common import U, D, S, save, provenance, code_sources
from local_descent import step41_state
from contact_recovery import preserves_loads
from physics_guided_geometry import tangent_frames, retract


def farthest_load(search, state):
    """Exact scan with safe upper bounds from feasible cone points (including 0)."""
    best = None
    records = []
    for k, mask in enumerate(state['masks']):
        indices = np.flatnonzero(~mask)
        if not len(indices):
            continue
        targets = np.asarray([U.target(search.states[k][0].targets[i]) for i in indices])
        upper = .5*np.einsum('ij,ij->i', targets, targets)
        projected = 0
        while len(upper):
            p = int(np.argmax(upper))
            if best is not None and upper[p] < best['loss']-1e-10:
                break
            i = int(indices[p])
            value = search.projection(state, k, i)
            projected += 1
            if best is None or value['loss'] > best['loss']:
                best = dict(pose_index=k, load_index=i, loss=float(value['loss']),
                            residual=value['residual'].tolist())
            # Projection belongs to this pose's cone, so distance to this point
            # is an upper bound for distance to that cone for EVERY other load.
            point = targets[p] + value['residual']
            delta = targets-point
            upper = np.minimum(upper, .5*np.einsum('ij,ij->i',delta,delta)+1e-10)
            upper[p] = -np.inf
        records.append(dict(pose_index=k, failed_loads=len(indices), projected=projected))
    return best, records


def direct_gradient(search, state, directions, target, h):
    frames = tangent_frames(directions)
    gradient = np.zeros((len(directions),2)); probes=[]
    base = target['loss']; k=target['pose_index']; i=target['load_index']
    for pose in range(len(directions)):
        for axis in range(2):
            values={}; row=dict(pose_index=pose, axis=axis, step_radians=h)
            for sign in [-1,1]:
                z=np.zeros_like(gradient);z[pose,axis]=sign*h
                d=retract(directions,frames,z)
                if np.min(np.sum(d*search.normals,axis=1)) < -1e-12:
                    row[str(sign)]=dict(error='outside legal hemisphere');continue
                try:
                    trial=search.exact(d)
                    value=float(search.projection(trial,k,i)['loss'])
                    values[sign]=value
                    row[str(sign)]=dict(loss=value,counts=trial['counts'])
                except (RuntimeError,ValueError) as error:
                    row[str(sign)]=dict(error=str(error))
            if len(values)==2:
                gradient[pose,axis]=(values[1]-values[-1])/(2*h)
                row['formula']='central difference'
            elif 1 in values:
                gradient[pose,axis]=(values[1]-base)/h
                row['formula']='forward difference at boundary'
            elif -1 in values:
                gradient[pose,axis]=(base-values[-1])/h
                row['formula']='backward difference at boundary'
            else:
                row['formula']='unresolved component';row['unresolved']=True
            row['derivative']=float(gradient[pose,axis]);probes.append(row)
            print('DIRECT DERIVATIVE',pose,axis,row['derivative'],flush=True)
    return gradient, probes


def worst_wrench_search(search, iterations=3):
    began=time.monotonic()
    directions=np.asarray([r['direction_fixture'] for r in search.initial['state_results']],float)
    if search.start_directions is not None:
        supplied=np.load(search.start_directions)['directions']
        if not np.allclose(supplied,directions,atol=1e-12,rtol=0):
            raise ValueError('must start from saved Step4.1 directions')
    initial=step41_state(search,directions)
    np.savez_compressed(search.out/'initial_directions.npz',directions=directions)
    errors=[];constructed=False;events=[];stop='iteration_limit'
    try:
        current=search.exact(directions);constructed=True
    except (RuntimeError,ValueError) as error:
        errors.append(str(error));current=initial
    area_mode=getattr(search,'contact_area_enabled',False)
    area_model=None
    if area_mode:
        from contact_area_sensitivity import ContactAreaSensitivity,area_gradient,area_changes
        area_model=ContactAreaSensitivity(search)
    exact_initial=current['counts'].copy() if constructed else None
    for iteration in range(iterations):
        if constructed and all(m.all() for m in current['masks']):
            stop='force_and_exit_feasible';break
        target,scan=farthest_load(search,current)
        if target is None:
            stop='geometry_unresolved';break
        row=dict(iteration=iteration+1,before_counts=current['counts'],before_directions=directions.tolist(),
                 target=target,farthest_scan=scan,trials=[],gradient_probes=[])
        print('DIRECT TARGET',target,scan,flush=True)
        accepted=False
        # A second finite-difference scale diagnoses exact contact plateaus.
        for degrees in [.25,1.]:
            if area_mode:
                g,probes=area_gradient(search,area_model,directions,target,np.deg2rad(degrees))
            else:
                g,probes=direct_gradient(search,current,directions,target,np.deg2rad(degrees))
            row['gradient_probes'].append(dict(step_degrees=degrees,gradient=g.tolist(),probes=probes))
            save(search.out/'optimization_trace.json',events+[row])
            if any(p.get('unresolved') for p in probes):
                row['derivative_unresolved']=True;continue
            if np.linalg.norm(g)<1e-10:
                continue
            frames=tangent_frames(directions);size=g.size;radius=np.deg2rad(2.)
            A=np.zeros((len(directions),size))
            for k in range(len(directions)):A[k,2*k:2*k+2]=search.normals[k]@frames[k]
            b=np.sum(directions*search.normals,axis=1)
            desired=-radius*g.ravel()/np.linalg.norm(g)
            fit=minimize(lambda x:(.5*np.sum((x-desired)**2),x-desired),np.zeros(size),jac=True,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda x:A@x+b,jac=lambda x:A),
                             dict(type='ineq',fun=lambda x:radius**2-x@x,jac=lambda x:-2*x)],
                options={'maxiter':40,'ftol':1e-12})
            if not fit.success or np.min(A@fit.x+b)<-1e-12:
                continue
            if area_mode:
                local_base=area_model.state(directions)
                wrench=U.target(search.states[target['pose_index']][0].targets[target['load_index']])
                local_loss=area_model.loss(local_base,target['pose_index'],wrench)
            for fraction in [1.,.5,.25,.125,.0625]:
                d=retract(directions,frames,fraction*fit.x.reshape(g.shape))
                trial_record=dict(fraction=fraction,difference_step_degrees=degrees,directions=d.tolist(),accepted=False)
                try:
                    if area_mode:
                        local_trial=area_model.state(d,local_base)
                        local_after=area_model.loss(local_trial,target['pose_index'],wrench)
                        trial_record.update(proxy_loss_before=local_loss,proxy_loss_after=local_after,
                            **area_changes(local_base['active'],local_trial['active'],area_model.areas))
                        if local_after>=local_loss-max(1e-10,local_loss*1e-5):
                            trial_record['screened_out']=True
                            row['trials'].append(trial_record)
                            continue
                    trial=search.exact(d)
                    frozen=float(search.projection(trial,target['pose_index'],target['load_index'])['loss'])
                    protected=preserves_loads(current,trial)
                    trial_record.update(counts=trial['counts'],target_loss=frozen,preserves_passed_loads=protected)
                    if protected and frozen<target['loss']-max(1e-10,target['loss']*1e-5):
                        worst,trial_scan=farthest_load(search,trial)
                        worst_loss=worst['loss'] if worst else 0.
                        trial_record.update(worst_loss=worst_loss,farthest_scan=trial_scan)
                        accepted=worst_loss<target['loss']-max(1e-10,target['loss']*1e-5)
                        if accepted:
                            current=trial;directions=d;constructed=True
                    trial_record['accepted']=accepted
                except (RuntimeError,ValueError) as error:
                    trial_record['geometry_error']=str(error)
                row['trials'].append(trial_record)
                if accepted:break
            if accepted:break
        row.update(accepted=accepted,kept_directions=directions.tolist());events.append(row)
        save(search.out/'optimization_trace.json',events)
        print('DIRECT STEP',iteration+1,accepted,current['counts'],flush=True)
        if not accepted:
            stop='direct_geometry_descent_stalled';break
    passed=all(m.all() for m in current['masks'])
    if constructed:D.export_exact_obj(S.unpack(current['construction']['remaining']),search.out/'remaining_support.obj')
    np.savez_compressed(search.out/'directions.npz',directions=directions)
    report=dict(complete=True,algorithm='farthest original wrench; incremental union contact-area sensitivity' if area_mode else 'farthest original wrench distance; finite differences through exact geometry',
        pose_set=search.group['id'],poses=search.group['poses'],initial_counts=initial['counts'],exact_initial_counts=exact_initial,
        final_counts=current['counts'],force_passed=bool(passed),geometry_constructed=constructed,
        clearance_certified=constructed,force_exit_passed=bool(passed and constructed),full_fixture_accepted=False,
        connectivity_required=False,component_count=len(current['construction']['remaining'].decompose()) if constructed else None,
        stop_reason=stop,iterations=events,seconds=time.monotonic()-began,directions=directions.tolist(),
        initial_geometry_errors=errors,load_count_per_pose=[len(t.targets) for t,T in search.states],
        original_loads_reused=True,load_subsampling_for_final_acceptance=False,exact_evaluations=search.exact_calls,
        metric='existing scaled force/moment coordinates plus seventh no-uplift coordinate; squared Euclidean cone distance',
        acceptance='exact geometry/all original loads; preserve all passing loads; global worst failed-load distance strictly decreases',
        derivative='finite differences through actual reconstructed material and reoptimized reaction cone; .25 and 1 degree scales',
        no_uplift=U.description(),provenance=provenance([search.initial_path,search.seed_path,search.contact_path]+[p for t,T in search.states for p in t.inputs],
            [Path(__file__)]+search.additional_code+code_sources()))
    if area_mode:
        report.update(derivative='finite differences of sampled contact availability under all-pose lock union; no full geometry reconstruction during differentiation',
            area_quadrature_points_per_triangle=4,area_is_force_capacity=False,
            proxy_geometry='contact normal and continuous reverse-ray obstruction; full area/core/clearance checks deferred to exact acceptance')
    save(search.out/'report.json',report)
    print('DIRECT FINAL',search.group['id'],report['final_counts'],stop,flush=True)
    return report
