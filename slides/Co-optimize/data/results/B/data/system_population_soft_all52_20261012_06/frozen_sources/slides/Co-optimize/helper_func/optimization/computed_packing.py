"""Computed feasible layout bound and constrained compactness search.

Individually certify each pose. Sample separating-axis choices, solve minimum
floor-tangent translations, and verify complete collective exits/clearance.
This is an explicitly reported fallback, not evidence that tiny offsets suffice.
"""
import time
import numpy as np
from scipy.optimize import minimize
from co_common import S,D,U,J,save,union,material_volume,contact_boundary,supply,transform_points,provenance,code_sources
from placement_sampling import fixture_transform,shifted
from physics_guided_geometry import tangent_frames


def minimum_axis_layout(piece_bounds,obstacle_bounds,frames,rng,trials=48):
    n=len(frames);offsets=np.zeros((n,3));choices=[]
    for k in range(1,n):
        best=None;records=[]
        assignments=[[(axis,sign)]*k for axis in range(3) for sign in [-1,1]]
        assignments.extend([[(int(rng.integers(3)),int(rng.choice([-1,1]))) for j in range(k)] for _ in range(trials)])
        for assignment in assignments:
            A=[];b=[]
            for j,(axis,sign) in enumerate(assignment):
                if sign>0:
                    gap=max(obstacle_bounds[j,1,axis]-piece_bounds[k,0,axis],piece_bounds[j,1,axis]-obstacle_bounds[k,0,axis])+.005
                    a=frames[k,axis];rhs=offsets[j,axis]+gap
                else:
                    gap=max(piece_bounds[k,1,axis]-obstacle_bounds[j,0,axis],obstacle_bounds[k,1,axis]-piece_bounds[j,0,axis])+.005
                    a=-frames[k,axis];rhs=-offsets[j,axis]+gap
                A.append(a);b.append(rhs)
            A=np.asarray(A);b=np.asarray(b)
            fit=minimize(lambda x:(.5*x@x,x),np.zeros(2),jac=True,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda x:A@x-b,jac=lambda x:A)],options={'maxiter':80,'ftol':1e-12})
            if not fit.success or np.min(A@fit.x-b)<-1e-8:continue
            length=float(np.linalg.norm(fit.x))
            if best is None or length<best[0]:best=(length,frames[k]@fit.x,assignment)
        if best is None:raise RuntimeError('no separating axis layout found')
        offsets[k]=best[1];choices.append(dict(pose_index=k,translation_m=offsets[k].tolist(),axes=best[2]))
    return offsets,choices


def small_motion_path(start_directions,end_directions,end_offsets):
    angle=np.arccos(np.clip(np.sum(start_directions*end_directions,axis=1),-1,1))
    steps=max(1,int(np.ceil(max(float(angle.max()/np.deg2rad(1.)),float(np.linalg.norm(end_offsets,axis=1).max()/.001)))))
    path=[]
    for step in range(steps+1):
        fraction=step/steps;directions=start_directions.copy()
        ids=angle>1e-10
        directions[ids]=(np.sin((1-fraction)*angle[ids])[:,None]*start_directions[ids]+
                         np.sin(fraction*angle[ids])[:,None]*end_directions[ids])/np.sin(angle[ids])[:,None]
        path.append(dict(directions=directions.tolist(),offsets_m=(fraction*end_offsets).tolist()))
    return path


def computed_fallback(search,contraction_trials=6):
    began=time.monotonic();pieces=[];cores=[];directions=[];own_reports=[];bounds=[];obstacles=[];nominal_sweeps=[];padded_sweeps=[]
    for k,(task,T) in enumerate(search.states):
        seed=search.seed_parts[k]-search.work_obstacle
        frames=search.frames[k];options=[search.normals[k],search.initial_directions[k]]
        options.extend((search.normals[k]+sign*.3*frames[:,axis])/np.linalg.norm(search.normals[k]+sign*.3*frames[:,axis]) for axis in range(2) for sign in [-1,1])
        accepted=None;errors=[]
        for d in options:
            for fan in [8,2,16]:
                try:
                    nominal=search.clearance.sweep(search.length*d,fan,padded=False);padded=search.clearance.sweep(search.length*d,fan,padded=True)
                    allowed=search.allowed[search.mesh.face_normals[search.allowed]@d<=1e-9]
                    c=search.clearance.construct(seed,[nominal],[padded],allowed)
                    if not c['diagnostics']['geometry_resolved']:raise RuntimeError('individual clearance unresolved')
                    triangles=[];sources=[]
                    for part in c['remaining'].decompose():
                        if material_volume(part)<1e-12:continue
                        tri,src=contact_boundary(search.mesh,S.unpack(part),allowed);triangles.append(tri);sources.append(src)
                    full=supply(task,T,np.concatenate(triangles),np.concatenate(sources));mask,info=J.classify(full,task.targets)
                    if not mask.all():raise RuntimeError('individual original-load failure: '+str(int(mask.sum())))
                    endpoint=material_volume(shifted(S.solid(search.mesh),search.length*d)^seed)
                    if endpoint>=1e-10:raise RuntimeError('individual endpoint unresolved')
                    accepted=(d,c,nominal,padded,full,mask);break
                except (RuntimeError,ValueError) as error:errors.append(str(error))
            if accepted:break
        if accepted is None:raise RuntimeError('individual pose infeasible/unresolved '+search.group['poses'][k]+': '+str(errors))
        d,c,nominal,padded,full,mask=accepted;pieces.append(c['remaining']);cores.append(c['protected_contact_core']);directions.append(d)
        nominal_sweeps.append(nominal);padded_sweeps.append(padded);bounds.append(S.unpack(c['remaining']).bounds)
        pad_bounds=S.unpack(padded).bounds
        if not search.work_obstacle.is_empty():
            work_bounds=S.unpack(search.work_obstacle).bounds
            pad_bounds=np.stack([np.minimum(pad_bounds[0],work_bounds[0]),np.maximum(pad_bounds[1],work_bounds[1])])
        obstacles.append(pad_bounds)
        folder=search.out/'individual'/search.group['poses'][k];folder.mkdir(parents=True,exist_ok=True)
        D.export_exact_obj(S.unpack(c['remaining']),folder/'support.obj');np.savez_compressed(folder/'force.npz',mask=mask,supply_7d=full)
        own_reports.append(dict(pose=search.group['poses'][k],counts=int(mask.sum()),direction=d.tolist(),diagnostics=c['diagnostics'],errors=errors,working_region_overlap_m3=material_volume(c['remaining']^search.work_obstacle)))
        print('INDIVIDUAL BASELINE',search.group['poses'][k],int(mask.sum()),flush=True)
    offsets,assignments=minimum_axis_layout(np.asarray(bounds),np.asarray(obstacles),search.frames,search.rng)
    directions=np.asarray(directions)
    # Verify the collective certificate on actual solids, not merely AABBs.
    moved=[shifted(p,o) for p,o in zip(pieces,offsets)]
    remaining=union(moved);core=union([shifted(c,o) for c,o in zip(cores,offsets)])
    nominal=[shifted(s,o) for s,o in zip(nominal_sweeps,offsets)];padded=[shifted(s,o) for s,o in zip(padded_sweeps,offsets)]
    # Subadditive union certificate avoids a numerically singular global
    # subtraction of separately translated coplanar contact cores.
    nominal_bound=sum(row['diagnostics']['nominal_sweep_overlap_m3'] for row in own_reports)
    padded_bound=sum(row['diagnostics']['padded_sweep_overlap_outside_contact_cores_m3'] for row in own_reports)
    work_bound=sum(row['working_region_overlap_m3'] for row in own_reports)
    for owner,piece in enumerate(moved):
        for blocker in range(search.n):
            if owner==blocker:continue
            nominal_bound+=material_volume(piece^nominal[blocker])
            padded_bound+=material_volume(piece^padded[blocker])
            work_bound+=material_volume(piece^shifted(search.work_obstacle,offsets[blocker]))
    diagnostics=dict(nominal_overlap_upper_bound_m3=nominal_bound,
        padded_overlap_outside_contact_cores_upper_bound_m3=padded_bound,
        working_region_overlap_upper_bound_m3=work_bound)
    if max(diagnostics.values())>=1e-10:raise RuntimeError('collective baseline certificate unresolved: '+str(diagnostics))
    # Every other support is outside the padded object neighbourhood, so each
    # individually certified pressure region remains unchanged under the union.
    count=[len(task.targets) for task,T in search.states]
    best=dict(remaining=remaining,directions=directions,offsets=offsets,masks=[np.ones(n,bool) for n in count],counts=count,
        transforms=[fixture_transform(T,o) for (task,T),o in zip(search.states,offsets)],diagnostics=diagnostics)
    compact_checks=[];low=0.;high=1.
    for _ in range(contraction_trials):
        factor=(low+high)/2;o=offsets*factor
        try:
            trial=search.exact(directions,o);passed=all(m.all() for m in trial['masks'])
            compact_checks.append(dict(factor=factor,counts=trial['counts'],passed=passed))
            if passed:best=trial;high=factor
            else:low=factor
        except (RuntimeError,ValueError) as error:
            compact_checks.append(dict(factor=factor,error=str(error)));low=factor
    path=small_motion_path(search.initial_directions,best['directions'],best['offsets'])
    save(search.out/'computed_layout_path.json',dict(policy='computed fallback branch; <=1mm and <=1degree per pose per increment; intermediate force feasibility not asserted',steps=path))
    save(search.out/'packing_certificate.json',dict(individual=own_reports,assignments=assignments,
        initial_packing_offsets_m=offsets.tolist(),collective_diagnostics=diagnostics,
        certificate='subadditivity of per-piece own-core clearance and exact cross-piece swept/work intersections; no core-union subtraction',compact_checks=compact_checks,
        path_steps=len(path)-1,seconds=time.monotonic()-began,full_fixture_accepted=False))
    return best
