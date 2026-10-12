"""Deterministic discrete re-seating jumps, evaluated with all-pose coverage."""
import numpy as np
from whole_search.common import transform_mesh
from whole_search.reuse_first import registered
from co_common import transform_points


def juxtapose(objective,current,*,budget=4,refine=None):
    model=objective.model;layout=current.layout;rows=[];seen=set()
    guests=sorted(layout.active,key=lambda k:current.coverage[list(layout.active).index(k)])
    # Include currently passing blockers: moving one may free another task.
    state=model.contact_delta.state(layout)
    priorities={}
    for blocker in layout.active:
        priorities[blocker]=sum(float(model.point_areas[state.locks[owner,blocker]&
            (state.coverage_counts[owner]>0)].sum()) for owner in guests if current.coverage[owner]<1-2e-6 and owner!=blocker)
    blockers=sorted(layout.active,key=lambda k:-priorities[k])
    guests=list(dict.fromkeys(guests[:2]+blockers[:2]))
    for guest in guests:
        world=model.native[layout.hosts[guest],:3,:3]@layout.directions[guest]
        hosts=sorted((k for k in layout.active if k!=guest),key=lambda k:
            -float(world@(model.native[layout.hosts[k],:3,:3]@layout.directions[k])))
        for host in hosts[:2]:
            trial=model.juxtapose(layout,guest,host)
            center_guest=transform_points(model.mesh.center_mass[None],trial.placements[guest])[0]
            center_host=transform_points(model.mesh.center_mass[None],layout.placements[host])[0]
            normal=model.floor_normal(trial,guest)
            alignment=center_host-center_guest;alignment-=normal*(alignment@normal)
            trial.placements[guest,:3,3]+=alignment
            if trial.key() in seen or trial.key()==layout.key():continue
            a=transform_mesh(model.mesh,trial.placements[guest]).bounds
            b=transform_mesh(model.mesh,layout.placements[host]).bounds
            intersection=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0]))
            overlap=float(np.prod(intersection)/min(np.prod(a[1]-a[0]),np.prod(b[1]-b[0])))
            if overlap<.03:continue
            seen.add(trial.key());rows.append((trial,dict(guest=model.poses[guest],host=model.poses[host],
                body_bbox_overlap_fraction=overlap,placement='world-horizontal centroid alignment')))
    # Restoring native reuse is a discrete jump too, and can reduce material.
    for guest in guests:
        if registered(layout,guest):continue
        trial=layout.copy();trial.hosts[guest]=guest;trial.placements[guest]=np.eye(4)
        trial.directions[guest]=layout.placements[guest,:3,:3].T@layout.directions[guest]
        if trial.key() not in seen:
            rows.insert(0,(trial,dict(guest=model.poses[guest],host=model.poses[guest],placement='restore native rotating reuse')))
            seen.add(trial.key())
    trials=[];evaluated=[]
    for candidate,detail in rows[:budget]:
        result=objective.evaluate(candidate);evaluated.append((result,detail))
        trials.append(dict(**detail,coverage=result.coverage.tolist(),estimated_material_cm3=result.volume_cm3))
    if not evaluated:return current,dict(operation='juxtapose',accepted=False,trials=[],random_design_sampling=False)
    # One best jump receives continuous branch refinement, even if the raw
    # jump initially loses coverage. A failed branch cannot replace a checked
    # feasible incumbent; actual acceptance remains outside this operation.
    candidate,detail=min(evaluated,key=lambda row:
        (not objective.covered(row[0]),objective.coverage_loss(row[0]),row[0].volume_cm3))
    raw=candidate.layout
    jump_layout=dict(placements=raw.placements.tolist(),directions=raw.directions.tolist(),
                     hosts=raw.hosts.tolist(),active=list(raw.active))
    branch=[]
    if refine is not None:candidate,branch=refine(candidate)
    accepted=objective.better(candidate,current)
    return (candidate if accepted else current),dict(operation='juxtapose',accepted=accepted,
        selected=detail,jump_layout=jump_layout,discrete_trials=trials,continuous_refinement=branch,
        random_design_sampling=False,all_poses_active=True)
