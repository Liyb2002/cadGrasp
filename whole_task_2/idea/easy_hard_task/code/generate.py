"""Render the two figures; optionally search poses again, keeping no check files.

Run from the repository root. All writes stay inside easy_hard_task/.
"""
import argparse
import json

import numpy as np

from whole_task_2.step1.common import ROOT, sha256
from cover import solve, improve, IncompleteCoverage
from planner import (HERE, OUTPUT, load_inputs, build_directions, candidates,
                     unlimited_candidates, targeted, certify_candidates,
                     rescue_faces, validate_solution, adjust_unlimited)
from render import render


def save_scene(path, mesh, setup, translation, poses, nominal, evidence, selected):
    """Keep only the numerical scene required to reproduce the final image."""
    masks = np.array([evidence[int(k)]['mask'] for k in selected])
    if not masks.any(0).all():
        raise IncompleteCoverage(np.flatnonzero(~masks.any(0)))
    order = []
    remaining = np.ones(len(mesh.faces),bool)
    todo = list(selected)
    while todo:
        k = max(todo,key=lambda c: float(mesh.area_faces[evidence[int(c)]['mask']&remaining].sum()))
        order.append(int(k)); remaining &= ~evidence[int(k)]['mask']; todo.remove(k)
    quality = np.full((len(order),len(mesh.faces)),-np.inf)
    for task,k in enumerate(order):
        proof = evidence[k]
        incidence = -(mesh.face_normals*proof['axis_object']).sum(1)
        tilt = np.arccos(np.clip(proof['axis_object']@nominal[k],-1.,1.))
        mask = proof['mask']
        quality[task,mask] = incidence[mask]-.12*tilt[mask]
    owner = quality.argmax(0)
    if set(np.unique(owner)) != set(range(len(order))):
        raise ValueError('A chosen task has no assigned surface')
    validate_solution(mesh,setup,poses,np.asarray(order),owner,evidence)
    np.savez_compressed(path,poses=poses[order],face_task=owner,
        tool_axis_object=np.stack([evidence[k]['axis_object'] for k in order]),
        setup_json=np.array(json.dumps(setup)),
        translation_mm=np.array(np.inf if translation is None else translation),
        mesh_sha256=np.array(sha256(ROOT/'objects/B/mesh.stl')))
    print('FULL COVERAGE:',len(order),'poses;',len(mesh.faces),'/',len(mesh.faces),'source faces',flush=True)


def compute(translation, destination, workers=8):
    mesh,setup,_ = load_inputs(translation)
    seeds,axes,reach = build_directions(mesh,setup)
    if translation is None:
        direction_masks = np.array([
            reach[axes@axis >= np.cos(np.radians(25.))-1e-12].any(0) for axis in axes])
        # Start with a small orientation set, then exchange with all dense
        # directions. No target pose count is supplied.
        seed_ids = np.arange(len(axes)-len(seeds),len(axes))
        chosen,_ = solve(direction_masks[seed_ids],mesh.area_faces,seconds=8.)
        chosen = improve(direction_masks,seed_ids[chosen],seconds=20.)
        poses,masks,nominal = unlimited_candidates(mesh,setup,axes,direction_masks)
    else:
        poses,masks,nominal,_ = candidates(mesh,setup,seeds,axes,reach)
        chosen,_ = solve(masks,mesh.area_faces,seconds=40.)
    evidence = {}
    seen = {tuple(np.round(W[:3].ravel(),9)) for W in poses}
    for iteration in range(16):
        fresh = np.array([k for k in chosen if int(k) not in evidence],int)
        if len(fresh):
            evidence.update(certify_candidates(mesh,setup,poses,masks,nominal,axes,reach,fresh,workers))
        for k,proof in evidence.items():
            masks[k] = proof['mask']
        actual = np.array([p['mask'] for p in evidence.values()])
        missing = np.flatnonzero(~actual.any(0))
        if len(missing):
            rescue_faces(mesh,setup,poses,nominal,axes,reach,evidence,missing,workers)
            if translation is None:
                adjust_unlimited(mesh,setup,poses,nominal,axes,reach,evidence,workers)
            for k,proof in evidence.items():
                masks[k] = proof['mask']
            actual = np.array([p['mask'] for p in evidence.values()])
            missing = np.flatnonzero(~actual.any(0))
        print('CHECK ROUND',iteration,'uncovered faces',len(missing),flush=True)
        if not len(missing):
            break
        available = np.array([k for k in range(len(poses)) if k not in evidence],int)
        target = np.zeros(len(mesh.faces),bool); target[missing] = True
        absent = target & ~masks[available].any(0)
        if absent.any():
            if translation is None:
                raise IncompleteCoverage(np.flatnonzero(absent))
            extra = targeted(mesh,setup,axes,reach,np.flatnonzero(absent),seen)
            if not len(extra[0]):
                raise IncompleteCoverage(np.flatnonzero(absent))
            poses = np.concatenate([poses,np.asarray(extra[0])])
            masks = np.concatenate([masks,np.asarray(extra[1])])
            nominal = np.concatenate([nominal,np.asarray(extra[2])])
            available = np.array([k for k in range(len(poses)) if k not in evidence],int)
        local,_ = solve(masks[available],mesh.area_faces,seconds=15.,target=target)
        chosen = available[local]
    else:
        raise IncompleteCoverage(missing)
    ids = np.array(list(evidence),int)
    local,_ = solve(masks[ids],mesh.area_faces,seconds=10.)
    save_scene(destination,mesh,setup,translation,poses,nominal,evidence,ids[local])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all','100mm','unlimited'),default='all')
    parser.add_argument('--recompute',action='store_true',help='Search from scratch instead of using saved scene data')
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    for name,translation,filename in (
            ('100mm',100.,'hard_working_tasks.png'),
            ('unlimited',None,'hard_working_tasks_unlimited.png')):
        if args.case not in ('all',name):
            continue
        scene = HERE/f'scene_{name}.npz'
        if args.recompute or not scene.exists():
            compute(translation,scene,args.workers)
        render(scene,OUTPUT/filename)


if __name__ == '__main__':
    main()
