"""All-original-load checks on the changed seating and constructed solid."""
from __future__ import annotations

import numpy as np

import stage5_common as C
import co_common as K
from whole_search.classify import classify
from whole_search.model import Layout
from whole_search.fast_search import FastModel

import base_geometry as G


def load_tasks(data, inputs):
    tasks=[]; body=None
    for row in data['rows']:
        task, native, candidate=K.state(inputs.name, row['pose'])
        tasks.append(task)
        if body is None: body=candidate
        elif not np.array_equal(body.faces, candidate.faces) or not np.allclose(body.vertices, candidate.vertices, atol=1e-12):
            raise ValueError('Task geometry no longer corresponds to one rigid object')
        for path in task.inputs: inputs.protected[C.relative(path)]=C.digest(path)
    return tasks,body


def ground_coverage(mesh, data):
    from scipy.spatial import ConvexHull
    reports=[]; grounds=[]
    for row in data['rows']:
        world=C.F.transform_points(mesh.vertices, row['fixture_transform'])
        points=np.unique(world[np.abs(world[:,2])<=1e-9], axis=0)
        if len(row['object_ground']): points=np.vstack([points,np.c_[row['object_ground'],np.zeros(len(row['object_ground']))]])
        if len(points)>=3 and np.linalg.matrix_rank(points[:,:2]-points[:,:2].mean(0), tol=1e-10)==2:
            hull=ConvexHull(points[:,:2]); points=points[hull.vertices]
            errors=row['hull'][:,:2]@hull.equations[:,:2].T+hull.equations[:,2]
            gap=max(0., float(np.max(errors)))
            passed=gap<=1e-9
        else: passed=False; gap=None
        reports.append(dict(pose=row['pose'],passed=passed,actual_ground_contact_vertex_count=len(points),
                            maximum_demand_hull_halfplane_gap_m=gap))
        grounds.append(points)
    return dict(passed=all(r['passed'] for r in reports), per_pose=reports),grounds


def validate_contacts(mesh, construction, tasks, data, *, all_poses=True):
    rows=[]; masks=[]; supplies=[]
    for k,(task,row,placement,allowed) in enumerate(zip(tasks, data['rows'], construction.placements, construction.allowed)):
        local=mesh.copy(); local.apply_transform(np.linalg.inv(placement))
        triangles, sources=K.contact_boundary(construction.body,local,allowed)
        if len(triangles):
            points=C.F.transform_points(triangles.reshape(-1,3),row['object_transform'])
            normals=np.repeat(-construction.body.face_normals[sources],3,axis=0)@row['object_transform'][:3,:3].T
            old_native=np.asarray(task.domain.data['frame']['T_world_mesh'])
            com=task.domain.com+row['object_transform'][:3,3]-old_native[:3,3]
            raw=np.c_[normals,np.cross(points-com,normals)]
            heads=K.U.heads(raw,task.scale)
        else: heads=np.empty((0,7))
        floor=K.U.floor(K.FLOOR.columns(task.floor,task.domain.com),task.scale)
        if not len(row['object_ground']): floor=floor[-1:]
        full=K.I.merge_columns(floor,heads)
        mask,info=classify(full,task.targets)
        rows.append(dict(pose=row['pose'],passed=bool(mask.all()),count=int(mask.sum()),load_count=len(mask),
                         actual_contact_triangle_count=len(triangles),reaction_ray_count=len(full),
                         object_floor_reactions_included=bool(len(row['object_ground'])),classifier=info))
        masks.append(mask);supplies.append(full)
        print('ACTUAL OBJECT CONTACTS',row['pose'],int(mask.sum()),'/',len(mask),flush=True)
        if not mask.all() and not all_poses: break
    return dict(passed=len(rows)==len(tasks) and all(r['passed'] for r in rows),per_pose=rows),masks,supplies


def validate_ground(grounds, tasks, data):
    reports=[]; masks=[]; supplies=[]
    for points,task,row in zip(grounds,tasks,data['rows']):
        columns=np.vstack([K.FLOOR.columns(point,np.zeros(3)) for point in points])
        full=np.unique(columns*task.scale,axis=0)
        target=row['wrench_world_origin']*task.scale
        mask,info=classify(full,target)
        reports.append(dict(pose=row['pose'],passed=bool(mask.all()),count=int(mask.sum()),load_count=len(mask),
                            actual_ground_extreme_count=len(points),ground_friction_model=K.FLOOR.description(),classifier=info))
        masks.append(mask);supplies.append(full)
        print('ACTUAL SYSTEM GROUND',row['pose'],int(mask.sum()),'/',len(mask),flush=True)
    return dict(passed=all(r['passed'] for r in reports),per_pose=reports,
                fixture_self_weight_ratio=0.,strength_checked=False),masks,supplies
