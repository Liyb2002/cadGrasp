"""Candidate placements and in-memory Step1 tool, arm and collision checks."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape, sha256
from whole_task_2.step1 import reach as R, plan_reach as P, arm as A
from whole_task_2.step1.workspace import onto, points_inside_box
from cover import greedy

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
ACCESS_HINTS = np.array([
    [0.36885968457883433, 0.02585365874413465, -0.9291254605390731],
    [0.9837524748948787, 0.17792571458628006, -0.023948031801434896],
    [-0.5409807937259111, 0.11636819409942278, -0.8329455109559822],
])


def fibonacci(count):
    i = np.arange(count)+.5
    z = 1-2*i/count
    phi = i*np.pi*(3-np.sqrt(5))
    radius = np.sqrt(1-z*z)
    return np.c_[radius*np.cos(phi), radius*np.sin(phi), z]


def load_inputs(translation_mm=100.):
    mesh = load_shape(ROOT/'objects/B/mesh.stl')
    source = ROOT/'whole_task_2/data/B/step1_reach'
    plan = json.loads((source/'plan.json').read_text())
    robot_params = ROOT/'whole_task_2/idea/robot_params/certification.json'
    center = np.array(json.loads(robot_params.read_text())['workspace_center_world_m'])
    setup = deepcopy(plan['setup'])
    setup.update(max_tilt_deg=25., working_direction_world=[0., -2**-.5, -2**-.5])
    setup['surface_workspace'] = dict(shape='cube', workspace_size_world_m=[.1]*3,
        workspace_center_world_m=center.tolist(), bounds_world_m=[(center-.05).tolist(), (center+.05).tolist()],
        applies_to='surface_process_point')
    if translation_mm is None:
        setup.pop('surface_workspace')
    files = [ROOT/'objects/B/mesh.stl', source/'plan.json', source/'screen.npz', robot_params,
             ROOT/'robots_config/04_kuka_lbr_med14_r820.json',
             *[ROOT/'whole_task_2/step1'/name for name in
               ('reach.py','arm.py','plan_reach.py','workspace.py','reach_video.py')]]
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    return mesh, setup, inputs


def build_directions(mesh, setup, count=72):
    with np.load(ROOT/'whole_task_2/data/B/step1_reach/screen.npz') as z:
        axes, reach = z['axes'].copy(), z['reach'].copy()
        key = json.loads(str(z['key']))
    assert key['shape'] == sha256(ROOT/'objects/B/mesh.stl')
    assert key['reach'] == sha256(ROOT/'whole_task_2/step1/reach.py')
    nominal = np.vstack([fibonacci(count), np.eye(3), -np.eye(3)])
    forced = []
    missing = np.flatnonzero(~reach.any(axis=0))
    if len(missing):
        for face in missing:
            # Mesh-specific hints are rechecked geometrically each time. They
            # only repair demonstrated depth-map false negatives, never fill a
            # missing patch by convention. No certificate files are needed.
            directions = np.array([a for a in ACCESS_HINTS
                if R.certify(mesh,int(face),a,.1,.02,.002,.0005,None)[0]])
            if not len(directions):
                directions = []
                for a in R.sphere_axes(2.):
                    if -mesh.face_normals[face]@a <= 0:
                        continue
                    if R.certify(mesh,int(face),a,.1,.02,.002,.0005,None)[0]:
                        directions.append(a)
                        if len(directions) == 3:
                            break
                directions = np.asarray(directions)
            if not len(directions):
                raise ValueError('No exact access witness found for face '+str(face))
            chosen = [0]
            for _ in range(min(2, len(directions)-1)):
                score = np.max(directions @ directions[chosen].T, axis=1)
                score[chosen] = np.inf
                chosen.append(int(np.argmin(score)))
            for axis in directions[chosen]:
                if not R.certify(mesh, int(face), axis, .1, .02, .002, .0005, None)[0]:
                    raise ValueError('Exact access witness changed')
                forced.append((int(face), axis))
                nominal = np.vstack([nominal, axis])
    pixel = .004*np.linalg.norm(mesh.extents)
    samples = int(6*mesh.area/pixel**2)
    screen = R.Screen(mesh, .1, .02, .002, pixel, samples, None)
    extra = nominal.copy()
    extra_masks = np.array([screen(a) for a in extra])
    # This corrects a demonstrated false negative with a real clearance witness.
    # It is not an uncertified hole fill.
    for face, axis in forced:
        extra_masks[np.all(np.isclose(extra, axis, atol=1e-12), axis=1), face] = True
    axes = np.vstack([axes, extra])
    reach = np.vstack([reach, extra_masks])
    return nominal, axes, reach


def anchors(mesh, mask, count=10):
    ids = np.flatnonzero(mask)
    points = mesh.triangles_center[ids]
    mean = np.average(points, weights=mesh.area_faces[ids], axis=0)
    chosen = [int(np.argmin(np.linalg.norm(points-mean, axis=1)))]
    distance = np.linalg.norm(points-points[chosen[0]], axis=1)
    for _ in range(min(count-1, len(ids)-1)):
        k = int(np.argmax(distance))
        chosen.append(k)
        distance = np.minimum(distance, np.linalg.norm(points-points[k], axis=1))
    return np.vstack([mean, points[chosen]])


def spatial_candidates(mesh, setup, rotation, visible, aiming_points, z_offsets=(-.035, 0., .035)):
    world_vertices = mesh.vertices @ rotation.T
    triangles = world_vertices[mesh.faces]
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    bounds = np.asarray(setup['surface_workspace']['bounds_world_m'])
    center = bounds.mean(axis=0)
    floor_shift = setup['table_height_m']-world_vertices[:, 2].min()
    masks, poses, seen = [], [], set()
    for point in aiming_points:
        for dz in z_offsets:
            shift = center-rotation@point
            shift[2] = max(shift[2]+dz, floor_shift)
            key = tuple(np.round(shift, 9))
            if key in seen:
                continue
            seen.add(key)
            mask = visible & ((lo+shift >= bounds[0]-1e-9) & (hi+shift <= bounds[1]+1e-9)).all(axis=1)
            if not mask.any():
                continue
            W = np.eye(4); W[:3, :3] = rotation; W[:3, 3] = shift
            poses.append(W); masks.append(mask)
    return poses, masks


def candidates(mesh, setup, nominal, axes, reach, rolls=3):
    poses, masks, direction, descriptions = [], [], [], []
    seen = set()
    world_axis = np.asarray(setup['working_direction_world'])
    for n, axis in enumerate(nominal):
        cone = axes@axis >= np.cos(np.radians(25.))-1e-12
        visible = reach[cone].any(axis=0)
        if not visible.any():
            continue
        points = anchors(mesh, visible)
        base = onto(axis, world_axis)
        for roll in np.linspace(0., 360., rolls, endpoint=False):
            rotation = Rotation.from_rotvec(world_axis*np.radians(roll)).as_matrix() @ base
            local_poses, local_masks = spatial_candidates(mesh, setup, rotation, visible, points)
            if not local_masks:
                continue
            local_masks = np.asarray(local_masks)
            target = local_masks.any(axis=0)
            keep = greedy(local_masks, mesh.area_faces, target)
            for k in keep:
                W = local_poses[k]
                key = tuple(np.round(W[:3].ravel(), 9))
                if key in seen:
                    continue
                seen.add(key)
                poses.append(W); masks.append(local_masks[k]); direction.append(axis)
                descriptions.append(dict(direction_index=n, roll_deg=float(roll), kind='initial'))
        if n % 12 == 0:
            print('CANDIDATE DIRECTIONS', n+1, '/', len(nominal), 'poses', len(poses), flush=True)
    return np.asarray(poses), np.asarray(masks, bool), np.asarray(direction), descriptions


def targeted(mesh, setup, axes, reach, faces, seen_poses, max_axes=5):
    poses, masks, nominal, descriptions = [], [], [], []
    a_world = np.asarray(setup['working_direction_world'])
    for face in faces:
        ids = np.flatnonzero(reach[:, face])
        ids = ids[np.argsort(-(axes[ids] @ -mesh.face_normals[face]), kind='stable')]
        good = []
        for k in ids:
            if good and np.max(axes[good]@axes[k]) > np.cos(np.radians(12.)):
                continue
            if R.certify(mesh, int(face), axes[k], .1, .02, .002, .0005, None)[0]:
                good.append(k)
            if len(good) == max_axes:
                break
        for k in good:
            axis = axes[k]
            visible = reach[axes@axis >= np.cos(np.radians(25.))-1e-12].any(axis=0)
            visible[face] = True
            for roll in np.arange(0., 360., 30.):
                rotation = Rotation.from_rotvec(a_world*np.radians(roll)).as_matrix() @ onto(axis, a_world)
                local_poses, local_masks = spatial_candidates(mesh, setup, rotation, visible,
                                                             mesh.triangles_center[[face]])
                for W, mask in zip(local_poses, local_masks):
                    key = tuple(np.round(W[:3].ravel(), 9))
                    if key in seen_poses or not mask[face]:
                        continue
                    seen_poses.add(key)
                    poses.append(W); masks.append(mask); nominal.append(axis)
                    descriptions.append(dict(kind='targeted', source_face=int(face), roll_deg=float(roll)))
    return poses, masks, nominal, descriptions


def certify_candidates(mesh, setup, poses, masks, nominal, axes, reach, indices, workers=8):
    """Exact tool clearances followed by four-point IK, links and table checks."""
    evidence = {}
    jobs = []
    for candidate in indices:
        mask = masks[candidate]
        cosine = axes @ nominal[candidate]
        cone = cosine >= np.cos(np.radians(25.))-1e-12
        data = dict(mask=np.zeros(len(mesh.faces), bool), axis_object=np.zeros((len(mesh.faces), 3)),
                    gaps=np.zeros((len(mesh.faces), 2)), joints=np.zeros((len(mesh.faces), 4, 7)),
                    arm_code=np.full(len(mesh.faces), -1, dtype=np.int8))
        evidence[int(candidate)] = data
        for face in np.flatnonzero(mask):
            ids = np.flatnonzero(cone & reach[:, face])
            ids = ids[np.argsort(-cosine[ids], kind='stable')]
            if len(ids):
                jobs.append((int(face), [(int(candidate), int(k), axes[k]) for k in ids],
                             .1, .02, .002, .0005, None))
    print('EXACT CERTIFICATION', len(indices), 'poses,', len(jobs), 'face/pose pairs', flush=True)
    with ProcessPoolExecutor(workers, initializer=P._init_mesh,
                             initargs=(str(ROOT/'objects/B/mesh.stl'), setup)) as pool:
        for face, candidate, k, needle, head, _ in pool.map(P._certify, jobs, chunksize=12):
            if candidate is None:
                continue
            d = evidence[candidate]
            d['mask'][face] = True
            d['axis_object'][face] = axes[k]
            d['gaps'][face] = needle, head
        chunks = []
        for candidate in indices:
            d = evidence[int(candidate)]; W = poses[candidate]
            faces = np.flatnonzero(d['mask'])
            faces = faces[A.order(mesh.triangles_center[faces] @ W[:3, :3].T)]
            chunks.extend((int(candidate), W, [(int(f), d['axis_object'][f]) for f in faces[k:k+60]])
                          for k in range(0, len(faces), 60))
        for rows in pool.map(P._arm_pairs, chunks):
            for face, candidate, code, Q in rows:
                d = evidence[candidate]
                d['arm_code'][face] = code
                d['joints'][face] = Q
                d['mask'][face] = code == A.OK
    for candidate in indices:
        d = evidence[int(candidate)]
        print('CHECKED POSE', int(candidate), int(d['mask'].sum()), '/', int(masks[candidate].sum()), 'faces', flush=True)
    return evidence


def replay_chunk(job):
    W, items = job
    from whole_task_2.step1 import plan_reach as state
    mesh, checker = state._mesh, state._arm
    count = 0
    for face, axis, joints in items:
        if not R.certify(mesh, face, axis, .1, .02, .002, .0005, None)[0]:
            raise ValueError('Tool clearance replay failed for source face '+str(face))
        if not checker.replay(face, axis, W, joints):
            raise ValueError('Arm replay failed for source face '+str(face))
        count += 1
    return count


def validate_solution(mesh, setup, poses, selected, owner, evidence, workers=8):
    if owner.shape != (len(mesh.faces),) or (owner < 0).any():
        raise ValueError('Every original face must have an actual working task')
    bounds = np.asarray(setup['surface_workspace']['bounds_world_m']) if 'surface_workspace' in setup else None
    world_axis = np.asarray(setup['working_direction_world'])
    jobs = []
    for task, candidate in enumerate(selected):
        W = poses[candidate]
        if (mesh.vertices @ W[:3, :3].T+W[:3, 3])[:, 2].min() < setup['table_height_m']-1e-9:
            raise ValueError('An object pose goes below the fixed table')
        ids = np.flatnonzero(owner == task)
        proof = evidence[int(candidate)]
        if not proof['mask'][ids].all():
            raise ValueError('Assigned face failed this pose check')
        axes, joints = proof['axis_object'][ids], proof['joints'][ids]
        Pworld = R.test_points(mesh)[ids] @ W[:3, :3].T+W[:3, 3]
        if bounds is not None and not points_inside_box(Pworld, bounds).all():
            raise ValueError('Assigned triangle leaves the fixed work cube')
        if not ((axes @ W[:3, :3].T) @ world_axis >= np.cos(np.radians(25.))-1e-12).all():
            raise ValueError('Assigned tool axis leaves the fixed direction cone')
        items = [(int(f), a, q) for f, a, q in zip(ids, axes, joints)]
        jobs.extend((W, items[k:k+40]) for k in range(0, len(items), 40))
    with ProcessPoolExecutor(workers, initializer=P._init_mesh,
                             initargs=(str(ROOT/'objects/B/mesh.stl'), setup)) as pool:
        count = sum(pool.map(replay_chunk, jobs))
    if count != len(mesh.faces):
        raise ValueError('Replay did not cover every source face exactly once')
    result = dict(status='passed', source_face_count=len(mesh.faces), replayed_face_pose_pairs=count,
                  assigned_exactly_once=True, full_source_surface_covered=True,
                  work_cube_fixed=True, robot_base_fixed=True, world_direction_and_cone_fixed=True,
                  exact_needle_head_clearance=True, four_point_IK_and_link_collision_replayed=True,
                  object_above_fixed_table=True, support_or_repositioning_path_certified=False)
    return result


def unlimited_candidates(mesh, setup, nominal, direction_masks):
    """One initial placement per sampled direction, without an artificial box."""
    world_axis = np.asarray(setup['working_direction_world'])
    center = np.array([-.0033003284091440337, -.6641652019446095, .325])
    poses = []
    for axis in nominal:
        rotation = onto(axis, world_axis)
        W = np.eye(4); W[:3,:3] = rotation
        W[:3,3] = center-rotation@mesh.centroid
        W[2,3] = max(W[2,3],setup['table_height_m']+.004-(mesh.vertices@rotation.T)[:,2].min())
        poses.append(W)
    return np.asarray(poses), direction_masks.copy(), nominal.copy()


def retry_pair(job):
    face, candidate, W, axes = job
    for axis in axes:
        passed, needle, head = R.certify(P._mesh,face,axis,.1,.02,.002,.0005,None)
        if not passed:
            continue
        code, joints = P._arm.pair(face,axis,W)
        if code == A.OK:
            return face, candidate, axis, joints, (needle,head)
    return face, candidate, None, None, None


def rescue_faces(mesh, setup, poses, nominal, axes, reach, evidence, faces, workers=8):
    """Try other tool axes for still-missing faces before adding object poses."""
    jobs = []
    for candidate, proof in evidence.items():
        cosine = axes@nominal[candidate]
        cone = cosine >= np.cos(np.radians(25.))-1e-12
        for face in faces:
            ids = np.flatnonzero(cone & reach[:,face])
            ids = ids[np.argsort(-cosine[ids],kind='stable')]
            # These jobs are only for the few holes, so test the full sampled
            # cone rather than assuming its first geometry-valid axis is best.
            if len(ids):
                jobs.append((int(face),int(candidate),poses[candidate],axes[ids]))
    with ProcessPoolExecutor(workers,initializer=P._init_mesh,
                             initargs=(str(ROOT/'objects/B/mesh.stl'),setup)) as pool:
        for face,candidate,axis,joints,gaps in pool.map(retry_pair,jobs):
            if axis is None:
                continue
            proof = evidence[candidate]
            proof['mask'][face] = True
            proof['axis_object'][face] = axis
            proof['joints'][face] = joints
            proof['gaps'][face] = gaps
            proof['arm_code'][face] = A.OK


def recheck_placement(mesh, setup, W, proof, workers=8):
    """Translation preserves object/tool clearance; redo all arm/table checks."""
    result = {k:v.copy() for k,v in proof.items()}
    result['mask'][:] = False
    faces = np.flatnonzero(np.linalg.norm(proof['axis_object'],axis=1)>.9)
    faces = faces[A.order(mesh.triangles_center[faces]@W[:3,:3].T)]
    jobs = [(0,W,[(int(f),proof['axis_object'][f]) for f in faces[j:j+60]])
            for j in range(0,len(faces),60)]
    with ProcessPoolExecutor(workers,initializer=P._init_mesh,
                             initargs=(str(ROOT/'objects/B/mesh.stl'),setup)) as pool:
        for rows in pool.map(P._arm_pairs,jobs):
            for face,_,code,Q in rows:
                result['arm_code'][face] = code
                result['joints'][face] = Q
                result['mask'][face] = code == A.OK
    return result


def adjust_unlimited(mesh,setup,poses,nominal,axes,reach,evidence,workers=8):
    """Move existing poses to fill IK holes before increasing the pose count."""
    if 'surface_workspace' in setup:
        raise ValueError('This placement search is for unlimited translation')
    shifts = ((0.,-.04,0.),(0.,-.08,0.),(0.,0.,.025),(0.,0.,.05),
              (0.,.04,0.),(0.,-.04,.025),(0.,.04,.025))
    for candidate in list(evidence):
        union = np.array([p['mask'] for p in evidence.values()]).any(0)
        missing = ~union
        if not missing.any():
            return
        proof = evidence[candidate]
        targets = missing & (np.linalg.norm(proof['axis_object'],axis=1)>.9)
        if not targets.any():
            continue
        others = [p['mask'] for k,p in evidence.items() if k!=candidate]
        other_union = np.array(others).any(0) if others else np.zeros(len(mesh.faces),bool)
        required = proof['mask'] & ~other_union
        for shift in shifts:
            W = poses[candidate].copy(); W[:3,3] += shift
            if (mesh.vertices@W[:3,:3].T+W[:3,3])[:,2].min()<setup['table_height_m']-1e-9:
                continue
            trial = recheck_placement(mesh,setup,W,proof,workers)
            lost = required & ~trial['mask']
            if lost.any() or not (targets & trial['mask']).any():
                new_poses = poses.copy();new_poses[candidate]=W
                rescue_faces(mesh,setup,new_poses,nominal,axes,reach,{candidate:trial},
                             np.flatnonzero(lost|targets),workers)
                lost = required & ~trial['mask']
            if not lost.any() and (targets & trial['mask']).any():
                poses[candidate] = W
                evidence[candidate] = trial
                print('ADJUSTED EXISTING POSE',candidate,'shift',shift,
                      'new faces',int((missing&trial['mask']).sum()),flush=True)
                break
