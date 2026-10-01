"""Search grounded task poses containing a floor-compatible multi-pose subset.

Candidate work patches and load rules are fixed before compatibility scoring.
Sparse samples may reject/prioritize candidates, but acceptance always replays
all 32768 original loads at the final saved transforms and work patches.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import trimesh

from sequence_export import ROOT, WorkRegions, digest, write

sys.path.insert(0, str(ROOT/'slides/baseline_algo'))
from step1.needs import ContinuousNeeds, sample_needs, DEFAULT_SAMPLE_COUNT, DEFAULT_SAMPLE_SEED, K, RAY_OFFSET_M
from cone_model import frame, CONE_HALF_DEG


def domain_for_pose(mesh, transform, mask, name):
    """The same mesh, area measure, cone and sampling implementation as Step1."""
    rotation, offset = transform[:3, :3], transform[:3, 3]
    world = trimesh.Trimesh(mesh.vertices@rotation.T+offset, mesh.faces, process=False)
    normals = -(mesh.face_normals@rotation.T)[mask]
    e1, e2 = frame(normals)
    return ContinuousNeeds(dict(object=name,
        frame=dict(moment_origin_m=(rotation@mesh.center_mass+offset).tolist()),
        load=dict(K=K, gravity_force_mg=[0., 0., -1.], magnitude_range_mg=[0., K], cone_half_deg=CONE_HALF_DEG),
        reachability=dict(ray_offset_m=RAY_OFFSET_M),
        geometry=dict(vertices_m=world.vertices, faces=world.faces, work_face_ids=np.flatnonzero(mask),
            inward_normals=normals, tangent1=e1, tangent2=e2, work_face_areas_m2=world.area_faces[mask])))


def local_demands(mesh, transform, mask, name, count=DEFAULT_SAMPLE_COUNT):
    domain = domain_for_pose(mesh, transform, mask, name)
    samples = sample_needs(domain, count=count, seed=DEFAULT_SAMPLE_SEED)
    loads = np.asarray(samples['need_wrench'])
    moments = loads[:, 3:]+np.cross(domain.com, loads[:, :3])
    points = np.column_stack((-moments[:, 1]/loads[:, 2], moments[:, 0]/loads[:, 2], np.zeros(count)))
    # All heights are affine: vertices of the full 2D hull are an exact reduction.
    hull = ConvexHull(points[:, :2]).vertices
    local = (points-transform[:3, 3])@transform[:3, :3]
    return local[hull], local, samples


def compatible(a, b, tolerance=1e-9):
    return (np.min(a.cloud@b.transform[2, :3]+b.transform[2, 3]) >= -tolerance
            and np.min(b.cloud@a.transform[2, :3]+a.transform[2, 3]) >= -tolerance)


def find_clique(adjacency, size, required=()):
    """Exact bounded-size clique search; a greedy prefix cannot hide a witness."""
    if size < 1:
        raise ValueError('Positive clique size required')
    required = tuple(required)
    if len(required) > size or len(set(required)) != len(required):
        raise ValueError('Invalid required vertices')
    if any(j not in adjacency[i] for i, j in itertools.combinations(required, 2)):
        return None
    candidates = set(range(len(adjacency)))-set(required)
    for i in required:
        candidates &= adjacency[i]
    def visit(chosen, remaining):
        if len(chosen) == size:
            return chosen
        if len(chosen)+len(remaining) < size:
            return None
        for vertex in sorted(remaining, key=lambda i: (-len(adjacency[i] & remaining), i)):
            remaining.remove(vertex)
            if len(chosen)+1+len(remaining) < size:
                break
            found = visit(chosen+[vertex], remaining & adjacency[vertex])
            if found is not None:
                return found
        return None
    return visit(list(required), candidates)


@dataclass
class Candidate:
    index: int
    transform: np.ndarray
    mask: np.ndarray
    area: dict
    cloud: np.ndarray
    full: bool = False


def seat(mesh, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:2, 3] = -(rotation@mesh.center_mass)[:2]
    transform[2, 3] = -(mesh.vertices@rotation.T)[:, 2].min()
    return transform


def separated(a, b, angle_deg=18.):
    return float(a[2, :3]@b[2, :3]) <= np.cos(np.radians(angle_deg))+1e-12


def search(name, destination, count=20, compatible_size=5, seed=20260929, budget=1600):
    if not 2 <= compatible_size <= count or budget < count:
        raise ValueError('Require 2 <= compatible-size <= pose-count <= candidate-budget')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT/'objects'/name
    mesh = trimesh.load(source/'mesh.stl', force='mesh')
    rest = np.asarray(json.loads((source/'poses.json').read_text())['rest']['T_world_mesh'])
    regions = WorkRegions(mesh)
    rng = np.random.default_rng(seed)
    candidates, adjacency = [], []
    began = time.monotonic()
    chosen = None
    for attempt in range(budget):
        # Cover the full sphere; robot feasibility is a separate later check.
        rotation = Rotation.random(random_state=rng).as_matrix()
        transform = seat(mesh, rotation)
        vertices = mesh.vertices@rotation.T+transform[:3, 3]
        pivot = vertices[np.argmin(vertices[:, 2])]
        com = rotation@mesh.center_mass+transform[:3, 3]
        if (not separated(transform, rest) or np.count_nonzero(vertices[:, 2] < 1e-8) != 1
                or np.linalg.norm(com[:2]-pivot[:2]) < .001):
            continue
        try:
            mask, area = regions.choose(transform, f'{name}/compatible_{seed}_{attempt}')
        except ValueError:
            continue
        cloud, _, _ = local_demands(regions.mesh, transform, mask, name, 1024)
        candidate = Candidate(attempt, transform, mask, area, cloud)
        neighbors = {i for i, old in enumerate(candidates)
                     if separated(transform, old.transform) and compatible(candidate, old)}
        index = len(candidates)
        candidates.append(candidate); adjacency.append(neighbors)
        for i in neighbors:
            adjacency[i].add(index)
        while True:
            witness = find_clique(adjacency, compatible_size, required=[index])
            if witness is None:
                break
            for i in witness:
                c = candidates[i]
                if not c.full:
                    c.cloud, _, _ = local_demands(regions.mesh, c.transform, c.mask, name)
                    c.full = True
                    for j in list(adjacency[i]):
                        if not compatible(c, candidates[j]):
                            adjacency[i].remove(j); adjacency[j].remove(i)
            if all(j in adjacency[i] for i, j in itertools.combinations(witness, 2)):
                chosen = witness
                break
        if index % 20 == 0 or chosen is not None:
            print(f'POSE SEARCH attempts={attempt+1} candidates={len(candidates)} '
                  f'edges={sum(map(len, adjacency))//2} witness={chosen} '
                  f'seconds={time.monotonic()-began:.1f}', flush=True)
        if chosen is not None:
            selected = list(chosen)
            # Fill the remainder with genuinely different gravity directions.
            for i in sorted(range(len(candidates)), key=lambda j: (-len(adjacency[j]), j)):
                if i not in selected and all(separated(candidates[i].transform, candidates[j].transform) for j in selected):
                    selected.append(i)
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
            chosen = None
    else:
        write(destination/'failure.json', dict(status='candidate_budget_exhausted',
            candidate_count=len(candidates), budget=budget, seed=seed, requested_count=count,
            compatible_size=compatible_size, no_global_infeasibility_claim=True))
        raise RuntimeError(f'No {count}-pose set with a certified {compatible_size}-clique in {budget} candidates')
    records = []
    for position, i in enumerate(selected, 1):
        c = candidates[i]
        cloud, full_cloud, samples = local_demands(regions.mesh, c.transform, c.mask, name)
        c.cloud, c.full = cloud, True
        np.savez_compressed(destination/f'pose_{position}.npz', T_world_mesh=c.transform,
            work_faces=c.mask, floor_demands_object_m=full_cloud,
            load_wrenches=np.asarray(samples['need_wrench']))
        records.append(dict(pose_id=f'pose_{position}', candidate_index=c.index,
            T_world_mesh=c.transform.tolist(), area=c.area, arrays=f'pose_{position}.npz',
            sha256=digest(destination/f'pose_{position}.npz')))
    accepted = [candidates[i] for i in selected]
    edges = [[j for j, other in enumerate(accepted) if j != i and compatible(c, other)]
             for i, c in enumerate(accepted)]
    witnesses = {str(k): [records[i]['pose_id'] for i in find_clique([set(x) for x in edges], k)]
                 for k in range(3, compatible_size+1)}
    report = dict(schema='floor_compatible_pose_plan_v1', object=name, seed=seed,
        pose_count=count, compatible_size=compatible_size, poses=records, witnesses=witnesses,
        compatibility=edges, original_sample_count=DEFAULT_SAMPLE_COUNT, sample_seed=DEFAULT_SAMPLE_SEED,
        floor_tolerance_m=1e-9, minimum_pairwise_gravity_direction_deg=18.,
        work_area_fraction=[.06, .10], K=K, cone_half_deg=CONE_HALF_DEG,
        candidate_count=len(candidates), attempted_count=attempt+1, budget=budget,
        robot_trajectory_verified=False, complete_fixture_verified=False,
        mesh_sha256=digest(source/'mesh.stl'), rest=dict(T_world_mesh=rest.tolist()),
        uniform_subdivision_rounds=regions.rounds,
        generator='codes/setup/compatible_pose_search.py', generator_sha256=digest(__file__))
    write(destination/'plan.json', report)
    print('POSE PLAN COMPLETE', witnesses, destination, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pose-count', type=int, default=20)
    parser.add_argument('--compatible-size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--candidate-budget', type=int, default=1600)
    parser.add_argument('--publish', action='store_true', help='Replace active task poses after validation; preserve old inputs and outputs as history')
    args = parser.parse_args()
    search(args.object, args.output, args.pose_count, args.compatible_size, args.seed, args.candidate_budget)
    if args.publish:
        from compatible_pose_export import publish
        publish(args.object, args.output)
