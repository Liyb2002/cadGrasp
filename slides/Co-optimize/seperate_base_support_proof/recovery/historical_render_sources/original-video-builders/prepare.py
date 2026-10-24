"""Prepare truthful geometry for four explanatory videos; upstream is read-only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
sys.path.insert(0, str(PROOF))
import diagnose as D
import certify as K
import base_seating as B
import base_geometry as G
import partial_ground as P
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from work_access import tangent_frame

C, np = D.C, D.np


def load(label):
    inputs = C.Inputs('B', D.BASE/label/'data/selection.json')
    inputs.base = D.BASE
    data = inputs.load(inputs.cases[0])
    B.load_demands(data, inputs)
    return inputs, data


def hull_common(row):
    return C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform']))


def lift(data, heights):
    heights = np.asarray(heights)
    return B.shifted(data, dict(fixture_lift_m=heights,
        total_object_world_translation_m=np.c_[np.zeros((len(heights), 2)), heights]))


def blend_seating(data, solution, fraction):
    return B.shifted(data, dict(fixture_lift_m=np.asarray(solution['fixture_lift_m'])*fraction,
        total_object_world_translation_m=np.asarray(solution['total_object_world_translation_m'])*fraction))


def floor_values(data):
    values = []
    for i, row in enumerate(data['rows']):
        a, b = P.floor_inequalities(data['states'], i)
        for j in range(len(data['rows'])):
            if i != j:
                values.append(dict(demand_pose=row['pose'], floor_pose=data['rows'][j]['pose'],
                    min_original_demand_height_mm=float((row['cloud'][:, :2]@a[j]+b[j]).min()*1000)))
    return values


def rim(row, width=.0015, height=.003, expansion=0., offset=(0., 0.)):
    inner = Polygon(row['hull'][:, :2]).buffer(expansion, join_style=2)
    outer = inner.buffer(width, join_style=2)
    contours = [(np.asarray(p.exterior.coords)[:-1]+offset)*1000 for p in (outer, inner)]
    value = G.md.CrossSection(contours, G.md.FillRule.EvenOdd).extrude(height*1000)
    return G.transform(value, np.linalg.inv(row['fixture_transform']))


def join_to_support(value, data, states=None):
    """A short visual connector; no force/contact capacity claim."""
    states = data['states'] if states is None else states
    vertices = G.unpack(value).vertices
    support = data['fixture'].vertices
    mask = np.ones(len(support), bool)
    for state in states:
        mask &= C.F.transform_points(support, state)[:, 2] > .001
    candidates = support[mask]
    if not len(candidates):
        candidates = support
    i, j = np.unravel_index(np.argmin(cdist(vertices, candidates)), (len(vertices), len(candidates)))
    corners = np.array([[x, y, z] for x in [-.001, .001] for y in [-.001, .001] for z in [-.001, .001]])
    connector = G.solid(C.trimesh.convex.convex_hull(np.vstack([vertices[i]+corners, candidates[j]+corners])))
    if states is not None:
        for state in states:
            connector = connector.trim_by_plane(tuple(state[2, :3]), -float(state[2, 3])*1000)
    return value + connector


def one_pose_interval(data, active):
    low, high = 0., float('inf')
    witnesses = []
    for i, row in enumerate(data['rows']):
        for j, state in enumerate(data['states']):
            if i == j:
                continue
            transform = state@np.linalg.inv(row['fixture_transform'])
            a, c, b = transform[2, :2], transform[2, 2], transform[2, 3]
            start = row['cloud'][:, :2]@a+b
            coefficient = float(j == active) - (row['horizontal_force_ratio']@a+c)*float(i == active)
            positive, negative = coefficient > 1e-12, coefficient < -1e-12
            if positive.any():
                bound = float((-start[positive]/coefficient[positive]).max())
                low = max(low, bound)
                witnesses.append(dict(kind='lower', demand_pose=row['pose'], floor_pose=data['rows'][j]['pose'], height_mm=bound*1000))
            if negative.any():
                bound = float((-start[negative]/coefficient[negative]).min())
                high = min(high, bound)
                witnesses.append(dict(kind='upper', demand_pose=row['pose'], floor_pose=data['rows'][j]['pose'], height_mm=bound*1000))
    return dict(active_pose=data['rows'][active]['pose'], lower_mm=low*1000,
        upper_mm=high*1000, feasible=low <= high, bounds=witnesses,
        applies_to_all_original_pressure_centers=True,
        raised_pose_object_ground_contacts_removed=True)


def both_z_lp(data):
    """Full 65,536-row pure-Z system, without extra wrap/head restrictions."""
    matrices, rhs = [], []
    for i, row in enumerate(data['rows']):
        for j, state in enumerate(data['states']):
            if i == j:
                continue
            t = state@np.linalg.inv(row['fixture_transform'])
            a, c, b = t[2, :2], t[2, 2], t[2, 3]
            matrix = np.zeros((len(row['cloud']), 2))
            matrix[:, i] = row['horizontal_force_ratio']@a+c
            matrix[:, j] -= 1.
            matrices.append(matrix)
            rhs.append((row['cloud'][:, :2]@a+b)*1000)
    answer = linprog(np.ones(2), A_ub=np.vstack(matrices), b_ub=np.concatenate(rhs),
        bounds=[(0., None)]*2, method='highs')
    assert answer.status == 2, answer.message
    return dict(infeasible=True, status=answer.status, original_constraint_count=sum(len(v) for v in rhs),
        head_wrap_constraints_omitted=True, bounding_box_omitted=True,
        note='Both positive lifts remove own ground contacts. The two one-zero boundary cases are rejected by the per-pose separators.')


def ground_base(data):
    parts = [rim(row, width=.001, height=.002) for row in data['rows']]
    joined = [join_to_support(value, data) for value in parts]
    base = G.combine(joined)
    for state in data['states']:
        base = base.trim_by_plane(tuple(state[2, :3]), -float(state[2, 3])*1000)
    mesh = G.unpack(base)
    checks = []
    for row in data['rows']:
        posed = C.F.transform_points(mesh.vertices, row['fixture_transform'])
        contacts = posed[np.abs(posed[:, 2]) <= 1e-8, :2]
        covered = P.covered(row['cloud'][:, :2], np.vstack([contacts, row['object_ground']]))
        checks.append(dict(pose=row['pose'], minimum_base_z_mm=float(posed[:, 2].min()*1000),
            actual_ground_vertex_count=len(contacts), covered_original_demands=int(covered.sum()),
            original_demands=len(covered)))
        assert posed[:, 2].min() >= -1e-8 and covered.all(), checks[-1]
    return base, checks


def work_geometry(data, inputs):
    tasks, body = D.load_tasks(data, inputs)
    accesses = [K.inner_access(body, task) for task in tasks]
    certificate = json.loads((PROOF/'certificates'/data['case']['id']/'report.json').read_text())
    proof = next(v for v in certificate['work_per_pose'] if v.get('unbounded_proof_found') and not v['passed'])
    target_row = next(row for row in data['rows'] if row['pose'] == proof['pose'])
    target = C.F.transform_points([list(proof['separator']['demand_xy_m'])+[0.]], np.linalg.inv(target_row['fixture_transform']))[0]
    patches, arrows, work_planes, solids, rays = [], [], [], [], []
    for row, task, access in zip(data['rows'], tasks, accesses):
        placement = np.linalg.inv(row['fixture_transform'])@row['object_transform']
        local = np.linalg.inv(placement)
        triangles = C.F.transform_points(body.triangles[task.domain.work_ids].reshape(-1, 3), placement).reshape(-1, 3, 3)
        normals = body.face_normals[task.domain.work_ids]@placement[:3, :3].T
        patches.append((triangles, row['color']))
        half_angle = float(task.domain.data['load']['cone_half_deg'])
        for index, (face, triangle, normal, count, family) in enumerate(zip(task.domain.work_ids, triangles, normals, access.counts, access.planes)):
            origin = triangle.mean(0)
            tangent = tangent_frame(normal)
            directions = [normal]
            if index % 8 == 0:
                directions += [np.cos(np.radians(half_angle))*normal + np.sin(np.radians(half_angle))*(np.cos(a)*tangent[:, 0]+np.sin(a)*tangent[:, 1])
                               for a in np.arange(6)*2*np.pi/6]
            for direction in directions:
                arrows.append((origin.copy(), direction.copy(), row['color']))
            planes = family[:count]
            work_planes.append((planes[:, :3]@local[:3, :3], planes[:, :3]@local[:3, 3]+planes[:, 3]))
            # The solid is capped beyond every displayed base candidate.
            phases = (np.arange(32)+.5)*2*np.pi/32
            directions_ring = normal+np.tan(np.radians(half_angle))*(np.cos(phases)[:, None]*tangent[:, 0]+np.sin(phases)[:, None]*tangent[:, 1])
            far = (triangle[:, None]+.5*directions_ring[None]).reshape(-1, 3)
            solids.append(G.solid(C.trimesh.convex.convex_hull(np.vstack([triangle, far]))))
            for source in np.vstack([triangle, origin]):
                difference = target-source
                distance = np.linalg.norm(difference)
                direction = difference/distance
                angle = float(np.degrees(np.arccos(np.clip(direction@normal, -1., 1.))))
                if angle < half_angle-1 and distance > .005:
                    rays.append(dict(pose=row['pose'], face=int(face), origin_m=source.tolist(),
                        direction=direction.tolist(), target_m=target.tolist(), target_distance_m=float(distance), angle_deg=angle,
                        original_cone_half_angle_deg=half_angle))
    chosen = min(rays, key=lambda r: r['angle_deg'])
    work_union = G.combine(solids)
    # Witness is on the inner edge of the rim, so show the ray extending past it.
    candidates = []
    for expansion, offset in [(0., (0., 0.)), (.014, (0., 0.)), (.026, (.014, -.008))]:
        base = rim(target_row, width=.002, height=.003, expansion=expansion, offset=offset)
        common = G.unpack(base)
        local = C.F.transform_points(common.vertices, target_row['fixture_transform'])
        # Test coverage from actual contact vertices; later examples must still enclose H.
        contact = local[np.abs(local[:, 2]) <= 1e-8, :2]
        assert P.covered(target_row['cloud'][:, :2], contact).all()
        cut = base ^ work_union
        assert G.volume(cut) > 1e-12
        candidates.append((base, cut, dict(expansion_mm=expansion*1000, offset_xy_mm=(np.asarray(offset)*1000).tolist(),
            covers_all_32768_pose14_demands=True, work_collision_cm3=G.volume(cut)*1e6)))
    return dict(tasks=tasks, body=body, proof=proof, patches=patches, arrows=arrows,
        planes=work_planes, work_union=work_union, witness=chosen, candidates=candidates,
        target_row=target_row)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs_a, pair = load('illegal/pose11+19')
    inputs_b, four = load('pose1+11+14+27')
    for inputs, data in [(inputs_a, pair), (inputs_b, four)]:
        path = PROOF/'certificates'/data['case']['id']/'report.json'
        inputs.protected[C.relative(path)] = C.digest(path)
    small = rim(pair['rows'][1], width=.002, height=.003)
    large = rim(pair['rows'][1], width=.002, height=.003, expansion=.024)
    solution = B.Seating(pair, width=.005, preserve_material=True, protect_full_wrap=False).solve('z_xy')
    assert solution['passed']
    moved = blend_seating(pair, solution, 1.)
    base, ground_checks = ground_base(moved)
    work = work_geometry(four, inputs_b)
    path = PROOF/'certificates'/four['case']['id']/f'{work["target_row"]["pose"]}_work.wkt'
    inputs_b.protected[C.relative(path)] = C.digest(path)
    for name, value in [('ground_minimal', small), ('ground_expanded', large), ('ground_feasible_xyz', base)]:
        G.unpack(value).export(output/f'{name}.obj')
    for i, (base, cut, _) in enumerate(work['candidates']):
        G.unpack(base).export(output/f'work_candidate_{i}.obj')
        G.unpack(cut).export(output/f'work_collision_{i}.obj')
    zero = dict(fixture_lift_m=[0., 0.], total_object_world_translation_m=[[0., 0., 0.]]*2)
    frame_arrays = {}
    for label, data in [('original', pair), ('lift_11_16mm', lift(pair, [.016, 0.])), ('lift_11_51mm', lift(pair, [.051, 0.])), ('xyz', moved)]:
        for row in data['rows']:
            for key in ['cloud', 'hull', 'fixture_transform', 'object_transform']:
                frame_arrays[f'{label}_{row["pose"]}_{key}'] = row[key]
    np.savez_compressed(output/'forward_hull_data.npz', **frame_arrays)
    certificate_a = json.loads((PROOF/'certificates/illegal/pose11+19/report.json').read_text())
    penetration = []
    for label, value in [('minimal', small), ('expanded', large)]:
        v = G.unpack(value).vertices
        penetration.append(dict(base=label, installed_minimum_z_mm={row['pose']:float(C.F.transform_points(v, row['fixture_transform'])[:, 2].min()*1000) for row in pair['rows']}))
    report = dict(original_pair_case=pair['case']['id'], work_case=four['case']['id'],
        original_ground_separator=certificate_a['ground_certificates'][0],
        one_pose_lift_intervals=[one_pose_interval(pair, i) for i in range(2)],
        both_pure_z=both_z_lp(pair), ground_ring_examples=penetration,
        xyz_geometric_seating=solution, xyz_original_demands_floor_checks=floor_values(moved),
        xyz_actual_base_checks=ground_checks,
        work_separator=work['proof']['separator'], work_radial_certificate=work['proof']['radial_certificate'],
        real_work_arrow=work['witness'], dense_work_arrows=len(work['arrows']),
        work_candidates=[row[2] for row in work['candidates']],
        validity_scope='Ground video: floor/coverage geometry only. Work video: fixed original layouts, unlimited base extent/topology.',
        pure_z_success_claim=False, xyz_force_replay=False, xyz_work_avoidance_checked=False,
        xyz_support_contact_capacity_claim=False,
        hull_update='Every original coupled wrench is translated forward: tau_new=tau_old+delta cross F; x=-tau_y/F_z, y=tau_x/F_z. No fitted or replacement demands.',
        transfer_scope='Object removed during fixture turns; installed-state penetration is the obstruction, not inability to lift an empty fixture during transfer.',
        saved_step4_and_objects_unchanged=True)
    C.save(output/'geometry_report.json', report)
    inputs_a.verify(); inputs_b.verify()
    return pair, four, small, large, solution, moved, base, work, report, [inputs_a, inputs_b]

