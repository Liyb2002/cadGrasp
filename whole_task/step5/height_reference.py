"""Choose one tool height with fixed Step4 seats and a fixed z=0 ground.

Lowering the tool and every installation preserves each object/fixture seat.
It changes the ground pressure demands, so old ground witnesses cannot be
reused. All cross-installation floor inequalities are affine in this shift.
Their bound is necessary; an actual connected witness is checked separately.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from whole_task.operations.state import transform
from whole_task.step1.common import sha256, write_json
from whole_task.step3 import independent_geometry as G
from whole_task.step3.membranes import MembraneModel
from whole_task.step4.membrane_search import read_group


def lower_layout(layout, distance):
    q = layout.copy()
    q.poses[:, 2, 3] -= distance
    q.material[:, :3, 3] += distance * q.material[:, :3, 2]
    return q


def lower_model(model, distance):
    workspace = deepcopy(model.workspace)
    workspace['reference_head_front_world_m'][2] -= distance
    return MembraneModel(model.source, model.outer, model.records, workspace,
                         model.physical, model.policy, model.name)


def model_for_report(model, report):
    return lower_model(model, report.get('height_reference', {}).get('lowering_m', 0.))


def floor_limit(layout, task_index, targets, com):
    """Exact upper bound from all paired endpoint pressures vs all floors."""
    F = targets[:, :3]
    xy = G.pressure_points(targets, com)
    M = layout.material[task_index]
    normals = layout.material[:, :3, 2]
    offsets = np.einsum('ij,ij->i', normals, layout.material[:, :3, 3])
    initial = transform(np.c_[xy, np.zeros(len(xy))], M)
    velocity = np.c_[F[:, :2] / F[:, 2, None], np.ones(len(F))] @ M[:3, :3].T
    slack = initial @ normals.T - offsets
    slope = velocity @ normals.T - 1.
    if slack.min() < -1e-8:
        raise ValueError('The starting pressure demands already cross an installation floor')
    bounds = np.full(slack.shape, np.inf)
    decreasing = slope < -1e-10
    bounds[decreasing] = np.maximum(0., slack[decreasing]) / -slope[decreasing]
    endpoint, other = np.unravel_index(bounds.argmin(), bounds.shape)
    return dict(max_lowering_m=float(bounds[endpoint, other]),
                endpoint_index=int(endpoint), other_index=int(other),
                original_slack_m=float(slack[endpoint, other]),
                slack_derivative=float(slope[endpoint, other]),
                original_pressure_xy_m=xy[endpoint].tolist()), slack, slope


def plan_height(model, groups, *, safety_m=1e-5):
    tasks = []
    pressure_bounds = []
    details = []
    for number, group in enumerate(groups, 1):
        q, core = read_group(group)
        part = model.subset(group['members'])
        prepared = part.prepare(q)
        for i, t in enumerate(prepared):
            T = np.linalg.inv(q.material[i])
            shell_z = float(transform(core.vertices, T)[:, 2].min())
            body_z = float(t['body_world'].vertices[:, 2].min())
            clearance = min(shell_z, body_z)
            if clearance < -1e-9:
                raise ValueError('The starting object or shell crosses the ground')
            tasks.append(dict(id=t['id'], set=number, shell_clearance_original_m=shell_z,
                              object_clearance_original_m=body_z,
                              natural_resting_head_height_m=float(model.head[2] - clearance)))
            bound, slack, slope = floor_limit(q, i,
                t['endpoints']['endpoint_need_wrench'], t['domain'].com)
            bound.update(task=t['id'], other_task=prepared[bound['other_index']]['id'], set=number)
            pressure_bounds.append(bound)
            details.append((t['id'], number, slack, slope, part.ids))
    geometric = min(min(r['shell_clearance_original_m'], r['object_clearance_original_m']) for r in tasks)
    limiting = min(pressure_bounds, key=lambda r: r['max_lowering_m'])
    maximum = min(geometric, limiting['max_lowering_m'])
    # A finite foot must still have nonzero area at the limiting plane.
    chosen = max(0., maximum - safety_m)
    worst = None
    for task, number, slack, slope, ids in details:
        at_rest = slack + geometric * slope
        endpoint, other = np.unravel_index(at_rest.argmin(), at_rest.shape)
        if worst is None or at_rest[endpoint, other] < worst['gap_m']:
            worst = dict(task=task, other_task=ids[other], set=number,
                         gap_m=float(at_rest[endpoint, other]), endpoint_index=int(endpoint))
    for row in tasks:
        row['shell_to_ground_at_geometric_reference_m'] = row['shell_clearance_original_m'] - geometric
        row['shell_to_ground_at_selected_reference_m'] = row['shell_clearance_original_m'] - chosen
    return dict(method='one_fixed_lowered_tool_all_installations', ground_z_m=0.,
                original_head_world_m=model.head.tolist(),
                head_world_m=(model.head - [0., 0., chosen]).tolist(),
                geometric_max_lowering_m=float(geometric),
                geometric_head_height_m=float(model.head[2] - geometric),
                pressure_max_lowering_m=limiting['max_lowering_m'],
                maximum_lowering_necessary_conditions_m=float(maximum),
                lowering_m=float(chosen), finite_foot_safety_m=safety_m,
                limiting_pressure_pair=limiting,
                geometric_reference_worst_pressure_floor_gap=worst,
                geometric_reference_passes_pressure_floors=bool(worst['gap_m'] >= -1e-8),
                tasks=tasks, pressure_bounds=pressure_bounds,
                step4_material_geometry_and_object_seats_changed=False,
                connected_witness_required=True, connected_witness_certified=False)


def save_plan(folder, plan):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'height_reference.json', plan)


def certify_group(model, group, out, lowering):
    """Build a fresh physical witness; never shift or trust the old witness."""
    import trimesh
    from whole_task.step4.ground_oracle import GroundOracle
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    saved = out / 'certificate.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        if result['core_sha256'] != group['sha256'] or result['lowering_m'] != lowering:
            raise ValueError('Height witness belongs to another reference or Step4 core')
        for item in (result['witness'], result['audit']):
            if sha256(item['file']) != item['sha256']:
                raise ValueError('Height witness or audit changed')
        return result
    original, core = read_group(group)
    q = lower_layout(original, lowering)
    part = model.subset(group['members'])
    np.testing.assert_allclose(q.material @ q.poses, original.material @ original.poses,
                               atol=1e-14, rtol=0.)
    candidate = dict(layout=q, support=core, tasks=part.prepare(q), construction={})
    witness, audit = GroundOracle(part).query(candidate)
    path = out / 'witness.npz'
    np.savez_compressed(path, vertices=witness.vertices, faces=witness.faces)
    # Replay the serialized triangles, including their actual coplanar feet.
    from whole_task.step4.ground_oracle import ground_enclosure, pressure_rows
    with np.load(path) as z:
        disk = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    ground = ground_enclosure(disk, pressure_rows(part, q))
    if not all(r['passed'] for r in ground):
        raise ValueError('Serialized height witness loses ground enclosure')
    audit['serialized_ground'] = ground
    write_json(out / 'audit.json', audit)
    result = dict(core_sha256=group['sha256'], lowering_m=lowering,
                  witness=dict(file=str(path.resolve()), sha256=sha256(path)),
                  audit=dict(file=str((out / 'audit.json').resolve()), sha256=sha256(out / 'audit.json')))
    write_json(saved, result)
    print('HEIGHT WITNESS PASS', part.ids, 'lowering_mm', lowering * 1000, flush=True)
    return result


def certify_saved_group(source, group, out, lowering):
    from whole_task.step3.run_membranes import load_model
    model, _, _ = load_model(source)
    return certify_group(lower_model(model, lowering), group, out, lowering)


def prepare_height_reference(model, source, groups, folder, workers=2):
    """Back off only if a necessary bound lacks a constructive certificate."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    folder = Path(folder)
    plan = plan_height(model, groups)
    start = plan['lowering_m']
    attempts = []
    for attempt, distance in enumerate(dict.fromkeys((start, start / 2, start / 4, 0.))):
        refs = [None] * len(groups)
        failures = []
        jobs = [(i, g, folder / f'trial_{attempt:02}' / f'set_{i+1:02}') for i, g in enumerate(groups)]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                pending = {pool.submit(certify_saved_group, source, g, out, distance): i for i, g, out in jobs}
                for future in as_completed(pending):
                    try:
                        refs[pending[future]] = future.result()
                    except (ValueError, RuntimeError) as ex:
                        failures.append(str(ex))
        else:
            shifted = lower_model(model, distance)
            for i, g, out in jobs:
                try:
                    refs[i] = certify_group(shifted, g, out, distance)
                except (ValueError, RuntimeError) as ex:
                    failures.append(str(ex))
        attempts.append(dict(lowering_m=distance, failures=failures))
        if failures:
            print('HEIGHT WITNESS RETRY', distance, failures, flush=True)
            continue
        plan.update(lowering_m=distance, head_world_m=(model.head - [0., 0., distance]).tolist(),
                    connected_witness_certified=True, certificates=refs, witness_attempts=attempts)
        for row in plan['tasks']:
            row['shell_to_ground_at_selected_reference_m'] = row['shell_clearance_original_m'] - distance
        save_plan(folder, plan)
        return lower_model(model, distance), plan
    raise ValueError('No certified common tool height with fixed Step4 seats: ' + str(attempts))
