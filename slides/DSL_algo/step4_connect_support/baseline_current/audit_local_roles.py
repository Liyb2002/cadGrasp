"""Ablate local head/floor reaction ports without changing the fixture solid.

Usage: python audit_local_roles.py RESULT_DIRECTORY

The directory contains geometry.npz, report.json and body0.obj ... bodyN.obj.
Bodies follow contacts in report['poses'] order, then their saved contact order.
Optional bridges.obj protects contacts supplied by additional connecting material.
This diagnoses boundary-contact roles in the existing rigid-body model. It does
not certify an internal force path, material sharing, stiffness or strength.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step0_pose_selection import equilibrium as Q
from step5_base.bearing import bearing_rays, seed_probes


def floor_contact(mesh, basis, offset, tolerance):
    """Actual boundary faces, edges and isolated vertices on the task floor.

    A convex hull is used only later to reduce equal-friction reaction rays;
    it is never used here to attribute physical contact material to a body.
    """
    world = (np.asarray(mesh.vertices)-offset)@basis.T
    if len(world) and world[:, 2].min() < -tolerance:
        raise ValueError('A supplied component extends below the task floor')
    on_floor = np.abs(world[:, 2]) <= tolerance
    faces = np.asarray(mesh.faces)
    floor_faces = faces[np.all(on_floor[faces], axis=1)]
    polygons = []
    for face in floor_faces:
        polygon = Polygon(world[face, :2])
        if polygon.area > 0:
            polygons.append(polygon)
    edges = np.asarray(mesh.edges_unique)
    floor_edges = edges[np.all(on_floor[edges], axis=1)]
    parts = polygons+[LineString(world[edge, :2]) for edge in floor_edges]
    if np.any(on_floor):
        parts.append(MultiPoint(world[on_floor, :2]))
    contact = unary_union(parts) if parts else GeometryCollection()
    return contact, dict(boundary_triangles=len(floor_faces),
                        boundary_edges=len(floor_edges),
                        boundary_vertices=int(on_floor.sum()),
                        area_m2=float(contact.area))


def contact_vertices(geometry):
    """All geometry vertices, including line/point contacts and hole boundaries."""
    if geometry.is_empty:
        return np.empty((0, 2))
    if geometry.geom_type == 'Polygon':
        pieces = [np.asarray(geometry.exterior.coords)[:, :2]]
        pieces += [np.asarray(ring.coords)[:, :2] for ring in geometry.interiors]
    elif hasattr(geometry, 'geoms'):
        pieces = [contact_vertices(part) for part in geometry.geoms]
    else:
        pieces = [np.asarray(geometry.coords)[:, :2]]
    return np.unique(np.concatenate(pieces), axis=0)


def reaction_vertices(geometry):
    vertices = contact_vertices(geometry)
    if len(vertices) >= 3 and np.linalg.matrix_rank(vertices-vertices[0]) == 2:
        vertices = vertices[ConvexHull(vertices).vertices]
    return vertices


def ablated_floor(actual, own, other_contacts, tolerance):
    """Remove only this body's contact region, keeping shared/ambiguous ports.

    The retained set is always a subset of the final solid's actual contact.
    Buffering OTHER bodies protects near-coincident ownership boundaries. The
    same tolerance matches final material to OWN contact: Boolean triangulation
    roundoff must not leave zero-area or microscopic 'unassigned' strips carrying
    unbounded floor reaction. Unassigned material farther than that tolerance
    remains. This is an explicit numerical component-attribution convention.
    Geometric differences may introduce newly extreme retained vertices, which
    must be included instead of simply deleting old convex-hull vertices.
    """
    exact_protected = unary_union(other_contacts) if other_contacts else GeometryCollection()
    protected = exact_protected.buffer(tolerance) if not exact_protected.is_empty else exact_protected
    raw_retained = actual.difference(own).union(actual.intersection(protected))
    owner_match = own.buffer(tolerance) if not own.is_empty else own
    retained = actual.difference(owner_match).union(actual.intersection(protected))
    raw_owner_boundary_extremes = []
    for vertex in reaction_vertices(raw_retained):
        point = Point(vertex)
        own_distance = float(own.distance(point)) if not own.is_empty else None
        protected_distance = float(exact_protected.distance(point)) if not exact_protected.is_empty else None
        if own_distance is not None and own_distance <= tolerance and (
                protected_distance is None or protected_distance > tolerance):
            raw_owner_boundary_extremes.append(dict(xy_m=vertex.tolist(), own_contact_distance_m=own_distance,
                other_or_bridge_contact_distance_m=protected_distance))
    # GEOS can reject an intersection of tiny, collinear line-only contacts
    # even though their two-dimensional overlap area is identically zero.
    overlap_area = (float(actual.intersection(own).intersection(protected).area)
                    if own.area > 0 and protected.area > 0 else 0.)
    return retained, dict(removed_area_m2=float(actual.difference(retained).area),
                         retained_area_m2=float(retained.area),
                         protected_overlap_area_m2=overlap_area,
                         ownership_matching_tolerance_m=tolerance,
                         raw_difference_reaction_vertices=len(reaction_vertices(raw_retained)),
                         raw_unprotected_owner_boundary_extreme_vertices=raw_owner_boundary_extremes,
                         raw_difference_is_not_used_for_equilibrium=True)


def matrix_for(task, contacts, floor, friction):
    points, normals, owners = bearing_rays(task.domain, contacts, task.floor, friction)
    # Q accepts even a line, a single point, or no floor contacts. The bearing
    # wrapper assumes a full-dimensional polygon and is inappropriate here.
    matrix, _ = Q.grounded_matrix(points, normals, owners, task.domain.com,
        task.scale, [dict(pads_xy_m=[reaction_vertices(floor)])], friction)
    return matrix


def check_samples(matrix, task, probes=48, chunk_size=4096):
    """Original-sample probes may find a failure; only all samples establish pass."""
    if len(task.targets) != 32768:
        raise ValueError('Expected exactly the original 32,768 sampled loads')
    targets = Q.padded_targets(task.targets/task.scale, task.scale, 12)
    solver = Q.BatchSolver(matrix)
    probe_ids = np.asarray(seed_probes(targets, probes), dtype=int)
    remaining = np.setdiff1d(np.arange(len(targets)), probe_ids)
    batches = [probe_ids]+[remaining[i:i+chunk_size] for i in range(0, len(remaining), chunk_size)]
    checked = 0
    for ids in batches:
        if not len(ids):
            continue
        result = solver.solve(targets[ids])
        checked += int(np.count_nonzero(result['assignment'] >= 0))
        if not result['passed']:
            diagnostic = result['diagnostics'][0] if result['diagnostics'] else dict(
                index=int(np.flatnonzero(result['assignment'] < 0)[0]), status='solver_unresolved')
            index = int(ids[diagnostic['index']])
            infeasible = diagnostic['status'] == 'infeasible_numeric'
            return dict(status='sample_infeasible_numeric' if infeasible else 'unresolved',
                all_original_samples_passed=False, checked_feasible_sample_count=checked,
                original_sample_count=len(targets), lp_count=solver.lp_count,
                witness=dict(original_sample_index=index,
                    conditioned_target=targets[index].tolist(),
                    demand_wrench= (task.targets[index]/task.scale).tolist(),
                    diagnostic=diagnostic),
                interpretation='Both existing simplex and IPM report infeasible; numerical witness, not an exact-arithmetic proof'
                    if infeasible else 'No infeasibility or all-sample completion claim')
    return dict(status='all_original_samples_passed', all_original_samples_passed=True,
        original_sample_count=len(targets), checked_feasible_sample_count=checked,
        lp_count=solver.lp_count)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, output=None, tolerance=1e-9, probes=48, chunk_size=4096):
    directory = Path(directory).resolve()
    report_path = directory/'report.json'
    report = json.loads(report_path.read_text())
    poses = report['poses']
    if len(poses) != 2:
        raise ValueError('This role audit requires exactly two poses')
    source = Path(report['source_schedule'])
    source = source if source.is_absolute() else ROOT/source
    contact_paths = [source.parent/f'contacts_{pose}.npz' for pose in poses]
    contacts = [I.read_contacts(path) for path in contact_paths]
    tasks = [read_task(report['object'], pose, poses) for pose in poses]
    if any(len(task.targets) != 32768 for task in tasks):
        raise ValueError('Refusing changed or augmented sample sets')
    geometry_path = directory/'geometry.npz'
    with np.load(geometry_path) as data:
        mesh = trimesh.Trimesh(data['vertices_m'], data['faces'], process=False)
        bases, offsets = data['rotations'].copy(), data['local_offsets_m'].copy()
    if bases.shape != (2, 3, 3) or offsets.shape != (2, 3):
        raise ValueError('Invalid saved task transforms')
    if not np.allclose(bases@bases.transpose(0, 2, 1), np.eye(3), atol=1e-10):
        raise ValueError('Saved rotations are not orthonormal')
    body_paths = [directory/f'body{i}.obj' for i in range(sum(map(len, contacts)))]
    body_meshes = [trimesh.load(path, force='mesh', process=False) for path in body_paths]
    bridge_path = directory/'bridges.obj'
    bridge = trimesh.load(bridge_path, force='mesh', process=False) if bridge_path.exists() else None
    inputs = [report_path, geometry_path, source, *contact_paths, *body_paths]
    if bridge is not None:
        inputs.append(bridge_path)
    inputs += [Path(path) for task in tasks for path in task.inputs]
    actual, actual_records, body_feet, bridge_feet = [], [], [], []
    for k in range(2):
        value, record = floor_contact(mesh, bases[k], offsets[k], tolerance)
        actual.append(value); actual_records.append(record)
        body_feet.append([floor_contact(part, bases[k], offsets[k], tolerance)[0] for part in body_meshes])
        bridge_feet.append(floor_contact(bridge, bases[k], offsets[k], tolerance)[0]
                           if bridge is not None else GeometryCollection())
    friction = []
    for pose in poses:
        matches = [check['friction_coefficient'] for check in report.get('checks', [])
                   if check.get('pose') == pose and 'friction_coefficient' in check]
        if len(matches) != 1:
            raise ValueError(f'Missing unambiguous saved friction coefficient for {pose}')
        friction.append(float(matches[0]))
    baseline = []
    for k, task in enumerate(tasks):
        print(f'Baseline {task.pose}: checking all original samples', flush=True)
        result = check_samples(matrix_for(task, contacts[k], actual[k], friction[k]), task, probes, chunk_size)
        baseline.append(dict(pose=task.pose, floor=actual_records[k], **result))
    roles = []
    if all(item['all_original_samples_passed'] for item in baseline):
        body_index = 0
        for k, group in enumerate(contacts):
            for local_index, contact in enumerate(group):
                other = 1-k
                print(f'Body {body_index}: head in {poses[k]}, floor in {poses[other]}', flush=True)
                head_matrix = matrix_for(tasks[k], group[:local_index]+group[local_index+1:], actual[k], friction[k])
                head = check_samples(head_matrix, tasks[k], probes, chunk_size)
                protected = [foot for i, foot in enumerate(body_feet[other]) if i != body_index]
                protected.append(bridge_feet[other])
                retained, details = ablated_floor(actual[other], body_feet[other][body_index], protected, tolerance)
                floor = check_samples(matrix_for(tasks[other], contacts[other], retained, friction[other]),
                                      tasks[other], probes, chunk_size)
                roles.append(dict(body=body_index, file=body_paths[body_index].name,
                    head_pose=poses[k], contact_index=local_index, candidate_id=contact['candidate_id'],
                    opposite_floor_pose=poses[other],
                    head_port_ablation=head,
                    opposite_floor_port_ablation=dict(**floor, geometry=details,
                        original_reaction_vertices=len(reaction_vertices(actual[other])),
                        retained_reaction_vertices=len(reaction_vertices(retained))),
                    both_ports_have_numerical_necessity_witnesses=(head['status'] == floor['status'] == 'sample_infeasible_numeric')))
                body_index += 1
    result = dict(schema='local_boundary_role_ablation_v2', object=report['object'], poses=poses,
        complete=all(item['all_original_samples_passed'] for item in baseline) and len(roles) == len(body_paths)
            and all(role[key]['status'] != 'unresolved' for role in roles
                    for key in ('head_port_ablation', 'opposite_floor_port_ablation')),
        scope='Reaction-port ablation only; unchanged geometry and original fixed sampled demands',
        fixture_geometry_modified=False, structural_strength_verified=False,
        internal_force_path_verified=False, material_reuse_certified=False,
        floor_coordinate_tolerance_m=tolerance,
        floor_ownership='Final boundary faces/edges/vertices matched to own body within the declared coordinate tolerance; other bodies and bridges protected at the same tolerance; unassigned material farther from own body retained',
        numerical_attribution_convention='Remove final contacts within tolerance of own contact unless another body or bridge provides contact within tolerance. This suppresses Boolean-roundoff boundary remnants; raw unaligned difference passes are not evidence of port redundancy.',
        bridges_protected=bridge is not None,
        attribution_limit='Component attribution is relative to supplied bodies and optional bridges; no unspecified overlapping component provenance is inferred',
        interpretation='Failure gives a numerically necessary boundary port for one original sample; passing ablation permits redundant roles and does not disprove local reuse',
        baseline=baseline, roles=roles,
        provenance=dict(inputs={str(path): sha256(path) for path in inputs}, code={str(Path(__file__).resolve()): sha256(__file__)}))
    output = Path(output) if output else directory/'local_role_audit.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--floor-tolerance', type=float, default=1e-9)
    parser.add_argument('--probe-count', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()
    if args.floor_tolerance <= 0 or args.probe_count < 1 or args.chunk_size < 1:
        parser.error('Tolerance, probe count and chunk size must be positive')
    audit(args.directory, args.output, args.floor_tolerance, args.probe_count, args.chunk_size)
