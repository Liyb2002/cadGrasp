"""Bounded positive-area geometric witnesses between local head/floor faces.

For each body, try at most six saved head triangles and eight actual opposite
floor triangles per head. A successful witness is the convex triangular loft
contained in BOTH this body and the final solid, with both end triangles on
their actual boundaries. This is geometric evidence, not an internal stress,
load-path, stiffness, strength or whole-contact-area certificate.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import QhullError
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT
from step2_local_support.geometry import hull_mesh
from step3_scheculer import contacts as I
from step4_connect_support.ground import solid64
from step4_connect_support.audit_local_roles import sha256


def triangle_area(triangle):
    return float(np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))/2)


def subtriangle(triangle, factor=.5):
    center = triangle.mean(axis=0)
    return center+factor*(triangle-center)


def boundary_check(mesh, triangle, tolerance):
    """Check the WHOLE end triangle against coplanar actual boundary triangles.

    Plane matching uses the declared distance tolerance. Coverage is a planar
    polygon difference, not three vertices or a centroid-distance probe.
    """
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    area = float(np.linalg.norm(normal)/2)
    if area <= 0:
        return dict(passed=False, area_m2=area, reason='zero_area_end')
    normal /= 2*area
    u = triangle[1]-triangle[0]
    u /= np.linalg.norm(u)
    axes = np.array([u, np.cross(normal, u)])
    query = Polygon((triangle-triangle[0])@axes.T)
    triangles = np.asarray(mesh.triangles)
    close = np.all(np.abs((triangles-triangle[0])@normal) <= tolerance, axis=1)
    close &= np.all(triangles.max(axis=1) >= triangle.min(axis=0)-tolerance, axis=1)
    close &= np.all(triangles.min(axis=1) <= triangle.max(axis=0)+tolerance, axis=1)
    polygons = [Polygon((face-triangle[0])@axes.T) for face in triangles[close]]
    polygons = [polygon for polygon in polygons if polygon.area > 0]
    missing = float(query.difference(unary_union(polygons)).area) if polygons else area
    allowed = min(1e-14, area*1e-8)
    return dict(passed=bool(missing <= allowed), area_m2=area,
        missing_boundary_area_m2=missing, allowed_missing_area_m2=allowed,
        matched_coplanar_boundary_triangles=len(polygons), plane_tolerance_m=tolerance)


def floor_candidates(mesh, basis, offset, tolerance):
    triangles = np.asarray(mesh.triangles)
    world = (triangles-offset)@basis.T
    ids = np.flatnonzero(np.all(np.abs(world[:, :, 2]) <= tolerance, axis=1))
    areas = np.asarray([triangle_area(triangles[index]) for index in ids])
    if not len(areas):
        return np.empty(0, dtype=int)
    # Ignore microscopic Boolean slivers when selecting a small witness menu.
    return ids[areas > max(1e-16, float(areas.max())*1e-4)]


def match_vertices(head, foot):
    order = min(itertools.permutations(range(3)),
                key=lambda indices: float(np.sum((head-foot[list(indices)])**2)))
    return foot[list(order)]


def find_witness(final_mesh, body_mesh, head_triangles, floor_basis, floor_offset,
                 tolerance=1e-9, head_limit=6, floor_limit=8, subtriangle_factor=.5):
    origin = np.asarray(final_mesh.bounds).mean(axis=0)
    scale = float(np.asarray(final_mesh.extents).max())
    final_solid = solid64(final_mesh, origin, scale)
    body_solid = solid64(body_mesh, origin, scale)
    head_areas = np.asarray([triangle_area(triangle) for triangle in head_triangles])
    head_ids = np.argsort(-head_areas, kind='stable')[:head_limit]
    floor_ids = floor_candidates(body_mesh, floor_basis, floor_offset, tolerance)
    floor_triangles = np.asarray(body_mesh.triangles)
    floor_areas = np.asarray([triangle_area(floor_triangles[index]) for index in floor_ids])
    trials, end_cache, best = [], {}, None
    for head_index in head_ids:
        if head_areas[head_index] <= 1e-16:
            continue
        head = subtriangle(head_triangles[head_index], subtriangle_factor)
        head_checks = dict(final=boundary_check(final_mesh, head, tolerance),
                           body=boundary_check(body_mesh, head, tolerance))
        if not all(check['passed'] for check in head_checks.values()):
            trials.append(dict(head_triangle_index=int(head_index), status='head_boundary_unverified', boundary=head_checks))
            continue
        if not len(floor_ids):
            break
        distance = np.linalg.norm(floor_triangles[floor_ids].mean(axis=1)-head.mean(axis=0), axis=1)
        # Half the menu favours nearby endpoints, half substantial actual faces.
        near_count = (floor_limit+1)//2
        chosen = list(floor_ids[np.argsort(distance, kind='stable')[:near_count]])
        for index in floor_ids[np.argsort(-floor_areas, kind='stable')]:
            if index not in chosen:
                chosen.append(int(index))
            if len(chosen) >= floor_limit:
                break
        for floor_index in chosen:
            foot = subtriangle(floor_triangles[floor_index], subtriangle_factor)
            if floor_index not in end_cache:
                end_cache[floor_index] = dict(final=boundary_check(final_mesh, foot, tolerance),
                                              body=boundary_check(body_mesh, foot, tolerance))
            end_checks = end_cache[floor_index]
            trial = dict(head_triangle_index=int(head_index), body_floor_face_index=int(floor_index))
            if not all(check['passed'] for check in end_checks.values()):
                trials.append(dict(**trial, status='floor_boundary_unverified', boundary=end_checks))
                continue
            foot = match_vertices(head, foot)
            try:
                wedge_mesh = hull_mesh(np.vstack([head, foot]))
            except QhullError:
                trials.append(dict(**trial, status='degenerate_loft'))
                continue
            wedge = solid64(wedge_mesh, origin, scale)
            volume = abs(float(wedge.volume()))*scale**3
            if volume <= 1e-15:
                trials.append(dict(**trial, status='insufficient_positive_volume', volume_m3=volume))
                continue
            missing_body_solid, missing_final_solid = wedge-body_solid, wedge-final_solid
            if missing_body_solid.status().name != 'NoError' or missing_final_solid.status().name != 'NoError':
                trials.append(dict(**trial, status='boolean_unresolved'))
                continue
            missing_body = abs(float(missing_body_solid.volume()))*scale**3
            missing_final = abs(float(missing_final_solid.volume()))*scale**3
            allowed = min(1e-14, volume*1e-7)
            passed = missing_body <= allowed and missing_final <= allowed
            measurement = dict(volume_m3=volume, missing_body_volume_m3=missing_body,
                missing_final_volume_m3=missing_final, allowed_missing_volume_m3=allowed,
                maximum_missing_volume_fraction=max(missing_body, missing_final)/volume)
            trials.append(dict(**trial, status='contained' if passed else 'leaves_material', **measurement))
            if best is None or measurement['maximum_missing_volume_fraction'] < best['maximum_missing_volume_fraction']:
                best = dict(**trial, **measurement)
            if passed:
                return dict(status='positive_area_direct_loft_verified', witness_found=True,
                    search=dict(head_limit=head_limit, floor_limit_per_head=floor_limit,
                        tried_pairs=sum('body_floor_face_index' in row for row in trials),
                        available_body_floor_faces=len(floor_ids)),
                    witness=dict(**trial, **measurement,
                        coordinate_frame='common fixture frame; metres',
                        head_triangle_m=head.tolist(), floor_triangle_m=foot.tolist(),
                        loft_vertices_m=np.asarray(wedge_mesh.vertices).tolist(),
                        loft_faces=np.asarray(wedge_mesh.faces).tolist(),
                        head_area_m2=triangle_area(head), floor_area_m2=triangle_area(foot),
                        centroid_distance_m=float(np.linalg.norm(head.mean(axis=0)-foot.mean(axis=0))),
                        corresponding_segment_lengths_m=np.linalg.norm(head-foot, axis=1).tolist(),
                        maximum_floor_plane_distance_m=float(np.abs(((foot-floor_offset)@floor_basis.T)[:, 2]).max()),
                        head_boundary=head_checks, floor_boundary=end_checks), trials=trials)
    return dict(status='no_witness_in_bounded_menu', witness_found=False,
        interpretation='Unverified in this small menu; not proof that no direct material connection exists',
        search=dict(head_limit=head_limit, floor_limit_per_head=floor_limit,
            tried_pairs=sum('body_floor_face_index' in row for row in trials),
            available_body_floor_faces=len(floor_ids)), best_failed_containment=best, trials=trials)


def audit(directory, output=None, tolerance=1e-9, head_limit=6, floor_limit=8):
    directory = Path(directory).resolve()
    report_path, geometry_path = directory/'report.json', directory/'geometry.npz'
    report = json.loads(report_path.read_text())
    poses = report['poses']
    if len(poses) != 2:
        raise ValueError('Exactly two task poses are required')
    source = Path(report['source_schedule'])
    source = source if source.is_absolute() else ROOT/source
    contact_paths = [source.parent/f'contacts_{pose}.npz' for pose in poses]
    contacts = [I.read_contacts(path) for path in contact_paths]
    with np.load(geometry_path) as data:
        final_mesh = trimesh.Trimesh(data['vertices_m'], data['faces'], process=False)
        bases, offsets = data['rotations'].copy(), data['local_offsets_m'].copy()
    inputs = [report_path, geometry_path, source, *contact_paths]
    bodies, index = [], 0
    for k, group in enumerate(contacts):
        for contact_index, contact in enumerate(group):
            body_path = directory/f'body{index}.obj'
            body = trimesh.load(body_path, force='mesh', process=False)
            inputs.append(body_path)
            heads = np.asarray(contact['triangles_m'])@bases[k]+offsets[k]
            print(f'Body {index}: bounded head-to-opposite-floor loft checks', flush=True)
            result = find_witness(final_mesh, body, heads, bases[1-k], offsets[1-k],
                                  tolerance, head_limit, floor_limit)
            bodies.append(dict(body=index, file=body_path.name, candidate_id=contact['candidate_id'],
                head_pose=poses[k], head_contact_index=contact_index,
                floor_pose=poses[1-k], **result))
            print(f'Body {index}: {result["status"]}', flush=True)
            index += 1
    result = dict(schema='local_positive_area_loft_witness_v1', object=report['object'], poses=poses,
        complete=True, witnessed_bodies=sum(body['witness_found'] for body in bodies),
        body_count=len(bodies), scope='Small geometric witnesses only; no new load cases or geometry changes',
        construction='Convex hull of two half-scale end subtriangles, including every straight interpolation between them',
        interpretation='A witness verifies positive-area local head/floor boundary connection within declared numerical tolerances; it does not certify whole-patch reuse or internal force flow',
        internal_force_path_verified=False, structural_strength_verified=False,
        floor_and_boundary_plane_tolerance_m=tolerance, bodies=bodies,
        provenance=dict(inputs={str(path): sha256(path) for path in inputs},
            code={str(Path(__file__).resolve()): sha256(__file__)}))
    output = Path(output) if output else directory/'local_wedge_audit.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plane-tolerance', type=float, default=1e-9)
    parser.add_argument('--head-limit', type=int, default=6)
    parser.add_argument('--floor-limit', type=int, default=8)
    args = parser.parse_args()
    if args.plane_tolerance <= 0 or not 1 <= args.head_limit <= 6 or not 1 <= args.floor_limit <= 10:
        parser.error('Positive tolerance and bounded menus (heads 1..6, floor 1..10) are required')
    audit(args.directory, args.output, args.plane_tolerance, args.head_limit, args.floor_limit)
