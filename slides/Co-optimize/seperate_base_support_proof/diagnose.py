"""Read-only necessary-condition diagnostics on fresh Step4 / Step5 outputs.

Ground proof is unbounded. Work proof initially uses the recorded Step5.2
box and deliberately permits disconnected, zero-thickness contacts. It omits
other floors, bodies and exits, and includes the object's own ground points.
Consequently a rejection cannot be blamed on a rim construction heuristic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

PROOF = Path(__file__).resolve().parent
COOPT = PROOF.parent
ROOT = COOPT.parents[1]
for path in [ROOT, COOPT/'helper_func', COOPT/'vis_func', COOPT/'step5.1',
             COOPT/'step5.2', COOPT/'step5.3', COOPT/'step5.3/placement_base']:
    sys.path.insert(0, str(path))
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
import stage5_common as C
import growth
from base_reactions import load_tasks
from work_access import WorkAccess

BASE = PROOF/'output/B'


def geometry_vertices(shape):
    if shape.is_empty:
        return np.empty((0, 2))
    if shape.geom_type == 'Polygon':
        return np.asarray(shape.exterior.coords)[:-1]
    if shape.geom_type in ('LineString', 'Point'):
        return np.asarray(shape.coords)
    return np.vstack([geometry_vertices(part) for part in shape.geoms])


def planes_in_floor(data, accesses, task_index):
    inverse = np.linalg.inv(data['rows'][task_index]['fixture_transform'])
    result = []
    for row, access in zip(data['rows'], accesses):
        placement = np.linalg.inv(row['fixture_transform']) @ row['object_transform']
        local = np.linalg.inv(placement) @ inverse
        for face, count, family in zip(access.face_ids, access.counts, access.planes):
            original = family[:count]
            result.append(dict(pose=row['pose'], face=int(face),
                A=original[:, :3]@local[:3, :2],
                b=original[:, :3]@local[:3, 3]+original[:, 3]))
    return result


def free_contacts(planes, bounds):
    low, high = np.asarray(bounds)[:, :2]
    rectangle = np.array([[*low], [high[0],low[1]], [*high], [low[0],high[1]]])
    polygons = []
    for cone in planes:
        polygon = rectangle.copy()
        for a, b in zip(cone['A'], cone['b']):
            polygon = growth.clip_polygon(polygon, -a, -float(b))
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            shape = Polygon(polygon)
            if shape.area > 1e-18:
                polygons.append(shape)
    forbidden = unary_union(polygons) if polygons else Polygon()
    free = Polygon(rectangle).difference(forbidden)
    return free, forbidden


def intersect_intervals(first, second):
    return [(max(a,c),min(b,d)) for a,b in first for c,d in second
            if max(a,c) <= min(b,d)]


def cone_radial_intervals(cone, radius, margin=1e-10):
    """Angles whose entire ray r>=radius lies strictly inside this cone.

    A(r*d)+b<=-margin for all r>=radius follows from A*d<=0 and
    radius*A*d+b<=-margin. Intersect their exact circle intervals.
    """
    turn = 2*np.pi
    intervals = [(0.,turn)]
    for a,b in zip(cone['A'],cone['b']):
        norm = float(np.linalg.norm(a))
        if norm <= 1e-14:
            if b > -margin:
                return []
            continue
        limit = min(-1e-12, -(float(b)+margin)/radius)/norm
        if limit < -1.:
            return []
        if limit >= 1.:
            continue
        phi = float(np.arctan2(a[1],a[0]))
        half = float(np.arccos(np.clip(limit,-1.,1.)))
        start = (phi+half) % turn
        length = turn-2*half
        end = start+length
        allowed = [(start,min(end,turn))]
        if end > turn:
            allowed.append((0.,end-turn))
        intervals = intersect_intervals(intervals, allowed)
        if not intervals:
            break
    return intervals


def radial_certificate(planes, radius):
    arcs = []
    for index, cone in enumerate(planes):
        for start,end in cone_radial_intervals(cone,radius):
            arcs.append(dict(start_rad=start,end_rad=end,cone_index=index,
                             source_pose=cone['pose'], source_face=cone['face']))
    # An explicit greedy finite subcover of the complete direction circle.
    selected, frontier = [], 0.
    remaining = sorted(arcs,key=lambda arc:(arc['start_rad'],-arc['end_rad'],arc['cone_index']))
    while frontier < 2*np.pi-1e-12:
        usable = [arc for arc in remaining if arc['start_rad']<=frontier+1e-13
                  and arc['end_rad']>frontier+1e-13]
        if not usable:
            return dict(passed=False,radius_m=radius,covered_until_rad=frontier)
        best = max(usable,key=lambda arc:(arc['end_rad'],-arc['cone_index']))
        selected.append(best)
        frontier = best['end_rad']
    return dict(passed=True,radius_m=radius,angular_subcover=selected,
                entire_exterior_of_circle_forbidden=True, original_cones_used=True,
                positive_plane_margin_m=1e-10, numeric_angle_tolerance_rad=1e-13,
                proof='for each direction a cone contains every r>=R; the arcs cover [0,2pi]')


def containment(hull, points):
    attainable = MultiPoint(points).convex_hull if len(points) else Polygon()
    distances = np.array([attainable.distance(MultiPoint([q])) if not attainable.is_empty else 1e10
                          for q in hull[:, :2]])
    return dict(passed=bool(np.max(distances)<=1e-9), maximum_gap_m=float(distances.max()),
                rejected_hull_vertices=np.flatnonzero(distances>1e-9).tolist(),
                attainable_convex_hull_xy_m=geometry_vertices(attainable).tolist())


def analyze(case):
    started = time.monotonic()
    selection = BASE/case['id']/'data/selection.json'
    inputs = C.Inputs('B', selection)
    inputs.base = BASE
    data = inputs.load(case)
    tasks, body = load_tasks(data, inputs)
    accesses = [WorkAccess(body, task.domain.work_ids,
                           task.domain.data['load']['cone_half_deg']) for task in tasks]
    ground = growth.floor_certificates(data['rows'])
    work = []
    out = PROOF/'diagnostics'/case['id']
    out.mkdir(parents=True, exist_ok=False)
    for i, row in enumerate(data['rows']):
        planes = planes_in_floor(data, accesses, i)
        free, forbidden = free_contacts(planes, data['bounds'])
        contacts = np.vstack([geometry_vertices(free), row['object_ground']])
        cover = containment(row['hull'], contacts)
        record = dict(pose=row['pose'], **cover, work_only=True, other_floors_checked=False,
                      body_exclusion_checked=False, exits_checked=False,
                      disjoint_zero_thickness_contacts_allowed=True,
                      own_object_ground_vertex_count=len(row['object_ground']),
                      bounds_m=data['bounds'].tolist(), box_independent=False,
                      work_forbidden_area_m2=float(forbidden.area),
                      free_contact_area_m2=float(free.area))
        (out/f'{row["pose"]}_free.wkt').write_text(free.wkt)
        (out/f'{row["pose"]}_work.wkt').write_text(forbidden.wkt)
        np.savez_compressed(out/f'{row["pose"]}_cones.npz',
                            **{f'A_{k}': cone['A'] for k,cone in enumerate(planes)},
                            **{f'b_{k}': cone['b'] for k,cone in enumerate(planes)})
        if not cover['passed']:
            # Increased windows are explicit diagnostics, never a global proof.
            record['expanded_windows'] = []
            for pad in [.02, .10, 1.]:
                extended = data['bounds'] + np.array([[-pad]*3,[pad]*3])
                more, _ = free_contacts(planes, extended)
                coverage = containment(row['hull'], np.vstack([geometry_vertices(more), row['object_ground']]))
                record['expanded_windows'].append(dict(padding_m=pad, **coverage))
            for radius in [.25,1.,10.]:
                certificate = radial_certificate(planes, radius)
                if certificate['passed']:
                    bounds = np.array([[-radius]*3,[radius]*3])
                    entire_free, entire_work = free_contacts(planes, bounds)
                    global_cover = containment(row['hull'], np.vstack([
                        geometry_vertices(entire_free), row['object_ground']]))
                    record['unbounded_work_coverage'] = dict(**global_cover,
                        radial_certificate=certificate, bounding_box_independent=True,
                        other_floors_bodies_exits_ignored=True,
                        disjoint_zero_thickness_contacts_allowed=True,
                        own_object_ground_included=True)
                    (out/f'{row["pose"]}_global_free.wkt').write_text(entire_free.wkt)
                    (out/f'{row["pose"]}_global_work.wkt').write_text(entire_work.wkt)
                    break
        work.append(record)
    inputs.verify()
    record = dict(id=case['id'], pose_count=case['pose_count'],
                  fresh_step4_passed=True, step4_experiment=case['experiment'],
                  ground_certificates=ground, ground_rejected=bool(ground),
                  work_only_box_rejected=any(not r['passed'] for r in work), work_per_pose=work,
                  seconds=time.monotonic()-started, inputs_unchanged=True)
    C.save(out/'report.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets', nargs='*')
    args = parser.parse_args()
    progress = json.loads((PROOF/'progress.json').read_text())
    cases = [r['case'] for r in progress['step5_results'] if 4<=r['case']['pose_count']<=6]
    if args.sets:
        cases = [case for case in cases if case['id'] in args.sets]
    for case in cases:
        out = PROOF/'diagnostics'/case['id']/'report.json'
        if out.exists():
            continue
        result = analyze(case)
        print('DIAG', result['id'], 'ground', result['ground_rejected'],
              'work_box', result['work_only_box_rejected'],
              'work_1m', [r['pose'] for r in result['work_per_pose']
                          if r.get('expanded_windows') and not r['expanded_windows'][-1]['passed']], flush=True)


if __name__ == '__main__':
    main()
