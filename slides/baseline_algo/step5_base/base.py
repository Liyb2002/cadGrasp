"""A demand-following floor ring, with only the withdrawal corridor removed."""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from shapely import constrained_delaunay_triangles, set_precision
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union

from step1.needs import OUTPUTS, OBJECTS, COORD, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step2_local_support import geometry as H, insertion as D
from step5_connect_support import whole_assembly as A, floor_design as FD
from step5_connect_support import belt_geometry as B, solids as S
from step6_connect_support import direction_first as X
from step5_base import bearing as M

STAGE = 'step5_base'
WIDTH_FRACTION = .05
HEIGHT_FRACTION = .04
GAP_FRACTION = .02
# A declared finite design family: offset in metres = fraction * object scale.
OFFSET_FRACTIONS = np.r_[np.arange(0, 201) * .001, .4, .8, 1.6, 3.2]
FRICTION_WITNESSES = (64., 256., 1024., 4096., 16384., 65536.)
MAX_REFINEMENTS = 2


def shadow(mesh, direction, height, gap, reach):
    """Conservative XY positions occupied by a thick base during its full ray.

    Project each obstacle face backwards through the expanded base Z slab.
    Horizontal rays use a length exceeding the entire candidate search region.
    Final acceptance additionally replays the actual 3D solids, independently.
    """
    d = np.asarray(direction)
    polygons = []
    for triangle in mesh.triangles:
        clipped = H.clip_plane(triangle, np.array([0., 0., -1., -gap]))
        if len(clipped) < 3:
            continue
        if d[2] <= 1e-10:
            clipped = H.clip_plane(clipped, np.array([0., 0., 1., -height-gap]))
            if len(clipped) < 3:
                continue
            xy = clipped[:, :2]
            cloud = np.vstack([xy, xy-reach*d[:2]])
        else:
            # Add intersections at the kink in max(0, (z-height-gap)/dz).
            low = H.clip_plane(clipped, np.array([0., 0., 1., -height-gap]))
            high = H.clip_plane(clipped, np.array([0., 0., -1., height+gap]))
            q = np.vstack([v for v in (low, high) if len(v)])
            first = np.maximum(0., (q[:, 2]-height-gap)/d[2])
            last = (q[:, 2]+gap)/d[2]
            cloud = np.vstack([q[:, :2]-first[:, None]*d[:2],
                               q[:, :2]-last[:, None]*d[:2]])
        polygon = MultiPoint(cloud).convex_hull
        if polygon.area > 0:
            polygons.append(polygon)
    # Covers horizontal cube padding used by the independent SweptScene check.
    precision=float(mesh.extents.max())*1e-10
    merged=set_precision(unary_union(polygons), precision)
    return merged.buffer(2*gap+2*precision, join_style=2)


def candidate_polygons(required, pivot, obstacle, scale):
    hull = Polygon(FD.boundary(np.vstack([required, COORD.floor(pivot)])))
    width = WIDTH_FRACTION*scale
    for fraction in OFFSET_FRACTIONS:
        inner = hull.buffer(float(fraction)*scale, join_style=2)
        outer = inner.buffer(width, join_style=2)
        material = outer.difference(inner).difference(obstacle)
        if material.geom_type != 'Polygon' or material.area <= scale**2*1e-12:
            continue
        footprint = unary_union([material, MultiPoint([COORD.floor(pivot)])]).convex_hull
        if not footprint.buffer(scale*1e-9).covers(hull):
            continue
        yield material, dict(kind='tight_demand_ring', offset_m=float(fraction)*scale,
            width_m=width, height_m=HEIGHT_FRACTION*scale, clearance_m=GAP_FRACTION*scale,
            footprint_area_m2=float(footprint.area), material_area_m2=float(material.area),
            inner_xy_m=np.asarray(inner.exterior.coords)[:-1].tolist(),
            outer_xy_m=np.asarray(outer.exterior.coords)[:-1].tolist(),
            construction='demand hull offset ring minus the object withdrawal shadow',
            actual_remaining_footprint_covers_demand=True)


def extrude(material, height):
    triangles = list(constrained_delaunay_triangles(material).geoms)
    polygons = [np.asarray(t.exterior.coords)[:3] for t in triangles]
    np.testing.assert_allclose(sum(Polygon(p).area for p in polygons), material.area,
                               rtol=1e-9, atol=1e-14)
    parts = []
    for p in polygons:
        bottom = COORD.lift_floor(p)
        parts.append(D.engine.hull_mesh(np.vstack([bottom, bottom+[0., 0., height]])))
    return parts, [p.tolist() for p in polygons]


def code_hashes():
    return {**D.code_hashes(), **I.hashes([Path(m.__file__) for m in (A, FD, B, S, X, H, M, M.Q)]),
            **I.hashes([Path(__file__)])}



def adaptive_search(count, evaluate, max_refinements=MAX_REFINEMENTS):
    """Find a certificate with exponential probes, then refine its area rank.

    Certificate failure is NOT a monotone infeasibility oracle: obstacle cuts
    and numerical certificates can be nonmonotone. This is a bounded search
    heuristic, never a proof that skipped candidates fail or a minimum proof.
    Every accepted state is kept while unsuccessful refinements are discarded.
    Stop after max_refinements further certificates: each can span a million
    loads, so a tiny area improvement must not trigger too many costly full-load checks.
    """
    if not isinstance(max_refinements, int) or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    if count == 0:
        return None, []
    probes = [0]
    rank = 1
    while rank < count-1:
        probes.append(rank)
        rank *= 2
    if count > 1:
        probes.append(count-1)
    tried, previous = [], -1
    for rank in probes:
        result = evaluate(rank)
        tried.append(rank)
        if result is None:
            previous = rank
            continue
        best, upper, lower = result, rank, previous+1
        refinements = 0
        while lower < upper and refinements < max_refinements:
            refinements += 1
            middle = (lower+upper)//2
            candidate = evaluate(middle)
            tried.append(middle)
            if candidate is None:
                lower = middle+1
            else:
                best, upper = candidate, middle
        return best, tried
    return None, tried


def equivalent_ground_groups(choices):
    """Group exactly identical extreme floor vertices; do not round geometry."""
    groups = {}
    for choice in choices:
        xy = M.floor_vertices(dict(pads_xy_m=[np.asarray(choice[5].exterior.coords)[:-1]]))
        key = xy.tobytes()
        groups.setdefault(key, []).append(choice)
    return list(groups.values())

def build(name):
    began = perf_counter()
    domain, contacts, schedule, points, directions, work, paths = A.read_inputs(name)
    if 'installation' in directions['direction_catalogue']:
        from step5_base import modular
        return modular.build(name)
    if work is not None:
        raise ValueError('Base design currently uses the surface-only access policy')
    out = OUTPUTS/name/pose_name()/STAGE
    out.mkdir(parents=True, exist_ok=True)
    for filename in ('geometry.npz', 'base.stl', 'base_mm.stl', 'trajectory.json', 'audit.json', 'bearing.npz'):
        (out/filename).unlink(missing_ok=True)
    I.save(out/'status.json', dict(complete=False, status='searching_demand_following_base'))
    floor = FD.prepare(points)
    count = len(directions['common_directions']['ids'])
    vectors, direction_info = X.scheduled_directions(directions, count)
    scale = float(domain.mesh.extents.max())
    gap = GAP_FRACTION*scale
    demand = floor['required_hull_xy_m']
    pivot = floor['original_pivot_m']
    largest = Polygon(FD.boundary(np.vstack([demand, COORD.floor(pivot)]))).buffer(
        (float(max(OFFSET_FRACTIONS))+WIDTH_FRACTION)*scale, join_style=2)
    bounds = np.asarray(largest.bounds).reshape(2, 2)
    # A horizontal shadow ray must span the enlarged search family too.
    reach = float(np.linalg.norm(np.maximum(bounds[1], domain.mesh.bounds[1, :2])-
                                 np.minimum(bounds[0], domain.mesh.bounds[0, :2])))+4*gap
    choices = []
    for rank, (direction, direction_id) in enumerate(zip(vectors, direction_info['chosen_direction_ids'])):
        obstacle = shadow(domain.mesh, direction, HEIGHT_FRACTION*scale, gap, reach)
        candidates = list(candidate_polygons(demand, pivot, obstacle, scale))
        for material, record in candidates:
            choices.append((record['footprint_area_m2'], record['material_area_m2'], rank,
                            direction_id, direction, material, record))
        print(name, pose_name(), 'base direction', rank+1, '/', count,
              'planar candidates', len(candidates), flush=True)
    choices.sort(key=lambda value: value[:3])
    report = dict(object=name, pose=pose_name(), stage=STAGE, complete=True, passed=False,
        status='adaptive_base_search_inconclusive', search_outcome='inconclusive', direction_search=direction_info,
        objective='certified shared bearing and insertion first; prefer the existing friction witness, then compact footprint and material area',
        offset_fractions=OFFSET_FRACTIONS.tolist(), candidate_count=len(choices),
        minimum_scope='adaptive probes of all-direction offset rings; skipped candidates and arbitrary shapes remain unresolved',
        global_minimum_claimed=False, optimal_within_enumerated_family=False,
        required_hull_xy_m=demand.tolist(), original_pivot_m=np.asarray(pivot).tolist(),
        area_lower_bound_m2=float(Polygon(floor['support_polygon_xy_m']).area),
        bearing_verified=False, assembly_verified=False, rejected_solid_candidates=[],
        rejected_bearing_candidates=[], bearing_is_selection_constraint=True,
        bearing_scope='fixed Step3 heads and candidate ground contacts, treated as one massless rigid support',
        friction_witness_search_order=list(FRICTION_WITNESSES),
        friction_search_scope='adaptive compact-to-large probes at each unchanged friction witness, then local size refinement; no exhaustive or monotonic feasibility claim',
        search_method='exponential area-rank probes then safeguarded refinement',
        search_is_exhaustive=False, max_refinements=MAX_REFINEMENTS, search_trials=[], timings_s={})
    groups = equivalent_ground_groups(choices)
    report['unique_ground_hull_count'] = len(groups)
    report['bearing_evaluation_limit'] = (len(FRICTION_WITNESSES)*
        (int(np.ceil(np.log2(max(2, len(groups)))))+2)+MAX_REFINEMENTS) if groups else 0
    report['timings_s']['planar_candidates'] = perf_counter()-began
    witnesses = set()
    mechanics_started = perf_counter()

    def evaluate(rank, friction):
        group = groups[rank]
        first = group[0]
        # No extrusion is needed to test equilibrium: only the exact extreme
        # vertices of this actual material enter the equivalent floor cone.
        virtual_base = dict(first[6], pads_xy_m=[np.asarray(first[5].exterior.coords)[:-1]])
        mechanics, forces = M.bearing(domain, contacts, points, virtual_base,
                                     friction_values=(friction,), witnesses=witnesses)
        if not mechanics['continuous_passed']:
            report['rejected_bearing_candidates'].append(dict(direction_id=int(first[3]),
                offset_m=first[6]['offset_m'], footprint_area_m2=first[0],
                equivalent_candidate_count=len(group), friction=friction, bearing=mechanics,
                reason='shared_bearing_not_certified_not_selected'))
            print(name, pose_name(), 'reject base bearing', first[3],
                  'offset', first[6]['offset_m'], 'friction', friction,
                  'known difficult loads', len(witnesses), flush=True)
            return None
        for _, _, _, direction_id, direction, material, raw_base in group:
            base = dict(raw_base)
            parts, polygons = extrude(material, base['height_m'])
            base['pads_xy_m'] = polygons
            # An exact-array replay guards the search/triangulation reduction.
            p, n, owners = M.bearing_rays(domain, contacts, pivot, friction)
            matrix, _ = M.grounded_matrix(p, n, owners, domain.com, forces['scale'], [base], friction)
            np.testing.assert_array_equal(matrix, forces['equilibrium_matrix'])
            scene = X.SweptScene(domain.mesh, direction, gap)
            if not all(scene.clear(p, ground=True, clearance=gap) for p in parts):
                report['rejected_solid_candidates'].append(dict(direction_id=direction_id,
                    offset_m=base['offset_m'], reason='continuous_3d_sweep_or_clearance'))
                continue
            joined, solid = B.union_parts(parts, scale)
            ground, triangles = A.footprint(parts, pivot, demand, scale)
            trajectory = X.straight_path(B.Scene(domain.mesh), parts, domain.com, direction)
            if not solid['one_solid'] or not ground['passed'] or trajectory is None:
                report['rejected_solid_candidates'].append(dict(direction_id=direction_id,
                    offset_m=base['offset_m'], reason='solid_footprint_or_trajectory'))
                continue
            return dict(base=base, direction_id=direction_id, direction=direction,
                        parts=parts, joined=joined, solid=solid, ground=ground,
                        triangles=triangles, trajectory=trajectory, mechanics=mechanics, forces=forces)
        return None

    selected = None
    for friction in FRICTION_WITNESSES:
        selected, ranks = adaptive_search(len(groups), lambda rank: evaluate(rank, friction),
                                          max_refinements=MAX_REFINEMENTS)
        report['search_trials'].append(dict(friction=friction, ground_hull_ranks=ranks))
        if selected is not None:
            break
    report['bearing_evaluation_count'] = sum(len(row['ground_hull_ranks']) for row in report['search_trials'])
    assert report['bearing_evaluation_count'] <= report['bearing_evaluation_limit']
    report['timings_s']['bearing_and_geometry_search'] = perf_counter()-mechanics_started
    if selected is not None:
        base = selected['base']; direction = selected['direction']
        report.update(passed=True, status='base_bearing_and_insertion_verified', search_outcome='certified', base=base,
            direction_id=selected['direction_id'], withdrawal_direction=direction.tolist(),
            insertion_direction=(-direction).tolist(), solid=selected['solid'], ground=selected['ground'],
            trajectory=selected['trajectory'], optimal_within_enumerated_family=False,
            selection_scope='best certified candidate encountered during adaptive refinement in the first successful friction tier; skipped and unresolved candidates are not proved impossible',
            bearing_verified=True, bearing=selected['mechanics'],
            area_above_lower_bound_m2=base['footprint_area_m2']-report['area_lower_bound_m2'])
        labels = [f'ground_strip_{i:04d}' for i in range(len(selected['parts']))]
        np.savez_compressed(out/'geometry.npz', **S.pack_parts(selected['parts'], labels, selected['joined']),
                            floor_triangles_m=selected['triangles'])
        np.savez_compressed(out/'bearing.npz', **selected['forces'])
        selected['joined'].export(out/'base.stl', file_type='stl_ascii')
        mm = selected['joined'].copy(); mm.apply_scale(1000); mm.export(out/'base_mm.stl', file_type='stl_ascii')
        I.save(out/'trajectory.json', selected['trajectory'])
    report['timings_s']['total_build'] = perf_counter()-began
    report['search_objectives'] = [dict(direction_id=int(row[3]), offset_m=row[6]['offset_m'],
        footprint_area_m2=row[0], material_area_m2=row[1]) for row in choices]
    report['provenance'] = dict(inputs=I.hashes(paths), code=code_hashes())
    report['artifacts'] = {p: sha256(out/p) for p in ('geometry.npz', 'base.stl', 'base_mm.stl', 'trajectory.json', 'bearing.npz') if (out/p).exists()}
    I.save(out/'base.json', report)
    I.save(out/'status.json', dict(complete=True, status=report['status'], base_sha256=sha256(out/'base.json')))
    print(name, pose_name(), 'Step5:', report['status'], flush=True)
    return report


def read(name):
    out = OUTPUTS/name/pose_name()/STAGE
    report = I.check_report(out/'base.json')
    state = json.loads((out/'status.json').read_text())
    assert state['complete'] and state['base_sha256'] == sha256(out/'base.json')
    return report, S.unpack_parts(I.load_npz(out/'geometry.npz')) if report['passed'] else []


def audit(name):
    domain, contacts, _, points, directions, _, _ = A.read_inputs(name)
    report, parts = read(name)
    if report.get('schema') == 'stationary_base_removable_module_v1':
        from step5_base import modular
        return modular.audit(name)
    out = OUTPUTS/name/pose_name()/STAGE
    checks = {}
    if report['passed']:
        direction = np.asarray(report['withdrawal_direction'])
        assert report['direction_id'] in directions['common_directions']['ids']
        np.testing.assert_array_equal(direction, directions['direction_catalogue']['vectors'][report['direction_id']])
        scene = X.SweptScene(domain.mesh, direction, report['base']['clearance_m'])
        assert all(scene.clear(p, ground=True) for p in parts)
        ground, _ = A.footprint(parts, points['original_pivot_m'], FD.prepare(points)['required_hull_xy_m'], scene.scale)
        assert ground['passed']
        joined, solid = B.union_parts(parts, scene.scale)
        assert solid['one_solid']
        from step5_connect_support import rigid_path as P
        assert P.replay(B.Scene(domain.mesh), parts, report['trajectory'])
        rebuilt = [Polygon(p) for p in report['base']['pads_xy_m']]
        np.testing.assert_allclose(unary_union(rebuilt).area, report['base']['material_area_m2'], rtol=1e-9)
        import trimesh
        np.testing.assert_array_equal(trimesh.load(out/'base.stl', process=False).triangles, joined.triangles)
        checks = dict(step3_direction=True, actual_footprint=True, continuous_sweep=True,
                      clearance=True, one_solid=True, exported_mesh=True)
        assert report['bearing_verified'] and report['bearing']['continuous_passed']
        arrays=I.load_npz(out/'bearing.npz')
        mu=report['bearing']['sufficient_friction_coefficient']
        p,n,owners=M.bearing_rays(domain,contacts,points['original_pivot_m'],mu)
        matrix,ground=M.grounded_matrix(p,n,owners,domain.com,arrays['scale'],[report['base']],mu)
        np.testing.assert_array_equal(matrix,arrays['equilibrium_matrix'])
        for key,value in [('contact_points_m',p),('contact_normals',n),('contact_owners',owners),
                          ('ground_points_m',ground['points_m']),('ground_forces',ground['forces']),
                          ('ground_owners',ground['owners'])]:
            np.testing.assert_array_equal(arrays[key],value)
        from step4_floor_contact.audit import replay
        for prefix,loads in [('sample',points['load_wrenches']),('continuous',points['continuous_outer_load_wrenches'])]:
            checks[prefix+'_bearing']=replay(arrays,prefix,loads,1,mu)
        targets=M.Q.padded_targets(points['continuous_outer_load_wrenches'],arrays['scale'],12)
        for k,ids in enumerate(arrays['continuous_basis_indices']):
            rows=np.flatnonzero(arrays['continuous_assignment']==k)
            if len(rows):
                assert M.Q.membership(matrix,ids,targets[rows],certified=True)[0].all()
    result = dict(complete=True, passed=True, design_passed=report['passed'], checks=checks,
                  provenance=dict(inputs=I.hashes([out/'base.json']), code=code_hashes()))
    I.save(out/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    for name in args.objects or OBJECTS:
        if not args.audit_only:
            build(name)
        audit(name)
