"""Connect all selected heads and a shared open base into one printed rigid body.

Step 3 remains the independent-head greedy baseline. Common-direction failure
is recorded here without deleting/reselecting any head or changing Step 3.
"""
from step1.needs import COORD
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh

from step1.needs import ContinuousNeeds, OUTPUTS, OBJECTS, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step2_local_support import insertion as D, work_volume as W
from step2_local_support import support_policy as POLICY
from step4_floor_contact import whole_assembly as F, equilibrium as Q
from step5_connect_support import ground as G, motion as M, solids as S, routing as T
from step5_connect_support import fixed_feet as H, surface_check as U
from step5_connect_support.fixed_feet import angles
from step5_connect_support.surface_check import surface_distances

STAGE = 'step5_connect_support'
SCHEMA = 'one_rigid_support_v1'
DEPTH_FACTORS = (1., .75, .5, .25, .125)
EXPANSIONS = (1., 1.35, 2.)


def common_directions(records):
    common = D.full()
    history = []
    for row in records:
        common = D.intersect(common, row['certified_directions'])
        history.append(dict(candidate_id=row['candidate_id'], intersection=common))
    return common, history


def open_base(mesh, required, angle, expansion=1.):
    """U opens along insertion +a; its back and arms clear the full object XY."""
    basis = G.frame(angle); scale = float(mesh.extents.max())
    obstacle = mesh.vertices@basis.T
    q = COORD.lift_floor(required)@basis.T
    combined = np.vstack([obstacle, q])
    margin = .025*scale*expansion
    back = float(combined[:, 0].min()-margin)
    front = float(max(q[:, 0].max()+margin, back+margin))
    low = float(combined[:, 1].min()-margin)
    high = float(combined[:, 1].max()+margin)
    width = .025*scale; height = .018*scale
    bounds = [([back-width, low-width, 0.], [back, high+width, height]),
              ([back-width, low-width, 0.], [front, low, height]),
              ([back-width, high, 0.], [front, high+width, height])]
    parts = [G.box(a, b, basis) for a, b in bounds]
    polygons = []
    for a, b in bounds:
        xy = np.array([[a[0], a[1], 0.], [b[0], a[1], 0.], [b[0], b[1], 0.], [a[0], b[1], 0.]])@basis
        polygons.append(COORD.floor(xy))
    return parts, dict(bearing_deg=float(angle), expansion=float(expansion),
        direction=basis[0].tolist(), width_m=width, height_m=height,
        pads_xy_m=[p.tolist() for p in polygons], back_local_x_m=back-width/2,
        back_local_y_bounds_m=[low, high], opening='insertion direction +a',
        construction='One continuous U base; footprint hull is not a filled plate')


def footprint(parts, pivot, required, scale):
    """Extract actual bottom triangles, including unexpected added ground material."""
    triangles = []
    for part in parts:
        on_floor = np.all(np.abs(part.triangles[:, :, 2]) <= scale*1e-10, axis=1)
        triangles.extend(part.triangles[on_floor])
    tri = np.asarray(triangles).reshape(-1, 3, 3)
    if not len(tri):
        return dict(passed=False, reason='no_actual_support_floor_faces'), tri
    points = np.vstack([COORD.floor(tri.reshape(-1, 3)), COORD.floor(pivot)])
    covered, hull = G.hull_coverage(required, points, scale*1e-9)
    maximum = float(np.max(required@hull.equations[:, :2].T+hull.equations[:, 2]))
    return dict(passed=bool(covered.all()), supplied_hull_xy_m=points[hull.vertices].tolist(),
        required_vertex_count=len(required), maximum_outside_distance_m=max(0., maximum),
        original_object_floor_point_included=True, floor_triangle_count=len(tri)), tri


def read_inputs(name):
    root = OUTPUTS/name/pose_name()
    schedule_path = root/'step3_scheculer/schedule.json'
    schedule = I.check_report(schedule_path)
    state = json.loads((schedule_path.parent/'status.json').read_text())
    assert state['complete'] and state['schedule_sha256'] == sha256(schedule_path)
    domain = ContinuousNeeds.read(root/'step_1_needs/needs.json')
    if not domain.mesh.is_watertight or not domain.mesh.is_winding_consistent:
        raise ValueError('One-body construction requires a closed outward-oriented object')
    path = schedule_path.parent/'final_contacts.npz'
    contacts = I.read_contacts(path)
    directions = I.check_report(schedule_path.parent/'insertion_directions.json')
    assert directions['selected_ids'] == schedule['selected_ids'] == [c['candidate_id'] for c in contacts]
    for contact, row in zip(contacts, directions['contacts']):
        assert row['geometry_signature'] == D.signature(contact, directions['normal_depth_m'])
    assert len(contacts) == len(directions['contacts'])
    floor = F.read(name); arrays = I.load_npz(F.output_folder(name)/floor['arrays_file'])
    audit_path = F.output_folder(name)/'audit.json'
    assert I.check_report(audit_path)['passed']
    work_path = root/W.STAGE/'work_volume.json'
    work = None
    if POLICY.ENFORCE_PROCESS_ACCESS:
        assert I.check_report(work_path.parent/'work_volume_audit.json')['passed']
        work = W.WorkVolume.read(work_path)
    paths = [schedule_path, schedule_path.parent/'status.json', path,
        schedule_path.parent/'insertion_directions.json', root/'step_1_needs/needs.json',
        F.output_folder(name)/'floor_contact.json', F.output_folder(name)/'floor_contact.npz', audit_path]
    if work is not None:
        paths += [work_path, work_path.parent/'work_volume.npz', work_path.parent/'work_volume_audit.json']
    return domain, contacts, schedule, arrays, directions, work, paths


class BudgetExhausted(RuntimeError):
    pass


def search(mesh, contacts, records, required, pivot, depth, work=None, edge_budget=2000, progress=None):
    """Finite shared-direction/base/backing/route menu; never modifies contacts."""
    if edge_budget < 1:
        raise ValueError('Use a positive whole-assembly edge budget')
    scale = float(mesh.extents.max())
    common, history = common_directions(records)
    report = dict(passed=False, status='not_constructed', common_directions=common,
        direction_intersections=history, attempts=[], selected_ids=[c['candidate_id'] for c in contacts],
        step3_selection_changed=False, global_impossibility_claimed=False,
        maximum_new_edge_checks=edge_budget, new_edge_checks=0)
    if not contacts:
        report['status'] = 'no_selected_heads'; return report, None
    if not D.nonempty(common):
        report['status'] = 'no_common_certified_head_direction'
        report['reason'] = 'Selected heads have no common certified horizontal insertion direction; unresolved directions are not certified. Step 3 is retained unchanged.'
        return report, None
    original_clear = T.Router.clear
    count = 0
    def clear(router, part):
        nonlocal count
        key = np.asarray(part.vertices, np.float64).tobytes()
        if key not in router.cache:
            if count >= edge_budget: raise BudgetExhausted()
            count += 1
        return original_clear(router, part)
    T.Router.clear = clear
    routers = {}; prepared = {}
    try:
        for allow_roadmap in (False, True):
            for angle in angles(common):
                direction = G.frame(angle)[0]
                if not M.contact_motion(mesh, contacts, direction)['passed']: continue
                for expansion in EXPANSIONS:
                    key = (angle, expansion)
                    if key not in prepared:
                        base_parts, base = open_base(mesh, required, angle, expansion)
                        okay = M.sweep_check(mesh, base_parts, direction)['passed']
                        if okay and work is not None: okay = work.check_parts(base_parts)['passed']
                        prepared[key] = (base_parts, base) if okay else None
                        if not okay:
                            report['attempts'].append(dict(bearing_deg=angle, expansion=expansion, rejection='base_sweep_or_work_volume'))
                    if prepared[key] is None: continue
                    base_parts, base = prepared[key]
                    if angle not in routers: routers[angle] = T.Router(mesh, work, angle, edge_budget=edge_budget)
                    router = routers[angle]
                    basis = G.frame(angle)
                    for factor in DEPTH_FACTORS:
                        grouped = [D.Analyzer(mesh, depth*factor).heads(c) for c in contacts]
                        heads = [p for group in grouped for p in group]
                        if any(p.vertices[:, 2].min() <= scale*1e-10 for p in heads): continue
                        if not M.sweep_check(mesh, heads, direction)['passed']: continue
                        if work is not None and not work.check_parts(heads)['passed']: continue
                        attempt = dict(bearing_deg=angle, expansion=expansion, depth_factor=factor,
                            spatial_search=allow_roadmap, connected_head_ids=[])
                        report['attempts'].append(attempt)
                        bars = []; routes = []
                        for c, head in zip(contacts, grouped):
                            y = float(np.clip((c['center_m']@basis.T)[1], *base['back_local_y_bounds_m']))
                            low, high = base['back_local_y_bounds_m']
                            candidates = [y, (low+high)/2, low+.1*(high-low), high-.1*(high-low)]
                            route = None
                            for target_y in dict.fromkeys(candidates):
                                anchor = COORD.floor(np.array([base['back_local_x_m'], target_y, 0.])@basis)
                                route = router.connect(head, anchor, base['height_m'], allow_roadmap=allow_roadmap)
                                if route is not None: break
                            if route is None:
                                attempt['failed_head_id'] = c['candidate_id']; break
                            bars.extend(route['parts']); routes.append(dict(candidate_id=c['candidate_id'], **route['record']))
                            attempt['connected_head_ids'].append(c['candidate_id'])
                            if progress is not None: progress(heads+bars+base_parts, attempt)
                        print('whole support:', angle, 'deg, expansion', expansion, 'backing', factor,
                              len(routes), '/', len(contacts), 'heads connected', flush=True)
                        if len(routes) != len(contacts): continue
                        parts = heads+bars+base_parts
                        labels = ([f'contact_head_{i:04d}' for i in range(len(heads))]+
                            [f'connector_{i:04d}' for i in range(len(bars))]+[f'ground_strip_{i}' for i in range(len(base_parts))])
                        joined, solid = S.union_parts(parts, scale)
                        if not solid['one_solid']:
                            attempt['rejection'] = 'not_one_connected_solid'; continue
                        sweep = M.sweep_check(mesh, parts, direction)
                        if not sweep['passed']:
                            attempt['rejection'] = 'whole_solid_sweep'; continue
                        if work is not None and not work.check_parts(parts)['passed']:
                            attempt['rejection'] = 'whole_solid_work_volume'; continue
                        interface = np.vstack([p for c in contacts for p in
                            (c['triangles_m'].reshape(-1, 3), c['triangles_m'].mean(axis=1))])
                        gap = float(surface_distances(joined, interface).max())
                        if gap > scale*1e-9:
                            attempt['rejection'] = 'contact_interface_changed'; continue
                        ground, floor_triangles = footprint(parts, pivot, required, scale)
                        if not ground['passed']:
                            attempt['rejection'] = 'floor_hull_does_not_cover_demand'; continue
                        report.update(passed=True, status='one_connected_support_geometry_verified',
                            direction=direction.tolist(), bearing_deg=angle, base=base, routes=routes,
                            backing_depth_factor=factor, solid=solid, sweep=sweep, ground=ground,
                            maximum_contact_gap_m=gap, new_edge_checks=count)
                        return report, dict(parts=parts, labels=labels, joined=joined, floor_triangles=floor_triangles)
        report['status'] = 'no_whole_support_in_search_menu'
    except BudgetExhausted:
        report['status'] = 'whole_support_search_budget_exhausted'
    finally:
        T.Router.clear = original_clear
        report['new_edge_checks'] = count
    return report, None


def bearing(domain, contacts, floor, base):
    """Workpiece plus ONE support, sharing every head and original-pivot reaction."""
    points, normals, owners = Q.contact_rays(domain, contacts, floor['original_pivot_m'])
    owners[owners >= 0] = 0
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    attempts = []; accepted = None
    for mu in (1., 4., 16., 64.):
        matrix, ground = Q.grounded_matrix(points, normals, owners, domain.com, scale, [base], mu)
        solver = Q.BatchSolver(matrix)
        sample = solver.solve(Q.padded_targets(floor['load_wrenches'], scale, 12))
        attempts.append(dict(friction=mu, passed=sample['passed'], diagnostics=sample['diagnostics']))
        if sample['passed']:
            accepted = mu, matrix, ground, solver, sample; break
    report = dict(sampled_passed=accepted is not None, continuous_passed=False, attempts=attempts,
        body_count=1, reactions_shared=True, finite_friction_menu_is_sufficient_not_complete=True)
    if accepted is None: return report, {}
    mu, matrix, ground, solver, sample = accepted
    continuous = solver.solve(Q.padded_targets(floor['continuous_outer_load_wrenches'], scale, 12), certified=True)
    report.update(continuous_passed=continuous['passed'], sufficient_friction_coefficient=mu,
                  continuous_diagnostics=continuous['diagnostics'])
    arrays = dict(contact_points_m=points, contact_normals=normals, contact_owners=owners,
        ground_points_m=ground['points_m'], ground_forces=ground['forces'], ground_owners=ground['owners'],
        moment_origin_m=domain.com, scale=scale, equilibrium_matrix=matrix)
    for prefix, result in [('sample', sample), ('continuous', continuous)]:
        arrays[prefix+'_assignment'] = result['assignment']
        arrays[prefix+'_coefficients_mg'] = result['weights']
        arrays[prefix+'_basis_indices'] = np.asarray(solver.bases, int).reshape(-1, 12)
    return report, arrays


def build(name, edge_budget=2000):
    domain, contacts, schedule, floor, directions, work, paths = read_inputs(name)
    out = OUTPUTS/name/pose_name()/STAGE; out.mkdir(parents=True, exist_ok=True)
    for filename in ('geometry.npz', 'support.stl', 'insertion.gif', 'bearing.npz', 'preview_geometry.npz', 'trajectory.json'):
        (out/filename).unlink(missing_ok=True)
    I.save(out/'status.json', dict(complete=False, status='constructing_one_rigid_support'))
    preview(name, domain, contacts, floor, out, 'Searching one connected support')
    def progress(parts, attempt):
        joined, _ = S.union_parts(parts, float(domain.mesh.extents.max()))
        labels = ['preview_part']*len(parts)
        np.savez_compressed(out/'preview_geometry.npz', **S.pack_parts(parts, labels, joined))
        I.save(out/'progress.json', dict(complete=False, attempt=attempt, candidate_geometry_only=True))
    geometry, module = search(domain.mesh, contacts, directions['contacts'], floor['required_hull_xy_m'],
        floor['original_pivot_m'], directions['normal_depth_m'], work, edge_budget, progress)
    mechanics = dict(sampled_passed=False, continuous_passed=False, status='no_complete_geometry')
    artifacts = {}
    if module is not None:
        np.savez_compressed(out/'geometry.npz', **S.pack_parts(module['parts'], module['labels'], module['joined']),
            floor_triangles_m=module['floor_triangles'])
        module['joined'].export(out/'support.stl', file_type='stl_ascii')
        length = geometry['sweep']['length_m']; direction = np.asarray(geometry['direction'])
        I.save(out/'trajectory.json', dict(direction=direction.tolist(), length_m=length,
            start_translation_m=(-length*direction).tolist(), end_translation_m=[0., 0., 0.],
            motion='Entire printed support translates together; support(t)=support_final-(1-t)*L*a',
            all_heads_move_together=True, continuous_sweep_verified=True))
        mechanics, arrays = bearing(domain, contacts, floor, geometry['base'])
        if arrays: np.savez_compressed(out/'bearing.npz', **arrays)
        for f in ('geometry.npz', 'support.stl', 'trajectory.json', 'bearing.npz'):
            if (out/f).exists(): artifacts[f] = sha256(out/f)
    passed = bool(geometry['passed'] and mechanics['continuous_passed'] and schedule['continuous_coverage_proved'])
    status = ('one_rigid_support_verified' if passed else
              'one_rigid_support_geometry_only' if geometry['passed'] else geometry['status'])
    report = dict(object=name, pose=pose_name(), stage=STAGE, schema=SCHEMA, complete=True,
        passed=passed, status=status, selected_ids=[c['candidate_id'] for c in contacts],
        contact_count=len(contacts), support_count=int(module is not None),
        geometric_assembly_verified=geometry['passed'], geometry=geometry, bearing=mechanics,
        step3_continuous_coverage_proved=bool(schedule['continuous_coverage_proved']),
        step3_selection_changed=False, multi_pose_reuse_verified=False, physical_supports_verified=False,
        process_access_enforced=POLICY.ENFORCE_PROCESS_ACCESS,
        process_access_clearance_verified=bool(work is not None and geometry['passed']),
        scope='Single-pose one-body baseline. All heads are retained. Greedy selection is unchanged. A failed common-direction or shape menu does not trigger reselection or certify global impossibility. Strength and multi-pose reuse are not verified.',
        provenance=dict(inputs=I.hashes(paths), code={**D.code_hashes(), **I.hashes([
            Path(__file__), Path(F.__file__), Path(Q.__file__), Path(M.__file__), Path(S.__file__),
            Path(T.__file__), Path(G.__file__), Path(W.__file__), Path(POLICY.__file__), Path(H.__file__), Path(U.__file__)])}), artifacts=artifacts)
    I.save(out/'connection.json', report)
    digest = sha256(out/'connection.json')
    I.save(out/'status.json', dict(complete=True, status=status, connection_sha256=digest))
    I.save(out/'progress.json', dict(complete=True, status=status, connection_sha256=digest))
    preview(name, domain, contacts, floor, out, status)
    print(name, pose_name(), 'Step 5:', status, '; heads retained:', len(contacts), flush=True)
    return report


def preview(name, domain, contacts, floor, out, status):
    from PIL import Image, ImageDraw
    from step5_connect_support import visual_details as V
    modules = []
    path = out/'geometry.npz'
    if not path.exists(): path = out/'preview_geometry.npz'
    if path.exists(): modules = [(dict(candidate_id='assembly'), I.load_npz(path))]
    page = Image.new('RGB', (2200, 1240), V.R.PAPER); ink = ImageDraw.Draw(page)
    ink.text((30, 20), f'{name} / {pose_name()} / One rigid support', font=V.R.font(38), fill=V.R.INK)
    for i, show_object in enumerate((True, False)):
        picture = V.scene(domain, contacts, modules, 1080, show_object=show_object, xray=show_object, show_interfaces=True)
        page.paste(picture, (20+1100*i, 85))
    ink.text((30, 1170), status.replace('_', ' '), font=V.R.font(28), fill=V.MUTED)
    temporary = out/'.connection.png'; page.save(temporary); temporary.replace(out/'connection.png')


def audit(name):
    domain, contacts, schedule, floor, directions, work, _ = read_inputs(name)
    out = OUTPUTS/name/pose_name()/STAGE
    report = I.check_report(out/'connection.json'); assert report['schema'] == SCHEMA
    state = json.loads((out/'status.json').read_text())
    assert state['complete'] and state['connection_sha256'] == sha256(out/'connection.json')
    common, history = common_directions(directions['contacts'])
    assert report['geometry']['common_directions'] == common
    assert report['geometry']['direction_intersections'] == history
    assert report['selected_ids'] == [c['candidate_id'] for c in contacts]
    checks = dict(all_selected_heads_retained=True, common_direction_intersection_replayed=True)
    if report['geometric_assembly_verified']:
        a = I.load_npz(out/'geometry.npz'); parts = S.unpack_parts(a)
        joined, solid = S.union_parts(parts, float(domain.mesh.extents.max()))
        assert solid['one_solid']
        np.testing.assert_array_equal(joined.vertices, a['union_vertices_m'])
        np.testing.assert_array_equal(joined.faces, a['union_faces'])
        angle = report['geometry']['bearing_deg']; direction = np.asarray(report['geometry']['direction'])
        assert D.contains(common, angle)
        np.testing.assert_allclose(direction, G.frame(angle)[0], atol=1e-14)
        assert M.contact_motion(domain.mesh, contacts, direction)['passed']
        assert M.sweep_check(domain.mesh, parts, direction)['passed']
        if work is not None: assert work.check_parts(parts)['passed']
        for c in contacts:
            points = np.vstack([c['triangles_m'].reshape(-1, 3), c['triangles_m'].mean(axis=1)])
            gap = surface_distances(joined, points).max()
            assert gap <= domain.mesh.extents.max()*1e-9
        ground, tri = footprint(parts, floor['original_pivot_m'], floor['required_hull_xy_m'], float(domain.mesh.extents.max()))
        assert ground['passed']; np.testing.assert_array_equal(tri, a['floor_triangles_m'])
        rebuilt_base, base = open_base(domain.mesh, floor['required_hull_xy_m'], angle, report['geometry']['base']['expansion'])
        assert base == report['geometry']['base']
        saved_base = [part for part, label in zip(parts, a['part_labels']) if str(label).startswith('ground_strip_')]
        assert len(saved_base) == len(rebuilt_base)
        for p, q in zip(saved_base, rebuilt_base): np.testing.assert_array_equal(p.vertices, q.vertices)
        for part, label in zip(parts, a['part_labels']):
            if not str(label).startswith('ground_strip_'): assert part.vertices[:, 2].min() > domain.mesh.extents.max()*1e-10
        exported = trimesh.load(out/'support.stl', process=False)
        np.testing.assert_array_equal(exported.triangles, joined.triangles)
        trajectory = json.loads((out/'trajectory.json').read_text())
        assert trajectory['all_heads_move_together']
        np.testing.assert_allclose(trajectory['start_translation_m'], -trajectory['length_m']*direction)
        checks.update(one_solid=True, whole_sweep_replayed=True, all_contact_interfaces_preserved=True,
                      actual_floor_hull_verified=True, actual_feet_match_bearing_geometry=True, stl_replayed=True)
        if report['bearing']['sampled_passed']:
            from step4_floor_contact.audit import replay
            b = I.load_npz(out/'bearing.npz'); mu = report['bearing']['sufficient_friction_coefficient']
            p, n, owners = Q.contact_rays(domain, contacts, floor['original_pivot_m']); owners[owners >= 0] = 0
            for key, value in [('contact_points_m', p), ('contact_normals', n), ('contact_owners', owners)]:
                np.testing.assert_array_equal(b[key], value)
            matrix, g = Q.grounded_matrix(p, n, owners, domain.com, b['scale'], [base], mu)
            np.testing.assert_array_equal(matrix, b['equilibrium_matrix'])
            for key, value in [('ground_points_m', g['points_m']), ('ground_forces', g['forces']), ('ground_owners', g['owners'])]:
                np.testing.assert_array_equal(b[key], value)
            checks['sampled_bearing'] = replay(b, 'sample', floor['load_wrenches'], 1, mu)
            if report['bearing']['continuous_passed']:
                checks['continuous_bearing'] = replay(b, 'continuous', floor['continuous_outer_load_wrenches'], 1, mu)
                targets = Q.padded_targets(floor['continuous_outer_load_wrenches'], b['scale'], 12)
                for k, basis in enumerate(b['continuous_basis_indices']):
                    ids = np.flatnonzero(b['continuous_assignment'] == k)
                    if len(ids): assert Q.membership(matrix, basis, targets[ids], certified=True)[0].all()
    else:
        assert not (out/'support.stl').exists() and not (out/'geometry.npz').exists()
        if report['status'] == 'no_common_certified_head_direction': assert not D.nonempty(common)
    verified = bool(report['geometric_assembly_verified'] and report['bearing']['continuous_passed'] and schedule['continuous_coverage_proved'])
    assert report['passed'] == verified
    result = dict(object=name, pose=pose_name(), complete=True, passed=True, design_passed=verified,
        design_status=report['status'], checks=checks, evidence_audit_is_not_design_success=True,
        provenance=dict(inputs=I.hashes([out/'connection.json']), code=I.hashes([Path(__file__)])))
    I.save(out/'audit.json', result)
    print(name, pose_name(), 'Step 5 one-body audit passed:', report['status'], flush=True)
    return result


def draw(name, static_only=False):
    from PIL import Image, ImageDraw
    from step5_connect_support import visual_details as V
    domain, contacts, _, floor, _, _, _ = read_inputs(name)
    out = OUTPUTS/name/pose_name()/STAGE; report = I.check_report(out/'connection.json')
    preview(name, domain, contacts, floor, out, report['status'])
    (out/'insertion.gif').unlink(missing_ok=True)
    if report['geometric_assembly_verified'] and not static_only:
        data = I.load_npz(out/'geometry.npz'); trajectory = json.loads((out/'trajectory.json').read_text())
        entry = dict(candidate_id='assembly', ground_color=V.TEAL.tolist())
        start = np.asarray(trajectory['start_translation_m'])
        points = np.vstack([domain.mesh.vertices, data['union_vertices_m'], data['union_vertices_m']+start])
        view = V.fit(points, margin=1.12); ground = V.floor_triangles(points, .04*domain.mesh.extents.max())
        static = V.layer(V.arrays(domain, [], [], ground=ground), view, 700)
        frames = []
        for t in np.linspace(0, 1, 18):
            moving = V.layer(V.arrays(domain, contacts, [(entry, data)], translations=[(1-t)*start], show_object=False), view, 700)
            page = Image.new('RGB', (760, 850), V.R.PAPER); ink = ImageDraw.Draw(page)
            ink.text((25, 20), f'{name} / {pose_name()} / One rigid support', font=V.R.font(25), fill=V.R.INK)
            page.paste(V.merge_layers(static, moving), (30, 80))
            ink.text((25, 800), 'All heads, connectors and base move together.', font=V.R.font(21), fill=V.MUTED)
            frames.append(page)
        frames[0].save(out/'insertion.gif', save_all=True, append_images=frames[1:], duration=130, loop=0)
    artifacts = {p.name: sha256(p) for p in out.glob('*.png')}
    if (out/'insertion.gif').exists(): artifacts['insertion.gif'] = sha256(out/'insertion.gif')
    I.save(out/'views.json', dict(complete=True, one_rigid_support=True, all_heads_shown=True,
        candidate_geometry_is_not_success=not report['geometric_assembly_verified'],
        provenance=dict(inputs=I.hashes([out/'connection.json']), code=I.hashes([Path(__file__), Path(V.__file__)])), artifacts=artifacts))
