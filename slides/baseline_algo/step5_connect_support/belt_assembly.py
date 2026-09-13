"""One printed body: direction first, loose frame, demand-derived base and thick links."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from step1.needs import OUTPUTS, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step2_local_support import insertion as D
from step4_floor_contact import whole_assembly as F, equilibrium as Q
from step5_connect_support import whole_assembly as A, belt_geometry as B, rigid_path as P
from step5_connect_support import solids as S, visual_details as V, surface_check as U, floor_design as FD, piecewise_path as PP, video

from step5_connect_support import direction_first as X, failure_visuals as FV

STAGE = 'step5_connect_support'
SCHEMA = 'direction_first_loose_frame_v1'


def code_hashes():
    return {**D.code_hashes(), **I.hashes([Path(m.__file__) for m in
        (A, B, P, S, V, U, F, Q, FD, PP, video, X, FV, B.G, B.R, B.T)]) , **I.hashes([Path(__file__),Path(FV.__file__).with_name("failure_viewer.html")])}


def export_geometry(out, filename, parts, labels, scale):
    joined, solid = B.union_parts(parts, scale)
    np.savez_compressed(out/filename, **S.pack_parts(parts, labels, joined))
    return joined, solid


def search(domain, contacts, floor, depth, budget, progress=None, failure=None, direction_record=None):
    return X.search(domain, contacts, floor, depth, budget, progress, failure, direction_record)


def bearing_rays(domain, contacts, pivot, friction):
    """Use the same sufficient floor friction at the workpiece pivot and base."""
    points,normals,owners=Q.contact_rays(domain,contacts,pivot)
    assert owners[0] == -1 and np.count_nonzero(owners<0) == 1
    rays=np.array([[friction,0,1],[-friction,0,1],[0,friction,1],[0,-friction,1]],float)
    return (np.vstack([np.tile(points[0],(4,1)),points[1:]]),
            np.vstack([rays,normals[1:]]),np.r_[np.full(4,-1,int),np.zeros(len(owners)-1,int)])


def bearing(domain, contacts, floor, base):
    """Continue the friction menu after either sampled OR continuous failure."""
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    report = dict(sampled_passed=False, continuous_passed=False, body_count=1, reactions_shared=True,
        attempts=[], original_object_floor_friction_included=True, finite_friction_menu_is_sufficient_not_complete=True)
    arrays = {}
    for mu in (1., 4., 16., 64.):
        points,normals,owners=bearing_rays(domain,contacts,floor['original_pivot_m'],mu)
        matrix, ground = Q.grounded_matrix(points, normals, owners, domain.com, scale, [base], mu)
        solver = Q.BatchSolver(matrix)
        sampled = solver.solve(Q.padded_targets(floor['load_wrenches'], scale, 12))
        continuous = solver.solve(Q.padded_targets(floor['continuous_outer_load_wrenches'], scale, 12), certified=True) if sampled['passed'] else None
        report['attempts'].append(dict(friction=mu, sampled_passed=sampled['passed'],
            sampled_diagnostics=sampled['diagnostics'], continuous_passed=bool(continuous and continuous['passed']),
            continuous_diagnostics=continuous['diagnostics'] if continuous else []))
        if not sampled['passed']: continue
        report.update(sampled_passed=True, continuous_passed=continuous['passed'], sufficient_friction_coefficient=mu)
        arrays = dict(contact_points_m=points, contact_normals=normals, contact_owners=owners,
            ground_points_m=ground['points_m'], ground_forces=ground['forces'], ground_owners=ground['owners'],
            moment_origin_m=domain.com, scale=scale, equilibrium_matrix=matrix)
        for prefix, value in [('sample', sampled), ('continuous', continuous)]:
            arrays[prefix+'_assignment'] = value['assignment']
            arrays[prefix+'_coefficients_mg'] = value['weights']
            arrays[prefix+'_basis_indices'] = np.asarray(solver.bases, int).reshape(-1, 12)
        if continuous['passed']: break
    return report, arrays


def build(name, edge_budget=2000):
    domain, contacts, schedule, floor, directions, work, paths = A.read_inputs(name)
    if work is not None: raise ValueError('This belt baseline expects the current surface-only support policy')
    out = OUTPUTS/name/pose_name()/STAGE; out.mkdir(parents=True, exist_ok=True)
    for filename in ('geometry.npz', 'support.stl', 'belt.npz', 'preview_geometry.npz', 'trajectory.json', 'insertion.gif', 'insertion.mp4', 'bearing.npz', 'heads.npz', 'direction_search.json', 'video_metadata.json', 'support_mm.stl', 'failed_attempt.npz', 'failed_attempt.json'):
        (out/filename).unlink(missing_ok=True)
    for path in out.iterdir():
        if path.is_file() and path.name.startswith(('failure','failed_shape','workpiece_mm')):
            path.unlink()
    I.save(out/'status.json', dict(complete=False, status='finding_direction_before_loose_frame'))
    def progress(stage, parts, labels):
        export_geometry(out, {'heads':'heads.npz', 'belt':'belt.npz', 'assembly':'preview_geometry.npz'}[stage], parts, labels, float(domain.mesh.extents.max()))
        preview(name, domain, contacts, out, stage)
    retained_rank=(-1,-1)
    def failed(stage,parts,labels,info):
        nonlocal retained_rank
        rank=(FV.STAGES.index(stage),len(parts))
        if rank < retained_rank:return
        retained_rank=rank
        # Concatenation preserves disconnected and intersecting failed material.
        joined=trimesh.util.concatenate(parts)
        np.savez_compressed(out/'failed_attempt.npz',**S.pack_parts(parts,labels,joined))
        I.save(out/'failed_attempt.json',dict(info,stage=stage,part_count=len(parts),geometry_file='failed_attempt.npz',
            selection_rule='most advanced construction stage, then most retained material',complete=True))
    depth = directions['normal_depth_m'] if contacts else None
    geometry, module = search(domain, contacts, floor, depth, edge_budget, progress, failed, directions)
    I.save(out/'direction_search.json', geometry.get('direction_search', {}))
    mechanics = dict(sampled_passed=False, continuous_passed=False, status='no_connected_geometry')
    if module is not None:
        np.savez_compressed(out/'geometry.npz', **S.pack_parts(module['parts'], module['labels'], module['joined']), floor_triangles_m=module['floor_triangles'])
        module['joined'].export(out/'support.stl', file_type='stl_ascii')
        millimetres=module['joined'].copy(); millimetres.apply_scale(1000)
        millimetres.export(out/'support_mm.stl', file_type='stl_ascii')
        I.save(out/'trajectory.json', module['trajectory'])
        if module['trajectory']['passed']:
            mechanics, arrays = bearing(domain, contacts, floor, module['base'])
            if arrays: np.savez_compressed(out/'bearing.npz', **arrays)
        else:
            mechanics = dict(sampled_passed=False, continuous_passed=False,
                status='deferred_until_trajectory_verified', verification_attempted=False)
        module['trajectory']['bearing_verified']=bool(mechanics['continuous_passed'])
        I.save(out/'trajectory.json', module['trajectory'])
    else:
        I.save(out/'trajectory.json', dict(passed=False, status=geometry['status'],
            continuous_sweep_verified=False, direction_first=True, global_impossibility_claimed=False))
    passed = bool(module is not None and geometry['passed'] and mechanics['continuous_passed'] and schedule['continuous_coverage_proved'])
    report = dict(object=name, pose=pose_name(), stage=STAGE, schema=SCHEMA, complete=True, passed=passed,
        status='direction_first_support_verified' if passed else geometry['status'],
        selected_ids=[c['candidate_id'] for c in contacts], step3_selection_changed=False,
        geometry=geometry, bearing=mechanics, geometry_constructed=module is not None,
        trajectory_verified=bool(module is not None and geometry['passed']),
        step3_continuous_coverage_proved=bool(schedule['continuous_coverage_proved']),
        process_access_enforced=False, belt_avoids_work_faces=True, direction_first=True,
        frame_clearance_fraction=X.FRAME_GAP_FRACTION, contact_interfaces_preserved=True,
        stl_units={'support.stl':'metres','support_mm.stl':'millimetres'},
        physical_supports_verified=False, strength_verified=False, multi_pose_reuse_verified=False,
        scope='Fixed Step3 contact interfaces. First propose straight withdrawal directions and sweep heads, then construct a loose rear frame, directional open base and thick links with explicit clearance. Full assembly sweep and shared bearing are independently checked. Failure is not a proof against all paths; printing strength and robot kinematics are not certified.',
        provenance=dict(inputs=I.hashes(paths), code=code_hashes()),
        artifacts={f: sha256(out/f) for f in ('geometry.npz', 'support.stl', 'support_mm.stl', 'heads.npz', 'belt.npz', 'direction_search.json', 'trajectory.json', 'bearing.npz', 'failed_attempt.npz', 'failed_attempt.json') if (out/f).exists()})
    I.save(out/'connection.json', report)
    I.save(out/'status.json', dict(complete=True, status=report['status'], connection_sha256=sha256(out/'connection.json')))
    preview(name, domain, contacts, out, report['status'])
    print(name, pose_name(), 'Step5:', report['status'], flush=True)
    return report


def preview(name, domain, contacts, out, status):
    path = next((out/f for f in ('geometry.npz', 'preview_geometry.npz', 'belt.npz', 'heads.npz') if (out/f).exists()), None)
    modules = [(dict(candidate_id='assembly'), I.load_npz(path))] if path else []
    page = Image.new('RGB', (2200, 1260), V.R.PAPER); ink = ImageDraw.Draw(page)
    ink.text((25, 20), 'Direction first / loose frame / open base', font=V.R.font(36), fill=V.R.INK)
    for i, show_object in enumerate((True, False)):
        picture = V.scene(domain, contacts, modules, 1080, show_object=show_object, xray=show_object, show_interfaces=True, labels=False)
        page.paste(picture, (20+1100*i, 85))
    ink.text((25, 1180), status.replace('_', ' '), font=V.R.font(27), fill=V.MUTED)
    ink.text((25, 1220), 'Orange: exact contact patches. Blue: loose frame and thick links. Teal: actual floor material.', font=V.R.font(23), fill=V.MUTED)
    page.save(out/'connection.png')


def audit(name):
    domain, contacts, schedule, floor, directions, _, _ = A.read_inputs(name)
    floor = FD.prepare(floor)
    out = OUTPUTS/name/pose_name()/STAGE
    report = I.check_report(out/'connection.json'); assert report['schema'] == SCHEMA
    state = json.loads((out/'status.json').read_text())
    assert state['complete'] and state['connection_sha256'] == sha256(out/'connection.json')
    assert report['selected_ids'] == [c['candidate_id'] for c in contacts]
    scene = B.Scene(domain.mesh); checks = dict(selected_heads_retained=True)
    choices, diagnostics = X.scheduled_directions(directions, report['geometry']['direction_budget'])
    assert diagnostics == report['geometry']['direction_search']
    assert json.loads((out/'direction_search.json').read_text()) == diagnostics
    heads, head_labels = X.make_heads(domain.mesh, contacts, directions['normal_depth_m']) if contacts else ([], [])
    saved_heads = S.unpack_parts(I.load_npz(out/'heads.npz')) if contacts else []
    if not contacts:
        assert report['status'] == 'no_selected_heads'
        assert not report['geometry_constructed'] and not report['trajectory_verified']
        assert not (out/'heads.npz').exists()
    assert len(saved_heads) == len(heads)
    for saved, actual in zip(saved_heads, heads):
        np.testing.assert_array_equal(saved.vertices, actual.vertices)
        np.testing.assert_array_equal(saved.faces, actual.faces)
    for contact in contacts:
        assert set(contact['source_faces']).isdisjoint(domain.work_ids)
    checks.update(direction_constraints_recomputed=True, exact_original_head_cells_retained=True)
    witnesses=0
    for attempt in report['geometry']['attempts']:
        witness=attempt.get('collision_witness')
        if witness is None:continue
        index=attempt['witness_head_cell'];assert attempt['witness_head_label']==head_labels[index]
        displacement=np.asarray(witness['translation_m'])
        np.testing.assert_allclose(displacement,np.asarray(attempt['direction'])*witness['withdrawal_distance_m'],atol=1e-14)
        moved=heads[index].copy();moved.vertices=moved.vertices+displacement
        volume=scene.volume(moved)
        assert volume>1e-11*scene.scale**3
        np.testing.assert_allclose(volume,witness['intersection_volume_m3'],rtol=1e-10,atol=1e-18)
        witnesses+=1
    checks['actual_head_collision_witnesses_replayed']=witnesses
    if (out/'belt.npz').exists():
        direction = np.asarray(report['geometry']['withdrawal_direction'])
        belt = report['geometry']['belt']; swept=X.SweptScene(domain.mesh,direction,belt['gap_m'])
        swept.head_labels=head_labels
        rebuilt, frame = X.loose_frame(swept, heads, directions['normal_depth_m'],
                                       belt['gap_m'], belt['rear_extension_m'])
        assert rebuilt == belt and frame is not None
        saved = S.unpack_parts(I.load_npz(out/'belt.npz'))
        expected = heads+frame['necks']+frame['frame']
        assert len(saved) == len(expected)
        for first, second in zip(saved, expected):
            np.testing.assert_array_equal(first.vertices,second.vertices)
            np.testing.assert_array_equal(first.faces,second.faces)
        assert all(swept.clear(part,clearance=0.) for part in heads)
        checks.update(belt_connected=True,loose_frame_clearance_and_sweep_verified=True)
    if report['geometry_constructed']:
        data = I.load_npz(out/'geometry.npz'); parts = S.unpack_parts(data)
        joined, solid = B.union_parts(parts, scene.scale); assert solid['one_solid']
        assert all(scene.clear(p, ground=str(label).startswith('ground_strip_')) for p, label in zip(parts, data['part_labels']))
        for c in contacts:
            points = np.vstack([c['triangles_m'].reshape(-1, 3), c['triangles_m'].mean(axis=1)])
            assert U.surface_distances(joined, points).max() <= scene.scale*1e-9
        ground, tri = A.footprint(parts, floor['original_pivot_m'], floor['required_hull_xy_m'], scene.scale)
        assert ground['passed']; np.testing.assert_array_equal(tri, data['floor_triangles_m'])
        np.testing.assert_array_equal(joined.vertices, data['union_vertices_m'])
        np.testing.assert_array_equal(joined.faces, data['union_faces'])
        exported = trimesh.load(out/'support.stl', process=False)
        np.testing.assert_array_equal(exported.triangles, joined.triangles)
        base = report['geometry']['base']
        direction=np.asarray(report['geometry']['withdrawal_direction'])
        gap=report['geometry']['actual_frame_gap_m'];swept=X.SweptScene(domain.mesh,direction,gap)
        if base['kind']=='directional_open_u':
            rebuilt, record = X.open_u(domain.mesh,floor,direction,gap,base['expansion'],swept)
        else:
            rebuilt, record = B.open_ring(domain.mesh,base['seed_polygon_xy_m'],floor['required_hull_xy_m'],
                floor['original_pivot_m'],base['bearing_deg'],base['expansion'],base['cut_fraction'],swept)
            record=dict(record,kind='directional_open_ring',seed_polygon_xy_m=base['seed_polygon_xy_m'])
        assert record == base
        saved = [p for p, label in zip(parts, data['part_labels']) if str(label).startswith('ground_strip_')]
        assert len(saved) == len(rebuilt)
        for p, q in zip(saved, rebuilt): np.testing.assert_array_equal(p.vertices, q.vertices)
        for part,label in zip(parts,data['part_labels']):
            label=str(label)
            clearance=(0. if label.startswith('contact_head_') else
                       report['geometry']['belt']['neck_gap_m'] if label.startswith('neck_') else gap)
            assert swept.clear(part,ground=label.startswith('ground_strip_'),clearance=clearance)
        millimetres=trimesh.load(out/'support_mm.stl',process=False)
        np.testing.assert_allclose(millimetres.triangles/1000,joined.triangles,atol=1e-12,rtol=1e-12)
        checks.update(noncontact_clearance_verified=True,full_translation_ray_checked=True,millimetre_stl_checked=True)
        trajectory = json.loads((out/'trajectory.json').read_text())
        assert trajectory == report['geometry']['trajectory']
        if trajectory['passed']:
            checks['continuous_rigid_trajectory'] = (PP.replay if trajectory.get('path_kind') == 'piecewise_rigid' else P.replay)(scene, parts, trajectory)
        if report['bearing']['sampled_passed']:
            from step4_floor_contact.audit import replay
            arrays = I.load_npz(out/'bearing.npz'); mu = report['bearing']['sufficient_friction_coefficient']
            p,n,owners=bearing_rays(domain,contacts,floor['original_pivot_m'],mu)
            matrix, g = Q.grounded_matrix(p, n, owners, domain.com, arrays['scale'], [base], mu)
            np.testing.assert_array_equal(matrix, arrays['equilibrium_matrix'])
            for key, value in [('contact_points_m', p), ('contact_normals', n), ('contact_owners', owners),
                ('ground_points_m', g['points_m']), ('ground_forces', g['forces']), ('ground_owners', g['owners'])]:
                np.testing.assert_array_equal(arrays[key], value)
            checks['sampled_bearing'] = replay(arrays, 'sample', floor['load_wrenches'], 1, mu)
            if report['bearing']['continuous_passed']:
                checks['continuous_bearing'] = replay(arrays, 'continuous', floor['continuous_outer_load_wrenches'], 1, mu)
                targets = Q.padded_targets(floor['continuous_outer_load_wrenches'], arrays['scale'], 12)
                for k, ids in enumerate(arrays['continuous_basis_indices']):
                    which = np.flatnonzero(arrays['continuous_assignment'] == k)
                    if len(which): assert Q.membership(matrix, ids, targets[which], certified=True)[0].all()
        checks.update(one_solid=True, actual_open_ring_hull_verified=True, contact_interfaces_preserved=True, exported_stl_checked=True)
    verified = bool(report['geometry_constructed'] and report['trajectory_verified'] and report['bearing']['continuous_passed'] and schedule['continuous_coverage_proved'])
    assert report['passed'] == verified
    result = dict(object=name, pose=pose_name(), complete=True, passed=True, design_passed=verified,
        design_status=report['status'], checks=checks, evidence_audit_is_not_design_success=True,
        provenance=dict(inputs=I.hashes([out/'connection.json']), code=code_hashes()))
    I.save(out/'audit.json', result)
    print(name, pose_name(), 'Step5 audit passed:', report['status'], flush=True)
    return result


def draw(name, static_only=False):
    domain, contacts, _, _, _, _, _ = A.read_inputs(name)
    out = OUTPUTS/name/pose_name()/STAGE; report = I.check_report(out/'connection.json')
    preview(name, domain, contacts, out, report['status'])
    if report['trajectory_verified'] and not static_only:
        data = I.load_npz(out/'geometry.npz'); path = report['geometry']['trajectory']
        from step5_connect_support import video
        video.render(domain, data, path, out)
    if not report['passed']:
        FV.saved_failure(name,static_only)
    I.save(out/'views.json', dict(complete=True, failure_visualization_saved=not report['passed'], direction_first_loose_frame_drawn=True, trajectory_success=report['trajectory_verified'],
        contact_labels_drawn=False,
        provenance=dict(inputs=I.hashes([out/'connection.json']), code=code_hashes()),
        artifacts={p.name: sha256(p) for p in out.iterdir() if p.suffix in ('.png', '.gif', '.mp4') and not p.name.startswith('proposal_')}))
