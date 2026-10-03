"""Read current results and build explicitly unaccepted supports for exit replay.

Failed constructors have no finished body to animate. For those cases ONLY,
keep the exact contacts, roots and common registration, and use the copied
nearest-floor lofts with stationary-object/floor carving. Motion exclusions are
deliberately retained as material so the user can see the collision. This is a
new diagnostic candidate, never a replacement for the physical search result.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step2_local_support import geometry as G
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support.baseline_current import build_coupled_saddle as S, connection_fallback as CF
from step4_connect_support.baseline_current.fixture_view import cells_for
from step4_connect_support.baseline_current.codesign_port import build_local_bodies as L
from step4_connect_support.baseline_current.run_sequential_k import foot_menu


def read_case(output):
    output = Path(output).resolve()
    report_path = output/'data/report.json'
    report = I.check_report(report_path)
    cache_path = output/'data/codesign_cache/sweep_cache.json'
    cache = json.loads(cache_path.read_text())
    I.check_hashes(cache['inputs'])
    bases, offsets = (np.asarray(report['placement'][key]) for key in ('bases', 'offsets'))
    np.testing.assert_allclose(cache['rotations'], bases, atol=1e-14, rtol=0)
    np.testing.assert_allclose(cache['offsets'], offsets, atol=1e-14, rtol=0)
    direction_ids = report['placement'].get('direction_ids', report['attempts'][-1]['direction_ids'])
    source = output/'data/codesign_inputs_surface/data/independent_input/schedule.json'
    schedule = json.loads(source.read_text())
    tasks, groups, catalogues, paths = [], [], [], [report_path, cache_path, source]
    for pose, relative in zip(report['poses'], schedule['source_reports']):
        p = I.ROOT/relative
        own = I.check_report(p)
        assert own['poses'] == [pose]
        task = read_task(report['object'], pose, folder=p.parent.parent/'step_1_needs')
        contact_path = p.parent/f'final_contacts_{pose}.npz'
        catalogue_path = p.parent.parent/'step2_local_support'/f'candidates_{pose}.json'
        tasks.append(task); groups.append(I.read_contacts(contact_path))
        catalogues.append(np.asarray(json.loads(catalogue_path.read_text())['direction_catalogue']['vectors']))
        paths += [p, contact_path, catalogue_path, *task.inputs]
    directions = np.array([cat[i] for cat, i in zip(catalogues, direction_ids)])
    np.testing.assert_allclose(cache['directions'], directions, atol=1e-14, rtol=0)
    obj = tasks[0].domain.mesh.copy()
    obj.vertices = obj.vertices@bases[0]+offsets[0]
    for k, task in enumerate(tasks):
        np.testing.assert_allclose(task.domain.mesh.vertices@bases[k]+offsets[k], obj.vertices, atol=1e-12, rtol=0)
    records = {r['candidate_id']: r for r in report['generated_support_roots']}
    patches = []
    for k, (task, group) in enumerate(zip(tasks, groups)):
        for c in group:
            record = records[c['candidate_id']]
            displacements = G.vertex_offsets(task.domain.mesh, record['constructor_root_normal_depth_m'])[0]
            cells = [v@bases[k]+offsets[k] for v in cells_for(c, task.domain, displacements)]
            root = L.union([S.solid(G.hull_mesh(v)) for v in cells])
            points = c['triangles_m'].reshape(-1, 3)@bases[k]+offsets[k]
            patch = trimesh.Trimesh(points, np.arange(len(points)).reshape(-1, 3), process=False)
            patches.append(dict(id=c['candidate_id'], owner=k, root=root, mesh=patch))
    return SimpleNamespace(output=output, report=report, object=obj, tasks=tasks,
        groups=groups, poses=report['poses'], bases=bases, offsets=offsets,
        directions=directions, direction_ids=direction_ids,
        exits=-np.einsum('ki,kij->kj', directions, bases), patches=patches,
        demands=[pressure_centers(t.targets/t.scale, t.domain.com)[0] for t in tasks], paths=paths)


def component_preserving(value, required):
    for part in sorted(value.decompose(), key=lambda q: -q.volume()):
        if all(abs(float((r-part).volume()))*S.SCALE**3 <= 8e-14 for r in required):
            return part
    return None


def diagnostic_support(case):
    if case.report['constructed']:
        path = case.output/'shape.obj'
        case.paths.append(path)
        mesh = trimesh.load(path, force='mesh', process=False)
        return mesh, dict(kind='saved_step4_body', diagnostic_candidate=False,
            description='原 Step4 实体，完整验收未通过', source_sha256=I.sha256(path))
    obj = S.solid(case.object)
    relief = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    forbidden = obj.minkowski_sum(relief)

    def carve(value):
        for b, o in zip(case.bases, case.offsets):
            value = value.trim_by_plane(b[2].tolist(), float(b[2]@o/S.SCALE))
        return value-forbidden

    bodies, records = [], []
    for p in case.patches:
        k = p['owner']
        v = S.unpack(p['root']).vertices
        v = v[ConvexHull(v).vertices]
        behind = v+.008*case.directions[k]@case.bases[k]
        try:
            body, record = L.grow_initial_body(v, behind, p['root'], case.bases, case.offsets, k,
                carve=carve, connected_piece=component_preserving, nearby_search=True)
        except RuntimeError as error:
            body, record = p['root'], dict(unreached_floor=True, error=str(error))
        bodies.append(body)
        records.append(dict(id=p['id'], **record))
    # Use the smallest original demand-derived foot menu. These are still
    # diagnostic bearing terminals; original load acceptance is NOT implied.
    feet = next(foot_menu(case, case.bases, case.offsets))
    terminals, skipped = [], []
    for k, polygons in enumerate(feet):
        for j, polygon in enumerate(polygons):
            pad = np.c_[polygon, np.zeros(len(polygon))]@case.bases[k]+case.offsets[k]
            terminal = np.vstack([pad, pad+.003*case.bases[k][2]])
            legal = carve(S.solid(G.hull_mesh(terminal)))
            if legal.is_empty():
                skipped.append([k, j]); continue
            order = sorted(range(len(bodies)), key=lambda i:np.linalg.norm(case.patches[i]['mesh'].centroid-pad.mean(0)))
            for i in order:
                p = case.patches[i]
                v = S.unpack(p['root']).vertices
                candidate = carve(S.solid(G.hull_mesh(np.vstack([v, terminal]))))+bodies[i]
                connected = component_preserving(candidate, [bodies[i], legal])
                if connected is not None:
                    bodies[i] = connected
                    terminals.append(dict(pose=case.poses[k], terminal=j, head=p['id']))
                    break
            else:
                skipped.append([k, j])
    full = L.union(bodies)
    connections = []
    while len(full.decompose()) > 1:
        before = len(full.decompose())
        try:
            full, _, record = CF.connect(full, carve, S.SCALE, case.object.bounds)
        except RuntimeError as error:
            connections.append(dict(failed=True, error=str(error))); break
        connections.append(record)
        assert len(full.decompose()) < before
    mesh = S.unpack(full)
    missing = max(abs(float((p['root']-full).volume()))*S.SCALE**3 for p in case.patches)
    floor_heights = [float(((mesh.vertices-o)@b.T)[:, 2].min()) for b, o in zip(case.bases, case.offsets)]
    overlap = abs(float((obj^full).volume()))*S.SCALE**3
    assert min(floor_heights) > -1e-9 and missing < 8e-14 and overlap < 8e-14
    assert mesh.is_watertight and mesh.is_winding_consistent
    return mesh, dict(kind='stationary_clearance_diagnostic_candidate', diagnostic_candidate=True,
        description='退出失败候选：保留会阻挡运动的材料，用于观察碰撞',
        construction='Copied nearest-floor lofts and carved-envelope joints; stationary object and every floor clipped',
        motion_exclusions_used=False, original_search_replaced=False, full_acceptance_passed=False,
        loads_rechecked=False, roots_preserved=True, maximum_missing_root_volume_m3=missing,
        initial_object_overlap_m3=overlap, minimum_floor_heights_m=floor_heights,
        components=len(full.decompose()), watertight=bool(mesh.is_watertight),
        initial_bodies=records, terminals=terminals, skipped_terminals=skipped, connections=connections)


def prepare(output):
    case = read_case(output)
    folder = case.output/'data/exit_replay'; folder.mkdir(exist_ok=True)
    source_hashes = I.hashes(case.paths)
    code_hashes = I.hashes([Path(__file__), Path(L.__file__), Path(CF.__file__), Path(G.__file__), Path(S.__file__)])
    saved = folder/'support.json'
    if saved.exists():
        old = json.loads(saved.read_text())
        if old['provenance'] == dict(inputs=source_hashes, code=code_hashes):
            I.check_report(saved)
            case.support = trimesh.load(case.output/'exit_support.obj', force='mesh', process=False)
            case.support_record = old['support']
            return case
    mesh, record = diagnostic_support(case)
    # A saved current body is a dependency as well, added by diagnostic_support.
    case.support, case.support_record = mesh, record
    target = case.output/'exit_support.obj'
    mesh.export(target, file_type='obj', digits=17, include_normals=False)
    reloaded = trimesh.load(target, force='mesh', process=False)
    np.testing.assert_allclose(reloaded.vertices, mesh.vertices, atol=1e-14, rtol=0)
    assert reloaded.is_watertight and reloaded.is_winding_consistent
    I.save(saved, dict(complete=True, diagnostic_only=True, support=record,
        poses=case.poses, direction_ids=case.direction_ids,
        fixture_directions_world=case.directions.tolist(), object_directions_common=case.exits.tolist(),
        provenance=dict(inputs=I.hashes(case.paths), code=code_hashes),
        artifacts={'../../exit_support.obj': I.sha256(target)}))
    print('EXIT SUPPORT', case.output.parent.name, record['kind'], len(S.solid(mesh).decompose()), 'components', flush=True)
    return case
