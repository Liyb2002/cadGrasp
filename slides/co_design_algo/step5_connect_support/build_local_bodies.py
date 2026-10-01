"""B/pose1+3: local head bodies with disconnected landing faces.

This is a concrete specimen, not a global geometry optimizer. Each original
head grows toward its nearest legal floor; additional bearing terminals reshape
nearby bodies by direct oblique lofts. No floor perimeter is constructed.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import trimesh

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support import build_coupled_saddle as S
from step2_local_support import geometry as G
from step5_connect_support.build_shared_geometry import export_viewer, pack
from step5_connect_support.refresh_shared_geometry_view import write_viewer
from step5_connect_support import connection_fallback as CF
from step5_connect_support import head_registration as H


def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def write_mesh(path, value):
    mesh = S.unpack(value)
    mesh.export(path, file_type='obj', digits=17, include_normals=False)
    return mesh


def floor_targets(vertices, bases, offsets, head_pose, policy='nearest'):
    """Rank planes by minimum Euclidean head-solid/plane distance in metres."""
    if policy not in ('nearest', 'opposite'):
        raise ValueError(f'Unknown initial floor policy: {policy}')
    if policy == 'opposite' and len(bases) != 2:
        raise ValueError('Opposite-floor replay requires exactly two poses')
    candidates = []
    for floor, (basis, offset) in enumerate(zip(bases, offsets)):
        height = (vertices-offset)@basis[2]
        distance = 0. if height.min() <= 0 <= height.max() else float(np.abs(height).min())
        if policy == 'nearest' or floor == 1-head_pose:
            candidates.append((distance, floor))
    return sorted(candidates)


def grow_initial_body(vertices, root, original, bases, offsets, head_pose, *,
                      carve, connected_piece, policy='nearest', nearby_search=False):
    """Try complete reference-style bodies in nearest-plane order."""
    targets = floor_targets(vertices, bases, offsets, head_pose, policy)
    seed_options = [(np.zeros(2), 0.)]
    if nearby_search:
        seed_options += [(r*np.array([np.cos(a),np.sin(a)]),r)
            for r in (.012,.025,.045,.07,.10) for a in np.arange(8)*np.pi/4]
    rejected = []
    for distance, floor in targets:
        w = S.local_to_world(vertices, bases[floor], offsets[floor])
        xy = w[:, :2]; center = xy.mean(0); xy = center+.7*(xy-center)
        reasons = set()
        for shift, radius in seed_options:
            pad = np.c_[xy+shift, np.zeros(len(xy))]@bases[floor]+offsets[floor]
            proposed = carve(S.solid(G.hull_mesh(np.vstack([vertices, root, pad]))))+original
            connected = connected_piece(proposed, [original])
            if connected is None:
                reasons.add('head_not_preserved_in_one_component'); continue
            world = S.local_to_world(S.unpack(connected).vertices, bases[floor], offsets[floor])
            landing = world[np.abs(world[:,2]) < 1e-9, :2]
            if len(landing) < 3 or S.MultiPoint(landing).convex_hull.area < 1e-6:
                reasons.add('no_connected_landing_area'); continue
            if not nearby_search and len(proposed.decompose()) != 1:
                reasons.add('disconnected_clipping_parts'); continue
            record = dict(policy=policy, target_floor_index=floor,
                distance_metric='minimum_head_solid_to_plane_distance_m', distance_m=distance,
                projection_shift_xy_m=shift.tolist(), projection_shift_radius_m=radius,
                candidate_floors=[dict(floor_index=f, distance_m=d) for d,f in targets],
                rejected_nearer_floors=rejected)
            return connected, record
        rejected.append(dict(floor_index=floor, distance_m=distance, reasons=sorted(reasons)))
    raise RuntimeError('An initial local head body cannot reach a legal floor: '+json.dumps(rejected))


def build(work, feet, assignments=None, *, case=None, placement=None, allow_failed=False,
          skip_unreachable=False, floor_policy='nearest', verify=False, cache_dir=None):
    began = tick = time.perf_counter(); timings = {}
    if case is None:
        tasks, groups, heads, directions, source, schedule, paths = S.inputs()
        bases = np.asarray([np.eye(3), S.ROTATION])
        offsets = np.asarray([np.zeros(3), -S.TRANSLATION@S.ROTATION])
        paths = paths+[Path(__file__).with_name('local_body_case.json')]
    else:
        tasks, groups, heads = case.tasks, case.groups, case.heads
        source, schedule, paths = case.source, case.schedule, case.paths
        bases, offsets = np.asarray(placement['bases']), np.asarray(placement['offsets'])
        directions = np.asarray(placement['directions'])
    registered, registration = H.register(groups, heads, bases, offsets)
    if not registration['all_heads_above_both_floors']:
        raise ValueError('Registered physical heads cross a floor')
    if case is not None:
        floor_check = H.floor_compatibility(tasks, case.demands, bases, offsets)
        if not floor_check['passed']:
            raise ValueError('Exact shared-head registration fails the floor necessary condition: '
                             + json.dumps(floor_check))
    if assignments is not None and any(i < 0 or i >= len(registered)
                                      for row in assignments for i in row):
        raise ValueError('Foot assignments must refer to the five registered physical heads')
    active_poses = {h.ident: h.active_poses for h in registered}
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else work
    cache.mkdir(parents=True, exist_ok=True)
    poses = [p.pose for p in tasks]
    object_name = tasks[0].domain.data['object']
    cache_key = dict(inputs=S.I.hashes(paths), rotations=bases.tolist(),
                     offsets=offsets.tolist(), directions=np.asarray(directions).tolist(),
                     relief_m=S.RELIEF, sweep_length_m=S.SWEEP_LENGTH,
                     sweep_code=S.I.sha256(Path(S.swept_solid.__code__.co_filename)))
    cache_file = cache/'sweep_cache.json'
    reuse = cache_file.exists() and json.loads(cache_file.read_text()) == cache_key
    sweeps = []
    for k, p in enumerate(tasks):
        path = cache/f'sweep{k}.npz'
        if reuse and path.exists():
            data = np.load(path); sweep = trimesh.Trimesh(data['v'], data['f'], process=False)
        else:
            obj = trimesh.Trimesh(p.domain.mesh.vertices@bases[k]+offsets[k], p.domain.mesh.faces, process=False)
            sweep = S.swept_solid(obj, -S.SWEEP_LENGTH*directions[k]@bases[k])
            np.savez_compressed(path, v=sweep.vertices, f=sweep.faces)
        sweeps.append(sweep)
    forbidden_file = cache/'forbidden.npz'
    if reuse and forbidden_file.exists():
        data = np.load(forbidden_file)
        forbidden = S.solid(trimesh.Trimesh(data['v'], data['f'], process=False))
    else:
        box = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        forbidden = union([S.solid(m).minkowski_sum(box) for m in sweeps])
        m = S.unpack(forbidden); np.savez_compressed(forbidden_file, v=m.vertices, f=m.faces)
        # Use the serialized boundary on BOTH cold and warm runs. Keeping the
        # unevaluated Boolean tree only on a cold run changes near-tie bridges.
        forbidden = S.solid(trimesh.Trimesh(m.vertices, m.faces, process=False))
    cache_file.write_text(json.dumps(cache_key, indent=2))
    print('Withdrawal exclusion ready; cache hit:', reuse, flush=True)
    now = time.perf_counter(); timings['sweep_preparation'] = now-tick; tick = now
    plan_file = cache/'construction_plan.json'
    plan_key = dict(inputs=cache_key, code=S.I.sha256(Path(__file__)),
                    fallback_code=S.I.sha256(Path(CF.__file__)),
                    feet=np.asarray(feet, dtype=object).tolist(), assignments=assignments,
                    floor_policy=floor_policy, skip_unreachable=skip_unreachable)
    # Geometry and parameters must match exactly. The cache stores decisions,
    # not the final mesh: all selected lofts and bridges are still constructed.
    plan_key = json.loads(json.dumps(plan_key))
    saved_plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    plan = saved_plan if saved_plan.get('key') == plan_key else None
    chosen = {(a['floor_index'], a['terminal']):a['head_body'] for a in plan['terminals']} if plan else {}

    def carve(value):
        for basis, offset in zip(bases, offsets):
            normal = basis[2]
            value = value.trim_by_plane(normal.tolist(), float(normal@offset/S.SCALE))
        return value-forbidden

    def fallback_carve(value):
        value = carve(value)
        # An expanded fallback must also finish outside the object after the
        # finite 500 mm sweep. Keep it behind each terminal separation plane.
        for task, direction, basis, offset in zip(tasks, directions, bases, offsets):
            backward = -direction@basis
            limit = (task.domain.mesh.vertices@(-direction)).min()+S.SWEEP_LENGTH+offset@backward-1e-6
            value = value.trim_by_plane((-backward).tolist(), float(-limit/S.SCALE))
        return value

    def connected_piece(value, required):
        for component in sorted(value.decompose(), key=lambda q:-q.volume()):
            if all(abs(float((item-component).volume()))*S.SCALE**3 <= 8e-14 for item in required):
                return component
        return None

    patches, bodies = [], []
    built_ids = set()
    for k in range(len(tasks)):
        for contact, cells in zip(groups[k], heads[k]):
            if contact['candidate_id'] in built_ids:
                continue
            built_ids.add(contact['candidate_id'])
            local = [v@bases[k]+offsets[k] for v in cells]
            v = np.concatenate(local); v = v[ConvexHull(v).vertices]
            original = union([S.solid(G.hull_mesh(q)) for q in local])
            root = v+.008*directions[k]@bases[k]
            body = None
            seed_file = cache/f'initial_body_{len(patches)}.npz'
            seed_key = json.dumps(dict(inputs=cache_key, code=S.I.sha256(Path(__file__)),
                head_index=len(patches), floor_policy=floor_policy,
                nearby_search=skip_unreachable), sort_keys=True)
            if seed_file.exists():
                with np.load(seed_file) as saved:
                    if 'seed_key' in saved and str(saved['seed_key']) == seed_key:
                        body = S.solid(trimesh.Trimesh(saved['v'], saved['f'], process=False))
                        initial_floor = json.loads(str(saved['initial_floor_json']))
            if body is None:
                body, initial_floor = grow_initial_body(v, root, original, bases, offsets, k,
                    carve=carve, connected_piece=connected_piece, policy=floor_policy,
                    nearby_search=skip_unreachable)
                m = S.unpack(body)
                np.savez_compressed(seed_file, v=m.vertices, f=m.faces, seed_key=seed_key,
                    initial_floor_json=json.dumps(initial_floor))
                body = S.solid(trimesh.Trimesh(m.vertices, m.faces, process=False))
            initial_floor['target_pose'] = poses[initial_floor['target_floor_index']]
            print('Initial body', len(patches), poses[k], '->', initial_floor['target_pose'],
                  round(initial_floor['distance_m']*1000, 3), 'mm', flush=True)
            patches.append(dict(pose=k, id=contact['candidate_id'], v=v, root=root,
                                active_poses=active_poses[contact['candidate_id']],
                                original=original, initial_floor=initial_floor))
            bodies.append(body)

    now = time.perf_counter(); timings['initial_bodies'] = now-tick; tick = now
    attachments, skipped_terminals = [], []
    for k, polygons in enumerate(feet):
        for j, polygon in enumerate(polygons):
            xy = np.asarray(polygon)
            pad = np.c_[xy, np.zeros(len(xy))]@bases[k]+offsets[k]
            terminal = np.vstack([pad, pad+.003*bases[k][2]])
            actual_terminal = carve(S.solid(G.hull_mesh(terminal)))
            terminal_mesh = S.unpack(actual_terminal)
            on_floor = S.local_to_world(terminal_mesh.vertices, bases[k], offsets[k])
            floor_vertices = on_floor[np.abs(on_floor[:, 2]) < 1e-9, :2]
            if len(floor_vertices) < 3 or S.MultiPoint(floor_vertices).convex_hull.area < 1e-6:
                if skip_unreachable:
                    skipped_terminals.append(dict(pose=k,terminal=j,reason='no_actual_floor_area'))
                    continue
                raise RuntimeError(f'Terminal {k}/{j} has no usable actual floor area')
            choices = []
            for index, patch in enumerate(patches):
                if assignments is not None and index != assignments[k][j]:
                    continue
                if plan and (k,j) in chosen and index != chosen[k,j]:
                    continue
                addition = carve(S.solid(G.hull_mesh(np.vstack([patch['v'], patch['root'], terminal]))))
                proposed = connected_piece(bodies[index]+addition,[bodies[index],actual_terminal])
                if proposed is None:
                    continue
                missing = abs(float((actual_terminal-proposed).volume()))*S.SCALE**3
                if missing > 8e-14:
                    continue
                distance = float(np.linalg.norm(pad.mean(0)-patch['v'], axis=1).min())
                added_cm3 = float((proposed.volume()-bodies[index].volume())*S.SCALE**3*1e6)
                choices.append((distance+added_cm3*.001, index, proposed, distance, added_cm3))
            if not choices:
                if skip_unreachable:
                    skipped_terminals.append(dict(pose=k,terminal=j,reason='no_direct_connected_loft'))
                    continue
                raise RuntimeError(f'No direct connected local loft reaches floor {k}, terminal {j}')
            _, index, bodies[index], distance, added_cm3 = min(choices, key=lambda x: x[0])
            attachments.append(dict(floor_pose=tasks[k].pose, terminal=j, head_body=index,
                head_pose=tasks[patches[index]['pose']].pose, cross_pose_endpoint=patches[index]['pose'] != k,
                polygon_xy_m=xy.tolist(), minimum_head_vertex_to_terminal_center_m=distance,
                added_volume_cm3=added_cm3, direct_local_loft=True))
            print('Local terminal', attachments[-1], flush=True)

    now = time.perf_counter(); timings['terminals'] = now-tick; tick = now
    full = union(bodies)
    connectors, connections = [], []
    bead = trimesh.creation.icosphere(subdivisions=1, radius=.004).vertices
    up = bases[0][2]+bases[1][2]
    up = up/np.linalg.norm(up) if np.linalg.norm(up)>1e-9 else bases[0][0]
    if plan:
        for connection in plan['connections']:
            if connection.get('kind') == 'carved_connection_envelope':
                joined, bridge = CF.replay(full, connection, fallback_carve, S.SCALE)
                connections.append(dict(connection, cached_replay=True,
                    added_volume_cm3=float((joined.volume()-full.volume())*S.SCALE**3*1e6)))
                connectors.append(bridge); full = joined
                continue
            route = np.asarray(connection['path_m'])
            bridge = carve(union([S.solid(G.hull_mesh(np.vstack([x+bead, y+bead])))
                                  for x,y in zip(route[:-1],route[1:])]))
            previous = len(full.decompose())
            joined = full+bridge
            if len(joined.decompose()) >= previous:
                raise RuntimeError('Cached bridge no longer joins components; clear this construction cache')
            connections.append(dict(connection, added_volume_cm3=float((joined.volume()-full.volume())*S.SCALE**3*1e6)))
            connectors.append(bridge); full = joined
    for iteration in range(12):
        components = full.decompose()
        if len(components) == 1:
            break
        meshes = [S.unpack(c) for c in components]
        trials = []
        for i in range(len(components)):
            va = meshes[i].vertices
            valid = np.ones(len(va), bool)
            for basis, offset in zip(bases, offsets):
                valid &= va@basis[2]-basis[2]@offset > .0045
            va = va[valid]
            for j in range(i):
                vb = meshes[j].vertices
                valid = np.ones(len(vb), bool)
                for basis, offset in zip(bases, offsets):
                    valid &= vb@basis[2]-basis[2]@offset > .0045
                vb = vb[valid]
                if not len(va) or not len(vb):
                    continue
                distances, near = cKDTree(vb).query(va)
                selected = []
                for q in np.argsort(distances):
                    a, b = va[q], vb[near[q]]
                    if any(np.linalg.norm(a-x[0])+np.linalg.norm(b-x[1]) < .008 for x in selected):
                        continue
                    selected.append((a, b)); trials.append((distances[q], i, j, a, b))
                    if len(selected) >= 12:
                        break
        success = False
        lifts = (0., .008, .016, .024, .04, .06) if skip_unreachable else (0., .008, .016, .024)
        for lift in lifts:
            for distance, i, j, a, b in sorted(trials, key=lambda x: x[0]):
                if distance > (.12 if skip_unreachable else .09):
                    continue
                route = [a, b] if lift == 0 else [a, (a+b)/2+lift*up, b]
                bridge = carve(union([S.solid(G.hull_mesh(np.vstack([x+bead, y+bead])))
                                      for x, y in zip(route[:-1], route[1:])]))
                if any((bridge^components[c]).volume()*S.SCALE**3 < 1e-12 for c in (i, j)):
                    continue
                joined = full+bridge
                if len(joined.decompose()) >= len(components):
                    continue
                connections.append(dict(path_m=np.asarray(route).tolist(), radius_m=.004,
                    added_volume_cm3=float((joined.volume()-full.volume())*S.SCALE**3*1e6)))
                connectors.append(bridge); full = joined; success = True
                print('Body joint', connections[-1], 'components', len(full.decompose()), flush=True)
                break
            if success:
                break
        if not success:
            full, bridge, record = CF.connect(full, fallback_carve, S.SCALE,
                                              S.unpack(forbidden).bounds)
            connectors.append(bridge); connections.append(record)
            print('Fallback connection', record, flush=True)
    mesh = S.unpack(full)
    record = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
        consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    record['one_solid'] = bool(record['component_count'] == 1 and record['watertight'] and record['consistently_wound'] and mesh.volume > 0)
    if not record['one_solid']:
        raise RuntimeError('Local bodies did not form one closed solid')
    plan_file.write_text(json.dumps(dict(key=plan_key, terminals=[dict(
        floor_index=poses.index(a['floor_pose']), terminal=a['terminal'], head_body=a['head_body'])
        for a in attachments], connections=connections), indent=2))
    probe = mesh.copy(); probe.apply_scale(1000)
    import io
    probe = trimesh.load(io.BytesIO(trimesh.exchange.stl.export_stl_ascii(probe).encode()), file_type='stl', force='mesh')
    if not probe.is_watertight or not probe.is_winding_consistent:
        full = full.simplify(1e-10/S.SCALE)
        mesh = S.unpack(full)
        record.update(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
                      consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
        if not record['watertight'] or not record['consistently_wound'] or record['component_count'] != 1:
            raise RuntimeError('Numerical sliver cleanup damaged the solid')
    passed = None
    checks, certificate = [], {}
    now = time.perf_counter(); timings['connections_and_topology'] = now-tick; tick = now
    if verify:
        passed = True
        try:
            checks, certificate = S.verify(tasks, groups, heads, directions, bases, offsets, mesh, sweeps)
        except RuntimeError as error:
            if not allow_failed or not hasattr(error, 'checks'):
                raise
            passed = False
            checks, certificate = error.checks, error.certificate
    now = time.perf_counter(); timings['full_verification'] = now-tick if verify else 0.; tick = now
    np.savez_compressed(work/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces, rotations=bases, local_offsets_m=offsets)
    certificate_name = ('equilibrium.npz' if passed else 'equilibrium_diagnostic.npz') if verify else None
    if certificate_name:
        np.savez_compressed(work/certificate_name, **certificate)
    write_mesh(work/'fixture.obj', full)
    mm = mesh.copy(); mm.apply_scale(1000)
    (work/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
    reloaded = trimesh.load(work/'fixture_mm.stl', force='mesh')
    if not reloaded.is_watertight or not reloaded.is_winding_consistent:
        raise RuntimeError('Exported STL is not a closed surface')
    np.testing.assert_allclose(reloaded.extents, mm.extents, atol=1e-10, rtol=0)
    body_records = []
    for i, (body, patch) in enumerate(zip(bodies, patches)):
        m = write_mesh(work/f'body{i}.obj', body)
        body_records.append(dict(index=i, head_pose=tasks[patch['pose']].pose,
            active_poses=[poses[k] for k in patch['active_poses']],
            candidate_id=patch['id'], volume_cm3=float(m.volume*1e6), file=f'body{i}.obj',
            initial_floor=patch['initial_floor']))
    if connectors:
        write_mesh(work/'bridges.obj', union(connectors))
    design = dict(initial_floor_policy=floor_policy, feet_xy_m=feet, attachments=attachments,
        skipped_terminals=skipped_terminals, local_bodies=body_records, connections=connections)
    S.I.save(work/'design.json', design)
    now = time.perf_counter(); timings['geometry_export'] = now-tick
    timings['total_before_viewer'] = now-began
    artifacts = ['fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'design.json']
    if certificate_name:
        artifacts.append(certificate_name)
    artifacts += [r['file'] for r in body_records]+(['bridges.obj'] if connectors else [])
    code = [Path(__file__), Path(H.__file__), Path(CF.__file__), Path(S.__file__), Path(S.swept_solid.__code__.co_filename),
            Path(S.Q.__file__), Path(S.W.__file__), Path(G.__file__), Path(S.SOL.__file__),
            Path(S.FLOOR.__file__), Path(S.bearing_rays.__code__.co_filename),
            Path(S.surface_distances.__code__.co_filename), Path(export_viewer.__code__.co_filename),
            Path(__file__).with_name('shared_geometry_viewer.html'), Path(__file__).with_name('export_shared_geometry.cjs')]
    report = dict(schema=f'{floor_policy}_floor_local_bodies_v1', object=object_name,
        poses=poses, particle=schedule['particle'], complete=True, passed=passed,
        timings_seconds=timings,
        verification=dict(performed=verify, independent_audit_performed=False,
            mode='full' if verify else 'construction_only',
            original_loads_resolved=verify,
            note='Closed connected geometry is constructed; omitted final checks are not reported as passed.'),
        withdrawal_cache=dict(hit=reuse, construction_plan_hit=plan is not None, directory=str(cache)),
        connection_fallback=dict(used=any(c.get('kind') == 'carved_connection_envelope' for c in connections),
            count=sum(c.get('kind') == 'carved_connection_envelope' for c in connections),
            method='expand local envelope, subtract exclusions; full free-envelope fallback'),
        physical_head_definition='five_unique_heads_one_shared_patch', registration=registration,
        status=('geometry_and_fixed_sample_equilibrium_passed' if passed else 'connected_geometry_failed_equilibrium_or_exit') if verify else 'connected_geometry_constructed_without_final_audit', design='local_head_bodies_with_disconnected_landing_faces',
        same_rigid_solid_in_both_poses=True, contact_groups=5, contact_patch_count=len(registered),
        shared_branch_definition='one complete contact surface and head solid shared by both poses',
        original_task_poses_changed=False, original_active_contacts_preserved=True,
        support_mass_ignored=True, extra_loads_added=False, structural_strength_verified=False,
        dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6), solid=record, checks=checks,
        source_schedule=str((source/'schedule.json').relative_to(S.ROOT)),
        task_fixture_transforms=[dict(pose=p.pose, rotation=b.tolist(), translation_m=(-b@o).tolist()) for p,b,o in zip(tasks,bases,offsets)],
        body_design=dict(method=f'head-to-{floor_policy}-floor local lofts, reshaped by discrete bearing terminals, joined above ground',
            initial_floor_policy=floor_policy,
            floor_perimeter_constructed=False, local_bodies=body_records, attachments=attachments, connections=connections,
            optimality_claim=False, local_role_reuse_verified=False,
            interpretation='Geometric construction; endpoint usefulness and direct material witnesses require separate audit'),
        presentation_description='五个物理头，黄色头在两种姿态下使用同一接触面与实体。局部身体连接为一件支撑；最终受力与退出验收状态见报告。',
        exports=dict(obj_units='m', stl_units='mm', stl_encoding='ASCII', reloaded_stl_watertight=True),
        artifacts={name:S.I.sha256(work/name) for name in artifacts},
        provenance=dict(inputs=S.I.hashes(paths), code=S.I.hashes(code)))
    S.I.save(work/'report.json', report)
    colors = {schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in schedule['selected_ids'] if i not in colors], ['#ac7098','#7196c0','#50a59b','#77a76a']))
    visual = [(h.ident, G.hull_mesh(np.concatenate(h.cells))) for h in registered]
    export_viewer(work, mesh, visual, report, tasks, bases, offsets, colors)
    html=(work/'index.html').read_text()
    data,_=json.JSONDecoder().raw_decode(html.split('const DATA=',1)[1])
    labels=[p['id']+' · '+tasks[p['pose']].pose for p in patches]
    data['local_bodies']=[dict(label=labels[i],id=p['id'],head_pose=tasks[p['pose']].pose,
        **pack(S.unpack(body))) for i,(p,body) in enumerate(zip(patches,bodies))]
    write_viewer(work,data)
    print(('VERIFIED' if passed else 'CONNECTED BUT FAILED') if verify else 'CONSTRUCTED; FINAL AUDIT DISABLED', record, report['volume_cm3'], flush=True)
    return artifacts+['report.json', 'index.html']


def run(parameters, output=None, work=None, floor_policy='nearest', verify=False):
    parameters = json.loads(Path(parameters).read_text())
    target = Path(output) if output else S.pair_folder('B', S.POSES, 'step3_scheculer').parent/'step5'
    with tempfile.TemporaryDirectory(prefix='cadgrasp_local_bodies_') as temporary:
        stage = Path(temporary)
        artifacts = build(stage, parameters['feet_xy_m'], parameters.get('assignments'), floor_policy=floor_policy,
                          verify=verify, cache_dir=(Path(work) if work else target)/'data/cache')
        import subprocess
        subprocess.run(['node', str(Path(__file__).with_name('export_shared_geometry.cjs')),
                        str(stage), '--render-only'], check=True)
        target.mkdir(parents=True, exist_ok=True)
        from step5_connect_support.run_fast_local_bodies import preserve_previous
        data = preserve_previous(target)
        report = json.loads((stage/'report.json').read_text())
        report['artifacts']['../shape.obj'] = report['artifacts'].pop('fixture.obj')
        report['artifacts']['../overview.png'] = S.I.sha256(stage/'overview.png')
        S.I.save(stage/'report.json', report)
        for name in artifacts+['overview.png', 'fixture.png']:
            destination = target/'shape.obj' if name == 'fixture.obj' else target/name if name == 'overview.png' else data/name
            if (stage/name).resolve() != destination.resolve():
                destination.write_bytes((stage/name).read_bytes())
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameters', type=Path, nargs='?', default=Path(__file__).with_name('local_body_case.json'), help='Case parameters JSON: feet_xy_m and optional assignments')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--work', type=Path, help='Persistent, input-hashed construction cache')
    parser.add_argument('--floor-policy', choices=('nearest', 'opposite'), default='nearest')
    parser.add_argument('--verify', action='store_true', help='Opt in to the full final geometry/load checks')
    args = parser.parse_args(); run(args.parameters, args.output, args.work, args.floor_policy, args.verify)
