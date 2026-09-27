"""B/pose1+3: local head bodies with disconnected landing faces.

This is a concrete specimen, not a global geometry optimizer. Each original
head grows toward the other task's floor; additional bearing terminals reshape
nearby bodies by direct oblique lofts. No floor perimeter is constructed.
"""
import argparse
import json
from pathlib import Path
import tempfile

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


def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def write_mesh(path, value):
    mesh = S.unpack(value)
    mesh.export(path, file_type='obj', digits=17, include_normals=False)
    return mesh


def build(work, feet, assignments=None):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    tasks, groups, heads, directions, source, schedule, paths = S.inputs()
    bases = np.asarray([np.eye(3), S.ROTATION])
    offsets = np.asarray([np.zeros(3), -S.TRANSLATION@S.ROTATION])
    cache_key = dict(inputs=S.I.hashes(paths), rotations=bases.tolist(),
                     offsets=offsets.tolist(), directions=np.asarray(directions).tolist(),
                     relief_m=S.RELIEF, sweep_length_m=S.SWEEP_LENGTH,
                     sweep_code=S.I.sha256(Path(S.swept_solid.__code__.co_filename)))
    cache_file = work/'sweep_cache.json'
    reuse = cache_file.exists() and json.loads(cache_file.read_text()) == cache_key
    sweeps = []
    for k, p in enumerate(tasks):
        path = work/f'sweep{k}.npz'
        if reuse and path.exists():
            data = np.load(path); sweep = trimesh.Trimesh(data['v'], data['f'], process=False)
        else:
            obj = trimesh.Trimesh(p.domain.mesh.vertices@bases[k]+offsets[k], p.domain.mesh.faces, process=False)
            sweep = S.swept_solid(obj, -S.SWEEP_LENGTH*directions[k]@bases[k])
            np.savez_compressed(path, v=sweep.vertices, f=sweep.faces)
        sweeps.append(sweep)
    forbidden_file = work/'forbidden.npz'
    if reuse and forbidden_file.exists():
        data = np.load(forbidden_file)
        forbidden = S.solid(trimesh.Trimesh(data['v'], data['f'], process=False))
    else:
        box = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        forbidden = union([S.solid(m).minkowski_sum(box) for m in sweeps])
        m = S.unpack(forbidden); np.savez_compressed(forbidden_file, v=m.vertices, f=m.faces)
    cache_file.write_text(json.dumps(cache_key, indent=2))
    print('Complete sweeps ready', flush=True)

    def carve(value):
        for basis, offset in zip(bases, offsets):
            normal = basis[2]
            value = value.trim_by_plane(normal.tolist(), float(normal@offset/S.SCALE))
        return value-forbidden

    def connected_piece(value, required):
        for component in sorted(value.decompose(), key=lambda q:-q.volume()):
            if all(abs(float((item-component).volume()))*S.SCALE**3 <= 8e-14 for item in required):
                return component
        return None

    patches, bodies = [], []
    for k in range(2):
        for contact, cells in zip(groups[k], heads[k]):
            local = [v@bases[k]+offsets[k] for v in cells]
            v = np.concatenate(local); v = v[ConvexHull(v).vertices]
            original = union([S.solid(G.hull_mesh(q)) for q in local])
            root = v+.008*directions[k]@bases[k]
            opposite = 1-k
            w = S.local_to_world(v, bases[opposite], offsets[opposite])
            xy = w[:, :2]; center = xy.mean(0); xy = center+.7*(xy-center)
            pad = np.c_[xy, np.zeros(len(xy))]@bases[opposite]+offsets[opposite]
            body = carve(S.solid(G.hull_mesh(np.vstack([v, root, pad]))))+original
            if len(body.decompose()) != 1:
                raise RuntimeError('An initial local head body is disconnected')
            patches.append(dict(pose=k, id=contact['candidate_id'], v=v, root=root, original=original))
            bodies.append(body)

    attachments = []
    for k, polygons in enumerate(feet):
        for j, polygon in enumerate(polygons):
            xy = np.asarray(polygon)
            pad = np.c_[xy, np.zeros(len(xy))]@bases[k]+offsets[k]
            terminal = np.vstack([pad, pad+.003*bases[k][2]])
            actual_terminal = carve(S.solid(G.hull_mesh(terminal)))
            choices = []
            for index, patch in enumerate(patches):
                if assignments is not None and index != assignments[k][j]:
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
                raise RuntimeError(f'No direct connected local loft reaches floor {k}, terminal {j}')
            _, index, bodies[index], distance, added_cm3 = min(choices, key=lambda x: x[0])
            attachments.append(dict(floor_pose=tasks[k].pose, terminal=j, head_body=index,
                head_pose=tasks[patches[index]['pose']].pose, cross_pose_endpoint=patches[index]['pose'] != k,
                polygon_xy_m=xy.tolist(), minimum_head_vertex_to_terminal_center_m=distance,
                added_volume_cm3=added_cm3, direct_local_loft=True))
            print('Local terminal', attachments[-1], flush=True)

    full = union(bodies)
    connectors, connections = [], []
    bead = trimesh.creation.icosphere(subdivisions=1, radius=.004).vertices
    up = bases[0][2]+bases[1][2]; up /= np.linalg.norm(up)
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
        for lift in (0., .008, .016, .024):
            for distance, i, j, a, b in sorted(trials, key=lambda x: x[0]):
                if distance > .09:
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
            raise RuntimeError('Local bodies need an unresolved connection')
    mesh = S.unpack(full)
    record = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
        consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    record['one_solid'] = bool(record['component_count'] == 1 and record['watertight'] and record['consistently_wound'] and mesh.volume > 0)
    if not record['one_solid']:
        raise RuntimeError('Local bodies did not form one closed solid')
    checks, certificate = S.verify(tasks, groups, heads, directions, bases, offsets, mesh, sweeps)
    np.savez_compressed(work/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces, rotations=bases, local_offsets_m=offsets)
    np.savez_compressed(work/'equilibrium.npz', **certificate)
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
            candidate_id=patch['id'], volume_cm3=float(m.volume*1e6), file=f'body{i}.obj'))
    if connectors:
        write_mesh(work/'bridges.obj', union(connectors))
    design = dict(feet_xy_m=feet, attachments=attachments, local_bodies=body_records, connections=connections)
    S.I.save(work/'design.json', design)
    artifacts = ['fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'equilibrium.npz', 'design.json']
    artifacts += [r['file'] for r in body_records]+(['bridges.obj'] if connectors else [])
    code = [Path(__file__), Path(S.__file__), Path(S.swept_solid.__code__.co_filename),
            Path(S.Q.__file__), Path(S.W.__file__), Path(G.__file__), Path(S.SOL.__file__),
            Path(S.FLOOR.__file__), Path(S.bearing_rays.__code__.co_filename),
            Path(S.surface_distances.__code__.co_filename), Path(export_viewer.__code__.co_filename),
            Path(__file__).with_name('shared_geometry_viewer.html'), Path(__file__).with_name('export_shared_geometry.cjs')]
    report = dict(object='B', poses=S.POSES, particle=9, complete=True,
        status='geometry_and_fixed_sample_equilibrium_passed', design='local_head_bodies_with_disconnected_landing_faces',
        same_rigid_solid_in_both_poses=True, contact_groups=5, contact_patch_count=6,
        shared_branch_definition='six original contact patches, including two distinct orange faces',
        original_task_poses_changed=False, original_active_contacts_preserved=True,
        support_mass_ignored=True, extra_loads_added=False, structural_strength_verified=False,
        dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6), solid=record, checks=checks,
        source_schedule=str((source/'schedule.json').relative_to(S.ROOT)),
        task_fixture_transforms=[dict(pose=p.pose, rotation=b.tolist(), translation_m=(-b@o).tolist()) for p,b,o in zip(tasks,bases,offsets)],
        body_design=dict(method='head-to-opposite-floor local lofts, reshaped by discrete bearing terminals, joined above ground',
            floor_perimeter_constructed=False, local_bodies=body_records, attachments=attachments, connections=connections,
            optimality_claim=False, local_role_reuse_verified=False,
            interpretation='Geometric construction; endpoint usefulness and direct material witnesses require separate audit'),
        presentation_description='从接触头附近向两种落地面延伸的局部身体，脚面分散，身体在地面上方连接。彩色保留头及附近，灰白色显示新增身体；橙色仍有两块不同接触面。',
        exports=dict(obj_units='m', stl_units='mm', stl_encoding='ASCII', reloaded_stl_watertight=True),
        artifacts={name:S.I.sha256(work/name) for name in artifacts},
        provenance=dict(inputs=S.I.hashes(paths), code=S.I.hashes(code)))
    S.I.save(work/'report.json', report)
    colors = {schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in schedule['selected_ids'] if i not in colors], ['#ac7098','#7196c0','#50a59b','#77a76a']))
    visual = [(c['candidate_id'], G.hull_mesh(np.concatenate([v@b+o for v in cs])))
              for gs,hs,b,o in zip(groups,heads,bases,offsets) for c,cs in zip(gs,hs)]
    export_viewer(work, mesh, visual, report, tasks, bases, offsets, colors)
    html=(work/'index.html').read_text()
    data,_=json.JSONDecoder().raw_decode(html.split('const DATA=',1)[1])
    labels=['橙色头 · Pose 1','紫色头 · Pose 1','蓝色头 · Pose 1',
            '橙色头 · Pose 3','青色头 · Pose 3','绿色头 · Pose 3']
    data['local_bodies']=[dict(label=labels[i],id=p['id'],head_pose=tasks[p['pose']].pose,
        **pack(S.unpack(body))) for i,(p,body) in enumerate(zip(patches,bodies))]
    write_viewer(work,data)
    print('VERIFIED', record, report['volume_cm3'], flush=True)
    return artifacts+['report.json', 'index.html']


def run(parameters, output=None, work=None):
    parameters = json.loads(Path(parameters).read_text())
    target = Path(output) if output else S.pair_folder('B', S.POSES, 'step3_scheculer').parent/'step5'
    with tempfile.TemporaryDirectory(prefix='cadgrasp_local_bodies_') as temporary:
        stage = Path(work) if work else Path(temporary)
        artifacts = build(stage, parameters['feet_xy_m'], parameters.get('assignments'))
        target.mkdir(parents=True, exist_ok=True)
        for name in artifacts:
            if stage.resolve() != target.resolve():
                (target/name).write_bytes((stage/name).read_bytes())
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameters', type=Path, nargs='?', default=Path(__file__).with_name('local_body_case.json'), help='Case parameters JSON: feet_xy_m and optional assignments')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--work', type=Path, help='Persistent, input-hashed construction cache')
    args = parser.parse_args(); run(args.parameters, args.output, args.work)
