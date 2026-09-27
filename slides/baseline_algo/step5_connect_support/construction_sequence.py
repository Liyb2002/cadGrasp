"""Replay the accepted local-body construction for visualization only.

Reconstruct actual Boolean additions and compare their final union with the
certified fixture. Never rewrite geometry, equilibrium, or their reports.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support import build_coupled_saddle as S
from step5_connect_support.build_local_bodies import union
from step5_connect_support.build_shared_geometry import pack
from step2_local_support import geometry as G


def replay(out, work):
    out, work = Path(out), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    report = json.loads((out/'report.json').read_text())
    if not report['complete'] or report['design'] != 'local_head_bodies_with_disconnected_landing_faces':
        raise ValueError('A verified local-body specimen is required')
    for group in ('inputs', 'code'):
        S.I.check_hashes(report['provenance'][group])
    for name, digest in report['artifacts'].items():
        if S.I.sha256(out/name) != digest:
            raise ValueError(f'Changed physical artifact: {name}')
    original_report_hash = S.I.sha256(out/'report.json')
    design = json.loads((out/'design.json').read_text())
    html = (out/'index.html').read_text()
    view, _ = json.JSONDecoder().raw_decode(html.split('const DATA=', 1)[1])
    tasks, groups, heads, directions, _, _, paths = S.inputs()
    with np.load(out/'geometry.npz') as stored:
        bases, offsets = stored['rotations'], stored['local_offsets_m']
        certified = S.solid(trimesh.Trimesh(stored['vertices_m'], stored['faces'], process=False))

    # The generator's input-keyed sweep cache can be reused, but is optional.
    key = dict(inputs=S.I.hashes(paths), rotations=bases.tolist(), offsets=offsets.tolist(),
               directions=np.asarray(directions).tolist(), relief_m=S.RELIEF,
               sweep_length_m=S.SWEEP_LENGTH,
               sweep_code=S.I.sha256(Path(S.swept_solid.__code__.co_filename)))
    cache = work/'sweep_cache.json'
    reuse = cache.exists() and json.loads(cache.read_text()) == key
    if reuse and (work/'forbidden.npz').exists():
        with np.load(work/'forbidden.npz') as stored:
            forbidden = S.solid(trimesh.Trimesh(stored['v'], stored['f'], process=False))
    else:
        sweeps = []
        box = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        for k, task in enumerate(tasks):
            mesh = trimesh.Trimesh(task.domain.mesh.vertices@bases[k]+offsets[k], task.domain.mesh.faces, process=False)
            sweeps.append(S.solid(S.swept_solid(mesh, -S.SWEEP_LENGTH*directions[k]@bases[k])).minkowski_sum(box))
        forbidden = union(sweeps)
        mesh = S.unpack(forbidden)
        np.savez_compressed(work/'forbidden.npz', v=mesh.vertices, f=mesh.faces)
        cache.write_text(json.dumps(key, indent=2))

    def carve(value):
        for basis, offset in zip(bases, offsets):
            value = value.trim_by_plane(basis[2].tolist(), float(basis[2]@offset/S.SCALE))
        return value-forbidden

    patches, bodies = [], []
    names = ['橙色头 · Pose 1', '紫色头 · Pose 1', '蓝色头 · Pose 1',
             '橙色头 · Pose 3', '青色头 · Pose 3', '绿色头 · Pose 3']
    colors = {part['id']: part['color'] for part in view['parts']}
    originals = []
    for k in range(2):
        for contact, cells in zip(groups[k], heads[k]):
            local = [v@bases[k]+offsets[k] for v in cells]
            v = np.concatenate(local); v = v[ConvexHull(v).vertices]
            original = union([S.solid(G.hull_mesh(q)) for q in local])
            root = v+.008*directions[k]@bases[k]
            patches.append(dict(pose=k, id=contact['candidate_id'], v=v, root=root, original=original))
            originals.append(dict(color=colors[contact['candidate_id']], **pack(S.unpack(original))))
            bodies.append(original)
    current = union(bodies)
    stages = [dict(title='先看六块原始接触头', detail='同一件支架；两边同时显示同一块材料的位置。',
                   phase=0, duration=3., kind='heads')]
    differences = []

    def record(next_solid, title, detail, phase, duration, start, target, **metadata):
        nonlocal current
        removed = abs(float((current-next_solid).volume()))*S.SCALE**3
        if removed > 8e-14:
            raise ValueError('Construction replay unexpectedly removes previous material')
        delta = next_solid-current
        mesh = S.unpack(delta)
        axis = np.asarray(target)-np.asarray(start)
        axis /= np.linalg.norm(axis)
        projections = mesh.vertices@axis
        stages.append(dict(title=title, detail=detail, phase=phase, duration=duration,
            kind='addition', mesh=pack(mesh), axis=axis.tolist(),
            reveal_min=float(projections.min()), reveal_max=float(projections.max()),
            added_volume_cm3=float(delta.volume()*S.SCALE**3*1e6), **metadata))
        differences.append(delta)
        current = next_solid
        print(f'{len(stages)-1:02d} {title}', flush=True)

    for i, patch in enumerate(patches):
        opposite = 1-patch['pose']
        world = S.local_to_world(patch['v'], bases[opposite], offsets[opposite])
        xy = world[:, :2]; xy = xy.mean(0)+.7*(xy-xy.mean(0))
        pad = np.c_[xy, np.zeros(len(xy))]@bases[opposite]+offsets[opposite]
        bodies[i] = carve(S.solid(G.hull_mesh(np.vstack([patch['v'], patch['root'], pad]))))+patch['original']
        if len(bodies[i].decompose()) != 1:
            raise ValueError('Disconnected initial body in replay')
        target_pose = tasks[opposite].pose.replace('pose_', 'Pose ')
        record(union(bodies), f'{names[i]}：身体向另一地面展开',
               f'彩色头保留；灰白身体延伸到 {target_pose} 的地面。', 1, 2.6,
               patch['v'].mean(0), pad.mean(0), body=i,
               target_polygon=pad[ConvexHull(xy).vertices].tolist(), target_pose=opposite)

    for attachment in design['attachments']:
        k = S.POSES.index(attachment['floor_pose']); i = attachment['head_body']
        patch = patches[i]; xy = np.asarray(attachment['polygon_xy_m'])
        pad = np.c_[xy, np.zeros(len(xy))]@bases[k]+offsets[k]
        terminal = np.vstack([pad, pad+.003*bases[k][2]])
        actual_terminal = carve(S.solid(G.hull_mesh(terminal)))
        addition = carve(S.solid(G.hull_mesh(np.vstack([patch['v'], patch['root'], terminal]))))
        choices = sorted((bodies[i]+addition).decompose(), key=lambda q:-q.volume())
        proposed = next((part for part in choices if all(abs(float((item-part).volume()))*S.SCALE**3 <= 8e-14
                         for item in (bodies[i], actual_terminal))), None)
        if proposed is None:
            raise ValueError('A recorded foot no longer connects directly')
        bodies[i] = proposed
        label = attachment['floor_pose'].replace('pose_', 'Pose ')
        record(union(bodies), f'{label}：补充第 {attachment["terminal"]+1} 处脚端',
               f'从{names[i]}的已有身体直接展开，补足落脚范围。', 2, 2.3,
               patch['v'].mean(0), pad.mean(0), body=i, target_polygon=pad.tolist(), target_pose=k)

    body_errors = []
    for i, body in enumerate(bodies):
        saved = S.solid(trimesh.load(out/f'body{i}.obj', process=False))
        error = (abs(float((body-saved).volume()))+abs(float((saved-body).volume())))*S.SCALE**3
        body_errors.append(error)
        if error > 8e-14:
            raise ValueError(f'Replayed body {i} differs from the accepted body: {error}')

    for i, connection in enumerate(design['connections']):
        route = np.asarray(connection['path_m'])
        bead = trimesh.creation.icosphere(subdivisions=1, radius=connection['radius_m']).vertices
        bridge = carve(union([S.solid(G.hull_mesh(np.vstack([a+bead, b+bead]))) for a,b in zip(route[:-1], route[1:])]))
        record(current+bridge, f'添加第 {i+1} 处短连接',
               '在地面上方连接局部身体，让支架成为一件。', 3, 1.5, route[0], route[-1])

    missing = abs(float((certified-current).volume()))*S.SCALE**3
    extra = abs(float((current-certified).volume()))*S.SCALE**3
    if max(missing, extra) > 8e-14:
        raise ValueError(f'Final replay does not match accepted geometry: {missing}, {extra}')
    reconstruction = union([union([p['original'] for p in patches])]+differences)
    display_error = (abs(float((reconstruction-certified).volume()))+abs(float((certified-reconstruction).volume())))*S.SCALE**3
    if display_error > 8e-14:
        raise ValueError('Displayed additions do not reconstruct the accepted fixture')
    stages.append(dict(title='完成：同一件支架，两种摆放',
        detail='局部身体兼任另一姿态的脚；彩色保留头部，灰白显示身体。', phase=4, duration=6., kind='complete'))
    start = 0.
    for stage in stages:
        stage['start'] = start; start += stage['duration']
    data = dict(poses=view['poses'], fixture=view['fixture'], parts=view['parts'], head_regions=view['head_regions'],
                heads=originals, stages=stages, duration=start, dimensions_mm=report['dimensions_mm'])
    template = Path(__file__).with_name('construction_viewer.html').read_text()
    three = (S.ROOT/'slides/reuse/vendor/three.min.js').read_text()
    (out/'construction.html').write_text(template.replace('__THREE__', three).replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    metadata = dict(schema='accepted_local_body_construction_replay_v1', complete=True,
        original_report_sha256=original_report_hash, poses=S.POSES, duration_seconds=start,
        body_symmetric_difference_m3=body_errors, final_missing_volume_m3=missing, final_extra_volume_m3=extra,
        displayed_union_symmetric_difference_m3=display_error, volume_tolerance_m3=8e-14,
        physical_artifacts_unchanged=True, mechanics_rerun=False,
        animation='Progressive reveal of actual Boolean additions; intermediate frames are construction illustrations, not assembly motions or feasible fixtures',
        stages=[{k:v for k,v in s.items() if k not in ('mesh','target_polygon')} for s in stages],
        provenance=dict(code=S.I.hashes([Path(__file__),Path(__file__).with_name('construction_viewer.html')]),
                        inputs={str(out/'report.json'):original_report_hash, str(out/'design.json'):S.I.sha256(out/'design.json')}))
    S.I.save(out/'construction_sequence.json', metadata)
    for name, digest in report['artifacts'].items():
        assert S.I.sha256(out/name) == digest, name
    assert S.I.sha256(out/'report.json') == original_report_hash
    print(f'Replay matches accepted fixture. Duration {start:.1f}s. Missing/extra: {missing:.3g}/{extra:.3g} m³', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=S.ROOT/'slides/baseline_algo/output/B/pose1+3/step5')
    parser.add_argument('--work', type=Path, help='Optional input-keyed construction cache')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='cadgrasp_construction_') as temporary:
        replay(args.output, args.work or Path(temporary))
