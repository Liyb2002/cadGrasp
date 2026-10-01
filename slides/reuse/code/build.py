"""Import the current B/pose1+3 fixture and frozen task poses for presentation.

No geometry synthesis or new load audit: shape.obj is copied byte-for-byte.
The existing six physical contact patches keep their five ID colors.
"""
from pathlib import Path
import json
import sys

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'slides/sys_floor'))
import steps as S


def pack(mesh):
    return dict(v=np.round(mesh.vertices, 10).ravel().tolist(),
                f=mesh.faces.ravel().tolist())


def run():
    case, placement, _, _ = S.recipe(S.SOURCE)
    source = S.SOURCE / 'shape.obj'
    report_path = S.SOURCE / 'data/report.json'
    report = json.loads(report_path.read_text())
    viewer_path = S.SOURCE / 'data/index.html'
    viewer, _ = json.JSONDecoder().raw_decode(viewer_path.read_text().split('const DATA=', 1)[1])
    paths = set(case.paths) | {source, report_path, viewer_path}
    hashes = {str(p.relative_to(ROOT)): S.digest(p) for p in sorted(paths)}
    mesh = trimesh.load(source, force='mesh', process=False)
    assert mesh.is_watertight and report['solid']['component_count'] == 1
    np.testing.assert_allclose(mesh.volume * 1e6, report['volume_cm3'], atol=1e-6)
    colors = {p['id']: p['color'] for p in viewer['parts']}
    overlays = []
    for region in viewer['head_regions']:
        triangles = S.clip_box(mesh.triangles, np.asarray(region['low']), np.asarray(region['high']))
        overlays.append(dict(id=region['id'], color=colors[region['id']],
                             v=triangles.reshape(-1).tolist(),
                             f=np.arange(len(triangles)*3).tolist()))
    poses = []
    for k, task in enumerate(case.tasks):
        basis, offset = placement['bases'][k], placement['offsets'][k]
        rotation, translation = basis, -basis @ offset
        world_fixture = mesh.vertices @ rotation.T + translation
        np.testing.assert_allclose(world_fixture, (mesh.vertices-offset) @ basis.T, atol=1e-12)
        assert world_fixture[:, 2].min() >= -1e-8
        poses.append(dict(name=task.pose.replace('_', ' ').title(), source=f'B / {task.pose}',
                          fixtureR=rotation.tolist(), fixtureT=translation.tolist(),
                          object=pack(task.domain.mesh), work=task.domain.work_ids.tolist(),
                          objectR=np.asarray(task.domain.data['frame']['T_world_mesh'])[:3, :3].tolist(),
                          withdrawalDirection=(-placement['directions'][k]).tolist()))
    data = dict(schema='saved_pose1_3_fixture_v1', colors=list(colors.values()),
                fixture=pack(mesh), headOverlays=overlays, poses=poses,
                fixtureColor='#dce2e2', sourceShape=str(source.relative_to(ROOT)),
                sourceShapeSha256=hashes[str(source.relative_to(ROOT))],
                scope='Saved Step5 geometry and task placement; robot motion is a presentation.',
                duration=0, robot=[], motion=[], storyTimes=[])
    (OUT / 'fixture.obj').write_bytes(source.read_bytes())
    (OUT / 'data.js').write_text('window.REUSE_DATA=' + json.dumps(data, separators=(',', ':')) + ';\n')
    manifest = dict(source_shape=str(source.relative_to(ROOT)), sources=hashes,
                    poses=case.poses, source_obj_copied_byte_for_byte=True,
                    fixture_sha256=S.digest(OUT / 'fixture.obj'),
                    volume_cm3=float(mesh.volume*1e6), watertight=bool(mesh.is_watertight),
                    connected_components=report['solid']['component_count'],
                    contact_patch_count=6, contact_id_count=5,
                    fixture_geometry_rebuilt=False, final_audit_rerun=False,
                    source_verification=report['verification'],
                    display='Neutral body, fixed original head colors with the saved 4 mm collar')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for relative, digest in hashes.items():
        assert S.digest(ROOT / relative) == digest
    for stale in ('layout.json', 'load_check.json'):
        (OUT / stale).unlink(missing_ok=True)
    print('Imported exact pose1+3 shape:', manifest['volume_cm3'], 'cm3; two poses.', flush=True)


if __name__ == '__main__':
    run()
