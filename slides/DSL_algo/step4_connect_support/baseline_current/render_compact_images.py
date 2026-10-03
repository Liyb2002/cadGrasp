"""Publish one overview; construction is rendered by draw_growth_steps."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import boxed_support as F, clean_render as V, publish_compact as P
from step4_connect_support.baseline_current.run_boxed_batch import protected_hashes


def render(group, source_name='boxed_support'):
    out = group/'step4'
    source = out/'data'/source_name
    report = I.check_report(source/'report.json')
    assert report['passed'] and report.get('exact_export_format',report.get('export_roundtrip_bit_exact',False))
    case, _ = F.read_case(group, source)
    mesh = trimesh.load(source/'shape.obj', force='mesh', process=False)
    bases = np.asarray(report['placement']['bases'])
    offsets = np.asarray(report['placement']['offsets'])
    directions = np.asarray(report['placement']['directions'])
    backup = out/'data/history'/('before_growing_images' if source_name == 'growing_support' else 'before_compact_images')
    backup.mkdir(parents=True, exist_ok=True)
    for image in out.glob('*.png'):
        if not (backup/image.name).exists():
            old = source/'previous_public_overview.png' if image.name == 'overview.png' and source_name == 'boxed_support' else image
            shutil.copy2(old if old.exists() else image, backup/image.name)
    renderer = V.Renderer()
    heads = [(c['candidate_id'], trimesh.Trimesh(
        c['triangles_m'].reshape(-1, 3)@b+o,
        np.arange(c['triangles_m'].size//3).reshape(-1, 3), process=False))
        for row, b, o in zip(case.groups, bases, offsets) for c in row]
    colors = V.head_colors([ident for ident, _ in heads])
    support = [P.piece(mesh, '#c2c9c8')]+[P.piece(head, colors[ident], 3e-6) for ident, head in heads]
    pictures, records = [], []
    for task, b, o, direction in zip(case.tasks, bases, offsets, directions):
        translation = -b@o
        installed = (mesh.vertices-o)@b.T
        cloud = np.vstack([installed, task.domain.mesh.vertices])
        object_piece = P.CPU.piece(dict(v=task.domain.mesh.vertices, f=task.domain.mesh.faces),
                                   '#91b5cd', smooth=True)
        object_piece['opacity'] = .68
        # Look into the actual withdrawal opening so the workpiece is visible.
        view = -direction+np.array([0., 0., .65])
        camera = P.CPU.fit(cloud, view, 1., padding=1.25)
        parts = [P.CPU.placed(part, b, translation) for part in support]
        parts.append(P.CPU.placed(object_piece))
        picture = renderer.render(parts, camera, 1100)
        filename = task.pose+'.png'
        pictures.append(picture)
        records.append(dict(pose=task.pose, filename=filename, view_world=view.tolist()))
    view = (-directions[0]+np.array([0., 0., .65]))@bases[0]
    camera = P.CPU.fit(mesh.vertices, view, 1., padding=1.25)
    picture = renderer.render([P.CPU.placed(part) for part in support], camera, 1100)
    pictures.append(picture)
    canvas = Image.new('RGB', (1100*len(pictures), 1100), 'white')
    for index, picture in enumerate(pictures):
        canvas.paste(picture, (1100*index, 0))
    canvas.save(out/'overview.png')
    extras=out/'data/history/before_two_public_images'
    extras.mkdir(parents=True,exist_ok=True)
    for image in out.glob('*.png'):
        if image.name not in ('overview.png','construction_steps.png'):
            shutil.copy2(image,extras/image.name)
            image.unlink()
    artifacts = [out/'overview.png']
    I.save(out/'data/compact_visualization.json', dict(complete=True, poses=case.poses,
        source=source_name, geometry_changed=False,
        per_pose=records, overview_order=[r['filename'] for r in records]+['support.png'],
        support_color='#c2c9c8', object_color='#91b5cd', object_opacity=.68,
        provenance=dict(inputs=I.hashes([source/'report.json', source/'shape.obj']),
                        code=I.hashes([Path(__file__), Path(V.__file__), Path(P.__file__), Path(P.CPU.__file__)])),
        artifacts={'../'+p.name:I.sha256(p) for p in artifacts}))
    assert set(p.name for p in out.glob('*.png')) <= {'overview.png','construction_steps.png'}
    print('NEW IMAGES', group.name, len(artifacts), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['boxed_support', 'growing_support'], default='boxed_support')
    args=parser.parse_args()
    root = I.OUTPUTS/'B'
    protected = protected_hashes(root)
    groups = sorted(p.parent.parent.parent.parent for p in root.glob(f'pose*+*/step4/data/{args.source}/report.json')
                    if p.parents[3].name != 'pose1+3')
    for group in groups:
        render(group,args.source)
    assert protected == protected_hashes(root)
    print('ALL PUBLIC STEP4 PNGS UPDATED; Step3 and pose1+3 unchanged', flush=True)
