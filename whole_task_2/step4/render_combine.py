"""Step4 figures from step4/combine.py's chosen chain, no text.

output/<object>/step4/merges.png: one row per merge, in order.  Left, parent A with a bunny in each of
its slots; middle, parent B moved by the merge's rigid motion into A's frame, its bunnies with it;
right, the merged support with every bunny.  Pink: the slot's task area.  One camera per row.

output/<object>/step4/support.png: the final support with every bunny, then alone from two sides,
then standing on its plate when it has one pose.

Drawn in the table frame of the support's first pose (that face down), the camera a little above.
"""
import json
import numpy as np
import trimesh
from PIL import Image

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT
from whole_task_2.operations.geometry import moved_mesh
from .seats import context, read_mesh
from .render_merge import VIEW, upright, panel, merges
from .combine import FOLDER


def render(object_name='B', output_root=DEFAULT_OUTPUT, size=(800, 680)):
    c = context(object_name); report = json.loads((ROOT/'whole_task_2/data'/object_name/FOLDER/'report.json').read_text())
    group = report['group']; out = output_root/object_name/'step4'; out.mkdir(parents=True, exist_ok=True)
    rows = merges(c, group['chain'], size); w, h = size
    canvas = Image.new('RGB', (3*w, len(rows)*h), 'white')
    for k, row in enumerate(rows):
        for j, p in enumerate(row): canvas.paste(p, (j*w, k*h))
    canvas.save(out/'merges.png')
    mesh = read_mesh(ROOT/group['file']); T = upright(c, group['slots'])
    points = np.vstack([moved_mesh(mesh, T).vertices, *[moved_mesh(c.base.source, T@np.asarray(s['X'])).vertices for s in group['slots']]])
    sides = [panel(c, mesh, group['slots'], T, points, size), panel(c, mesh, [], T, points, size),
             panel(c, mesh, [], T, points, size, (-.55, 1., .55))]
    with np.load(ROOT/group['file']) as z:
        if 'standing_vertices' in z.files:
            standing = trimesh.Trimesh(z['standing_vertices'], z['standing_faces'], process=False)
            S = np.asarray(z['table_from_support'])
            slots = [dict(s, X=(S@np.asarray(s['X'])).tolist()) for s in group['slots']]
            sides.append(panel(c, standing, slots, np.eye(4), np.vstack([standing.vertices, *[
                moved_mesh(c.base.source, np.asarray(s['X'])).vertices for s in slots]]), size))
    sheet = Image.new('RGB', (len(sides)*w, h), 'white')
    for j, p in enumerate(sides): sheet.paste(p, (j*w, 0))
    sheet.save(out/'support.png')
    print('STEP4 FIGURES', out/'merges.png', out/'support.png', flush=True)
    return out


if __name__ == '__main__':
    render()
