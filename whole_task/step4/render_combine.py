"""Step4 figures from step4/combine.py's chosen support, no text.

output/<object>/step4/merges.png: one row per merge, in order: parent A, parent B and the merged support,
each standing on its first pose with a bunny in every seat (pink: that seat's task areas).  Juxtaposition
puts A and B together; Rotational reuse keeps A's material only and puts B's tasks into A's seats.

output/<object>/step4/support.png: the final support with every bunny, alone from two sides, and standing
on its plate when it has one pose.
"""
import json
import numpy as np
import trimesh
from PIL import Image

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT
from whole_task.operations.geometry import moved_mesh
from .seats import context, read_mesh
from .render_merge import upright, panel
from .combine import FOLDER


def seated(c, slots):
    """One slot per seat, its members every task in that seat (for the pink)."""
    out = []
    for s in slots:
        for t in out:
            if np.allclose(np.asarray(t['X']), np.asarray(s['X']), atol=1e-9): t['members'] = t['members']+s['members']; break
        else: out.append(dict(s, members=list(s['members'])))
    return out


def standing(c, mesh, slots, size, view=(.55, -1., .55)):
    T = upright(c, slots); seats = seated(c, slots)
    points = np.vstack([moved_mesh(mesh, T).vertices, *[moved_mesh(c.base.source, T@np.asarray(s['X'])).vertices for s in seats]])
    return panel(c, mesh, seats, T, points, size, view)


def render(object_name='B', output_root=DEFAULT_OUTPUT, size=(800, 680), group=None):
    c = context(object_name)
    if group is None: group = json.loads((ROOT/'whole_task/data'/object_name/FOLDER/'report.json').read_text())['group']
    out = output_root/object_name/'step4'; out.mkdir(parents=True, exist_ok=True); w, h = size
    rows = [[standing(c, read_mesh(ROOT/f), s, size) for f, s in zip((*step['parent_files'], step['file']), (*step['parent_slots'], step['slots']))]
            for step in group['chain']]
    canvas = Image.new('RGB', (3*w, len(rows)*h), 'white')
    for k, row in enumerate(rows):
        for j, p in enumerate(row): canvas.paste(p, (j*w, k*h))
    canvas.save(out/'merges.png')
    mesh = read_mesh(ROOT/group['file']); T = upright(c, group['slots']); points = moved_mesh(mesh, T).vertices
    sides = [standing(c, mesh, group['slots'], size), panel(c, mesh, [], T, points, size), panel(c, mesh, [], T, points, size, (-.55, 1., .55))]
    with np.load(ROOT/group['file']) as z:
        if 'standing_vertices' in z.files:
            stand = trimesh.Trimesh(z['standing_vertices'], z['standing_faces'], process=False); S = np.asarray(z['table_from_support'])
            seats = [dict(s, X=(S@np.asarray(s['X'])).tolist()) for s in seated(c, group['slots'])]
            sides.append(panel(c, stand, seats, np.eye(4), np.vstack([stand.vertices, *[
                moved_mesh(c.base.source, np.asarray(s['X'])).vertices for s in seats]]), size))
    sheet = Image.new('RGB', (len(sides)*w, h), 'white')
    for j, p in enumerate(sides): sheet.paste(p, (j*w, 0))
    sheet.save(out/'support.png')
    print('STEP4 FIGURES', out/'merges.png', out/'support.png', flush=True)
    return out


if __name__ == '__main__':
    render()
