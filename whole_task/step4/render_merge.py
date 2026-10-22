"""Step4 figures from step4/merge.py's saved chain, no text.

output/<object>/step4/merges.png: one row per merge, in order.  Left, parent A with a bunny in each of
its slots; middle, parent B moved by the merge's rigid motion into A's frame, its bunnies with it;
right, the merged support with every bunny.  Pink: the slot's task areas.  One camera per row.

output/<object>/step4/support.png: the final merged support alone, from three sides.

Drawn in the table frame of the support's first pose (that face down), the camera a little above.
"""
import numpy as np
from PIL import Image

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT
from whole_task.step1.reach_video import Scene
from whole_task.operations.geometry import moved_mesh
from whole_task.step3.render_tool_membranes import image
from whole_task.step3.render_independent import BLUE
from .seats import context, read_mesh
from .merge import FOLDER

VIEW = (.55, -1., .55)


def upright(c, slots):
    """The rotation putting the support's first pose down: (task pose) X^-1 for its first slot."""
    s = slots[0]; T = np.eye(4)
    T[:3, :3] = np.asarray(c.record(s['members'][0], s['angles'][0])[0]['pose'])[:3, :3]@np.asarray(s['X'])[:3, :3].T
    return T


def panel(c, mesh, slots, T, points, size, view=VIEW):
    """mesh and a bunny in every slot, everything moved by T, the camera fitted to points."""
    scene = Scene(moved_mesh(mesh, T), c.base.setup, *size)
    try:
        g = scene.r; scene.view = np.asarray(view)/np.linalg.norm(view); scene.fit(points, 1.06); shapes = []
        for s in slots:
            bunny = g.Shape(scene.ctx, scene.shader, c.base.source.vertices, c.base.source.faces, g.GRAY, dynamic=True,
                            metallic=.03, roughness=.63)
            colors = np.tile(g.GRAY, (len(c.base.source.faces), 1))
            for i in s['members']: colors[c.base.records[i]['working_face_ids']] = g.PINK
            bunny.paint(colors); shapes.append((bunny, T@np.asarray(s['X'])))
        return image(scene, np.tile(BLUE, (len(mesh.faces), 1)), [], None, shapes)
    finally: scene.close()


def placed(c, slots, T):
    return [moved_mesh(c.base.source, T@np.asarray(s['X'])).vertices for s in slots]


def merges(c, chain, size):
    rows = []
    for step in chain:
        A, B = step['parent_slots']; slots = step['slots']; n = len(A)
        C = np.asarray(slots[n]['X'])@np.linalg.inv(np.asarray(B[0]['X']))       # B's motion into A's frame
        movedB = [dict(s, X=(C@np.asarray(s['X'])).tolist()) for s in B]
        T = upright(c, slots); meshes = [read_mesh(ROOT/f) for f in (*step['parent_files'], step['file'])]
        meshes[1] = moved_mesh(meshes[1], C)
        points = np.vstack([*(moved_mesh(m, T).vertices for m in meshes), *placed(c, slots, T)])
        rows.append([panel(c, m, s, T, points, size) for m, s in zip(meshes, (A, movedB, slots))])
    return rows


def render(object_name='B', output_root=DEFAULT_OUTPUT, size=(800, 680)):
    import json
    c = context(object_name); report = json.loads((ROOT/'whole_task/data'/object_name/FOLDER/'report.json').read_text())
    group = report['group']; out = output_root/object_name/'step4'; out.mkdir(parents=True, exist_ok=True)
    rows = merges(c, group['chain'], size); w, h = size
    canvas = Image.new('RGB', (3*w, len(rows)*h), 'white')
    for k, row in enumerate(rows):
        for j, p in enumerate(row): canvas.paste(p, (j*w, k*h))
    canvas.save(out/'merges.png')
    mesh = read_mesh(ROOT/group['file']); T = upright(c, group['slots']); points = moved_mesh(mesh, T).vertices
    views = [VIEW, (-.55, 1., .55), (1., .2, .9)]
    sides = [panel(c, mesh, [], T, points, size, v) for v in views]
    sheet = Image.new('RGB', (len(sides)*w, h), 'white')
    for j, p in enumerate(sides): sheet.paste(p, (j*w, 0))
    sheet.save(out/'support.png')
    print('STEP4 FIGURES', out/'merges.png', out/'support.png', flush=True)
    return out


if __name__ == '__main__':
    render()
