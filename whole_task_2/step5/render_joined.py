"""Step5 figures, no text.

output/<object>/step5/joined.png: one panel per slot, the whole support with the bunny in that slot
(pink: the slot's task areas), and last the support alone.  Same camera for every panel.

output/<object>/step5/use.png: every task at the station, the support turned on the table and its
slot slid under the needle; the KUKA holds the needle on the task's central certified face at the
joints Step1/Step4 certified there.  The arm rises and falls with the object, so each picture is
drawn at the certified height: the support and its table are raised by the same amount.
"""
import math
import numpy as np
from PIL import Image

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT
from whole_task_2.step1.reach_render import contact
from whole_task_2.step1.reach_video import STATION_VIEW, Scene, work_arm
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step3.render_tool_membranes import camera, image
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step4.seats import read_mesh
from .joined import FOLDER, model_from_disk


def overview(model, support, size=(900, 760), view=(.55, -1., .55)):
    """One image per slot (the support and the bunny in that slot), then the support alone."""
    scene = Scene(support, model.setup, *size)
    try:
        g = scene.r; blue = np.tile(BLUE, (len(support.faces), 1)); scene.view = np.asarray(view)/np.linalg.norm(view)
        scene.fit(np.vstack([support.vertices, *[moved_mesh(model.source, X).vertices for X in model.X]]), 1.06)
        images = []
        for s, members in enumerate(model.seats):
            bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
            colors = np.tile(g.GRAY, (len(model.source.faces), 1))
            for k, seat in enumerate(model.seat_of):
                if seat == s: colors[model.records[k]['working_face_ids']] = g.PINK
            bunny.paint(colors); images.append(image(scene, blue, [], None, [(bunny, model.X[s])]))
        return images+[image(scene, blue, [], None, [])]
    finally: scene.close()


def use(model, q, support, k, size=620):
    """Task k at the station, drawn at its certified height."""
    r = model.records[k]; P = q.poses[k]; T = np.asarray(r['pose'])@np.linalg.inv(P)      # lift to the certified height
    world = moved_mesh(support, T@np.linalg.inv(q.material[k])); pose = T@P
    faces = np.asarray(r['certified_face_ids']); f = contact(model.source, faces, pose); j = int(np.flatnonzero(faces == f)[0])
    tip = model.source.triangles_center[f]@pose[:3, :3].T+pose[:3, 3]; axis = pose[:3, :3]@r['axes'][j]
    links = matrices(work_arm(model.setup).forward(r['joints'][j][3], geometry=True)[2])
    scene = Scene(world, model.setup, size, size, arm=True)
    try:
        g = scene.r
        bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
        colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[r['working_face_ids']] = g.PINK; bunny.paint(colors)
        blue = np.tile(BLUE, (len(world.faces), 1)); scene.view = STATION_VIEW
        top = tip-axis*(model.setup['needle_length_m']+.07)
        scene.fit(np.vstack([world.vertices, moved_mesh(model.source, pose).vertices, tip, top]), 1.12)
        return image(scene, blue, [(tip, axis, False)], links, [(bunny, pose)])
    finally: scene.close()


def render(object_name='B', output_root=DEFAULT_OUTPUT, size=620, gap=40):
    c, model, seats, q, report = model_from_disk(object_name)
    support = read_mesh(ROOT/'whole_task_2/data'/object_name/FOLDER/'support.npz')
    out = output_root/object_name/'step5'; out.mkdir(parents=True, exist_ok=True)
    panels = overview(model, support); w, h = panels[0].size
    canvas = Image.new('RGB', (len(panels)*w, h), 'white')
    for k, panel in enumerate(panels): canvas.paste(panel, (k*w, 0))
    canvas.save(out/'joined.png')
    n = len(model.records); columns = 3
    sheet = Image.new('RGB', (columns*(size+gap)+gap, math.ceil(n/columns)*(size+gap)+gap), 'white')
    for k in range(n):
        sheet.paste(use(model, q, support, k, size), (gap+(k % columns)*(size+gap), gap+(k//columns)*(size+gap)))
    sheet.save(out/'use.png')
    print('STEP5 FIGURES', out/'joined.png', out/'use.png', flush=True)
    return out


if __name__ == '__main__':
    render()
