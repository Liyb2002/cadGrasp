"""pool.png of a big task's candidate pool (step3/big_init.py), no text: one panel per task, as the robot sees it --
the object at its placement, its work area pink, its support blue when it has one, the needle and head on its
middle face along the certified axis.  The camera is the same for every panel, on the needle's side.
"""
import json
from pathlib import Path

import numpy as np

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_render import contact, picture, tiles
from whole_task_2.step1.reach_video import Scene, tool_frame
from .render_independent import BLUE


def panel(scene, mesh, z, name, s):
    """One task, shifted by the work point (the scene's lights are about the origin)."""
    g = scene.r; point = np.asarray(s['work_point_m']); P = z[f'{name}_pose'].copy(); P[:3, 3] -= point
    faces = z[f'{name}_face_ids']; f = contact(mesh, faces, P); k = int(np.flatnonzero(faces == f)[0])
    tip = mesh.triangles_center[f]@P[:3, :3].T+P[:3, 3]; axis = P[:3, :3]@z[f'{name}_axes'][k]
    colors = np.tile(g.GRAY, (len(mesh.faces), 1)); colors[faces] = g.PINK
    shapes = [(part, tool_frame(tip, axis)) for part in scene.tools[False]]; points = [mesh.vertices@P[:3, :3].T+P[:3, 3]]
    if f'{name}_vertices' in z:
        v = z[f'{name}_vertices']-point; points.append(v)
        shapes.append((g.Shape(scene.ctx, scene.shader, v, z[f'{name}_faces'], BLUE, metallic=.05, roughness=.55), np.eye(4)))
    scene.fit(np.vstack(points+[[tip, tip-axis*(s['needle_length_m']+.07)]]), 1.08)
    return picture(scene, P, colors, [], fit=False, shapes=shapes)


def render(run, out, size=420, columns=10):
    folder = ROOT/'whole_task_2/data'/run/'step3_init'; report = json.loads((folder/'report.json').read_text())
    mesh = load_shape(ROOT/'objects'/report['object']/'mesh.stl'); s = report['setup']; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    a = np.asarray(s['working_direction_world'], float); h = -a.copy(); h[2] = 0; h /= np.linalg.norm(h)
    view = h+.6*np.cross([0., 0., 1.], h)+.55*np.array([0., 0., 1.])
    with np.load(folder/'initialization.npz') as z: z = {k: z[k] for k in z.files}
    scene = Scene(mesh, s, size, size)
    try:
        scene.view = view/np.linalg.norm(view)
        tiles([panel(scene, mesh, z, row['id'], s) for row in report['tasks']], columns).save(out/'pool.png')
    finally:
        scene.close()
    print('FIGURE', out/'pool.png', flush=True)
