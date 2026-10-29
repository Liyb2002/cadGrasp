"""Step4's combine, one task per step, from one sample's saved final chain (default B+03_kuka sample 0,
one Rotational reuse, then four Juxtapositions); no text.

process.png, left to right: the first task's support, then the group after each join (a join's group
before it is the previous column).  Top row with a bunny in every seat (pink: the seat's task areas),
bottom row the supports alone.  One camera for every panel, in the table frame of the first task's pose.

The camera direction is chosen to show the Rotational reuses: over directions spread on the upper
hemisphere, the one whose worst seen share is largest, over every reuse's targets -- its cut (the faces of
the group before it whose centres lie outside the group after it and farther than OFF from its surface),
the work areas of the seat's tasks before it, and the new task's work area after it.  A target's seen share
is its unoccluded projected area (by that state's support and bunnies) over its area.

Run from the repository root:  PYTHONPATH=. .venv/bin/python -B subdivide_task/idea/step4_process/code/render.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from subdivide_task.step1.common import ROOT
from subdivide_task.step1.reach_video import Scene
from subdivide_task.operations.geometry import moved_mesh
from subdivide_task.step3.render_tool_membranes import image
from subdivide_task.step3.render_independent import BLUE
from subdivide_task.step4.seats import context, read_mesh
from subdivide_task.step4.render_merge import VIEW, upright
from subdivide_task.step4.combine import FOLDER

OBJECT, SAMPLE = 'B+03_kuka', 0
HERE = Path(__file__).resolve().parents[1]
OFF = 1.5e-3        # m: a face centre this far outside the joined support was cut away


def bunny(c, scene, slot):
    g = scene.r
    shape = g.Shape(scene.ctx, scene.shader, c.base.source.vertices, c.base.source.faces, g.GRAY, dynamic=True,
                    metallic=.03, roughness=.63)
    colors = np.tile(g.GRAY, (len(c.base.source.faces), 1))
    for i in slot['members']: colors[c.base.records[i]['working_face_ids']] = g.PINK
    shape.paint(colors); return shape


def panel(c, mesh, slots, T, points, size, view):
    """mesh and a bunny in every slot, all moved by T."""
    scene = Scene(moved_mesh(mesh, T), c.base.setup, *size)
    try:
        scene.view = view; scene.fit(points, 1.04)
        shapes = [(bunny(c, scene, s), T@np.asarray(s['X'])) for s in slots]
        return image(scene, np.tile(BLUE, (len(mesh.faces), 1)), [], None, shapes)
    finally: scene.close()


def seated(slots):
    """One bunny per placement: slots holding the object at the same X share it."""
    out = []
    for s in slots:
        same = [o for o in out if np.allclose(o['X'], s['X'], atol=1e-9)]
        if same: same[0]['members'] = same[0]['members']+list(s['members'])
        else: out.append(dict(s, members=list(s['members'])))
    return out


def cut(a, ab):
    """a's faces cut away by the join: centres outside ab and off its surface."""
    c = a.triangles_center
    return ~ab.contains(c) & (cKDTree(trimesh.sample.sample_surface_even(ab, 300000, seed=0)[0]).query(c)[0] > OFF)


def hemisphere(n=400):
    """n directions spread evenly over the upper hemisphere (Fibonacci)."""
    k = np.arange(n)+.5; z = k/n; phi = k*np.pi*(3-5**.5); r = np.sqrt(1-z*z)
    return np.c_[r*np.cos(phi), r*np.sin(phi), z]


def seen(mesh, faces, occluders, views):
    """Per view: the share of the faces' area seen unoccluded, in projected area."""
    rays = occluders.ray; centres, normals, areas = mesh.triangles_center[faces], mesh.face_normals[faces], mesh.area_faces[faces]
    out = np.zeros(len(views))
    for j, v in enumerate(views):
        facing = normals@v; k = facing > 0
        if k.any():
            hidden = rays.intersects_any(centres[k]+1e-5*v, np.tile(v, (int(k.sum()), 1)))
            out[j] = (areas[k]*facing[k])[~hidden].sum()/areas.sum()
    return out


def best_view(c, chain, T):
    """The camera direction that best shows the Rotational reuses (see the module notes)."""
    def state(path, slots):
        mesh = moved_mesh(read_mesh(ROOT/path), T); placed = seated(slots)
        bunnies = [moved_mesh(c.base.source, T@np.asarray(s['X'])) for s in placed]
        return mesh, placed, bunnies, trimesh.util.concatenate([mesh]+bunnies)
    views = hemisphere(); shares = []
    for step in chain:
        if step['operation'] != 'rotational_reuse': continue
        before, after = state(step['parent_files'][0], step['parent_slots'][0]), state(step['file'], step['slots'])
        new = step['slots'][len(step['parent_slots'][0])]
        shares.append(seen(before[0], cut(before[0], after[0]), before[3], views))
        for (mesh, placed, bunnies, occluders), tasks in ((before, None), (after, new['members'])):
            k = next(j for j, s in enumerate(placed) if np.allclose(s['X'], new['X'], atol=1e-9))
            for i in placed[k]['members'] if tasks is None else tasks:
                shares.append(seen(bunnies[k], c.base.records[i]['working_face_ids'], occluders, views))
    if not shares: return np.asarray(VIEW)/np.linalg.norm(VIEW)
    best = int(np.argmax(np.min(shares, 0))); print('SEEN', np.asarray(shares)[:, best].round(2), flush=True)
    return views[best]


def render(object_name=OBJECT, sample=SAMPLE, size=(800, 680)):
    c = context(object_name)
    report = json.loads((ROOT/'subdivide_task/data'/object_name/FOLDER/'report.json').read_text())
    chain = next(f for f in report['finals'] if f['sample'] == sample)['chain']; T = upright(c, chain[0]['slots'])
    data = [(chain[0]['parent_files'][0], chain[0]['parent_slots'][0])]+[(step['file'], step['slots']) for step in chain]
    data = [(read_mesh(ROOT/f), seated(s)) for f, s in data]
    points = np.vstack([moved_mesh(m, T).vertices for m, _ in data]
                       + [moved_mesh(c.base.source, T@np.asarray(s['X'])).vertices for s in chain[-1]['slots']])
    view = best_view(c, chain, T); print('VIEW', view.round(3), flush=True)
    w, h = size; canvas = Image.new('RGB', (len(data)*w, 2*h), 'white')
    for k, (mesh, slots) in enumerate(data):
        canvas.paste(panel(c, mesh, slots, T, points, size, view), (k*w, 0))
        canvas.paste(panel(c, mesh, [], T, points, size, view), (k*w, h))
        if k: print('STEP', k, chain[k-1]['name'], chain[k-1]['operation'], flush=True)
    canvas.save(HERE/'process.png'); print('SAVED', HERE/'process.png', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--object', default=OBJECT); p.add_argument('--sample', type=int, default=SAMPLE)
    a = p.parse_args(); render(a.object, a.sample)
