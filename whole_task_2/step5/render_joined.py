"""Step5 figures, no text.

output/<object>/step5/joined.png: one panel per slot, the whole support with the bunny in that slot
(pink: the slot's task areas), and last the support alone.  Same camera for every panel.

output/<object>/step5/use.png: every task at the station, the support turned on the table and its
slot slid under the needle; the KUKA holds the needle on the task's central certified face at the
joints Step1/Step4 certified there.  The arm rises and falls with the object, so each picture is
drawn at the certified height: the support and its table are raised by the same amount.

Step5's figure (combined.render) draws every pose of the support (poses: one per resting face) above every task's
use of it, and beside it <name>_hulls.png (hull_figure): in one pose, where the hull every resting face's feet must
enclose is, in 01_hulls_and_support.mp4's drawing (transparent.py).
"""
import math
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from scipy.spatial import ConvexHull

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT
from whole_task_2.step1.reach_render import contact
from whole_task_2.step1.reach_video import STATION_VIEW, Scene, kuka as work_arm
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.operations.state import transform
from whole_task_2.step3.render_tool_membranes import camera, image
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step4.seats import read_mesh, SAME_DOWN
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


def turned(view, deg):
    """The view turned deg about the vertical (counterclockwise from above)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg)); return np.array([c*view[0]-s*view[1], s*view[0]+c*view[1], view[2]])


def resting(q):
    """The installations by the support's face on the table: [(its down, support frame; [task indices])], in order
    of first use."""
    out = []
    for k, M in enumerate(q.material):
        g = M[:3, :3]@np.array([0., 0., -1.]); b = next((b for b in out if 1-g@b[0] < SAME_DOWN), None)
        if b is None: out.append((g, [k]))
        else: b[1].append(k)
    return out


def snapshot(scene, W, colors, shapes=()):
    """The scene's body (its mesh) at W with more shapes: an RGB image."""
    return Image.frombytes('RGB', scene.size, scene.draw(W, colors, [], None, (), shapes)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def poses(model, q, support, size=620, view=(.55, -1., .55)):
    """Every pose of the support: one panel per resting face, the support as its first installation there stands
    it, the bunny in its seat (pink: the areas of the tasks on that face)."""
    out = []
    for g, ks in resting(q):
        k = ks[0]; W = np.linalg.inv(q.material[k]); scene = Scene(support, model.setup, size, size)
        try:
            r = scene.r
            bunny = r.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, r.GRAY, dynamic=True, metallic=.03, roughness=.63)
            colors = np.tile(r.GRAY, (len(model.source.faces), 1))
            for j in ks: colors[model.records[j]['working_face_ids']] = r.PINK
            bunny.paint(colors); scene.view = np.asarray(view, float)/np.linalg.norm(view)
            scene.fit(np.vstack([transform(support.vertices, W), moved_mesh(model.source, q.poses[k]).vertices]), 1.1)
            out.append(snapshot(scene, W, np.tile(BLUE, (len(support.faces), 1)), [(bunny, q.poses[k])]))
        finally: scene.close()
    return out


def hull_figure(model, q, support, path, main=900, small=340, view=(.55, -1., .55), side=(1., -1., 1.)):
    """Where every hull is, in one pose (01_hulls_and_support.mp4's drawing, copied): left, the support as its first
    installation stands it with, for every resting face in its colour, the hull its feet must enclose there -- the
    convex hull of every demand's centre of pressure (the weight and the needle's push, Step5's ground rule) of the
    tasks on that face, on that face's table: a faint translucent fill, its outline, dots on its edges every 2 mm and
    some of the centres inside; right, the same in every pose, the bunny a ghost.  Saved to path."""
    from whole_task_2.step4.ground_oracle import ground_points
    from . import transparent as T
    tasks = model.prepare(q); renderer = T.Renderer(); faces = resting(q); hulls = []
    for n, (g, ks) in enumerate(faces):
        cloud = np.vstack([transform(np.c_[xy, np.zeros(len(xy))], q.material[k]) for k in ks
                           for xy in [ground_points(tasks[k]['endpoints']['endpoint_need_wrench'], tasks[k]['domain'].com)]])
        u = np.cross(g, [1., 0., 0.] if abs(g[0]) < .9 else [0., 1., 0.]); u /= np.linalg.norm(u); v = np.cross(g, u)
        hull = cloud[ConvexHull(np.c_[cloud@u, cloud@v]).vertices]
        edges = np.vstack([a+(b-a)*f[:, None] for a, b in zip(hull, np.roll(hull, -1, 0))
                           for f in [np.arange(max(1, int(np.ceil(np.linalg.norm(b-a)/.002))))/max(1, int(np.ceil(np.linalg.norm(b-a)/.002)))]])
        fan = trimesh.Trimesh(hull, np.array([[0, i, i+1] for i in range(1, len(hull)-1)]), process=False)
        hulls.append(dict(color=T.COLORS[n % len(T.COLORS)], hull=hull, edges=edges, fan=fan,
                          dots=np.unique(np.vstack([cloud[np.linspace(0, len(cloud)-1, 256, dtype=int)], hull]), axis=0)))

    def parts(W, cam, size, opacity):
        out = [T.placed(T.piece(support, T.BLUE, opacity), W)]
        for h in hulls:
            platform = T.piece(h['fan'], h['color'], .06); platform['unlit'] = True; out.append(T.placed(platform, W))
            border = T.outline_piece(transform(h['hull'], W), h['color'], cam, size, half_width_pixels=.8)
            border['colors'] = np.broadcast_to(T.rgb(h['color']), border['triangles'].shape); border['opacity'] = .28
            dots = T.points_piece(transform(h['dots'], W), h['color'], cam, size, half_size_pixels=1.05 if size == main else .65)
            edges = T.points_piece(transform(h['edges'], W), h['color'], cam, size, half_size_pixels=1.4 if size == main else .85)
            dots['opacity'] = .8; edges['opacity'] = .95; out += [T.placed(x, np.eye(4)) for x in (border, dots, edges)]
        return out
    extent = lambda W: np.vstack([transform(support.vertices, W), *[transform(h['hull'], W) for h in hulls]])
    W = np.linalg.inv(q.material[0]); cam = T.camera(extent(W), view, 1.12); left = renderer.render(parts(W, cam, main, .78), cam, main)
    rows = (len(faces)+1)//2; right = Image.new('RGB', (700, max(main, 55+rows*450)), 'white')
    for i, (g, ks) in enumerate(faces):
        k = ks[0]; W = np.linalg.inv(q.material[k]); body = moved_mesh(model.source, q.poses[k])
        cam = T.camera(np.vstack([extent(W), body.vertices]), side)
        right.paste(renderer.render(parts(W, cam, small, .82)+[T.placed(T.piece(body, T.GRAY, .18), np.eye(4))], cam, small),
                    (7+(i % 2)*346, 55+(i//2)*450))
    sheet = Image.new('RGB', (main+right.width, right.height), 'white'); sheet.paste(left, (0, (right.height-main)//2)); sheet.paste(right, (main, 0))
    d = ImageDraw.Draw(sheet); d.line((main, 0, main, sheet.height-1), fill='#eef0f2', width=1)
    for r in range(1, rows): d.line((main+1, r*450, sheet.width-1, r*450), fill='#eef0f2', width=1)
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP5 HULLS', path, flush=True)


def use(model, q, support, k, size=620, view=STATION_VIEW):
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
        blue = np.tile(BLUE, (len(world.faces), 1)); scene.view = np.asarray(view, float)/np.linalg.norm(view)
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
