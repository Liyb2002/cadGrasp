"""change_task: Step2's first covers overlap, so the borders between the areas can move.  Drawn from a run's saved
Step2 data, read only (the same sets and colours as Step2's figure, step2/directions_figure.py): what each chosen work
direction reaches (Step1's screen of needle axes within the tilt of that direction), on the target area.

The figure: on top every cover alone as Step2's figure shows it (row "1 Cover": the bunny at its area's pose, the
station's needle, head and gold cone at its seed, its faces in its colour, pale: the rest); below all covers put
together on one bunny: a face one direction reaches is its colour, a face several reach the mix of their colours (a
new colour for every overlap).

The video (--video): one bunny cut into areas without overlap, the borders drawn, and the borders moving.  It starts
with every face in the area of the direction that reaches it most squarely and ends with Step2's areas; direction by
direction (W01, W02, ...) the bunny turns that direction's faces that go to another area towards the camera, and the
areas that take them push their borders in (each face changes when the front from its new area reaches it); at the
end one turn round the bunny.

  python -m whole_task_2.operations.render_change_task B+whole_shape_v2
  python -m whole_task_2.operations.render_change_task B+whole_shape_v2 --video
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape, object_of
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR
from whole_task_2.step1.reach_video import Scene
from whole_task_2.step2.directions_figure import AREA_COLORS, CHUNK, MARGIN, OFFSET, PALE, placed, station_view, unit

HERE = Path(__file__).resolve().parent
TILE = 200            # px: a cover in the figure's top row
BUNNY = 1050          # px: the figure's bunny, height (its width: the row of tiles)
FRAME = (1080, 1080)  # px: the video
UPRIGHT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])   # the mesh's +Y up
FPS = 30
START, TURN, MOVE, HOLD, SPIN, END = 1., .8, 1.5, .3, 4., .6       # s: the video's beats
TILT = np.radians(60)  # the most the bunny tips over to show faces from below or above


def covers(run):
    """The mesh, Step1's setup, the chosen directions (object frame), each one's area pose, its seed, its cover
    (direction x face) and Step2's areas (each face's area)."""
    work = ROOT/'whole_task_2/data'/run; plan = json.loads((work/'step1_reach/plan.json').read_text()); s = plan['setup']
    report = json.loads((work/'step2_seats/report.json').read_text())
    with np.load(work/'step1_reach/plan.npz') as z: target = z['target'].astype(bool)
    with np.load(work/'step1_reach/screen.npz') as z: axes, reach = z['axes'], z['reach']
    with np.load(work/'step2_seats/supports.npz') as z: poses = [z[a['id']+'_pose'] for a in plan['areas']]
    mesh = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl')
    V = np.asarray([unit(a['axis_object']) for a in plan['areas']]); cosb = np.cos(np.radians(s['max_tilt_deg']))
    sets = np.asarray([reach[axes@v >= cosb-1e-12].any(0) for v in V]) & target
    seeds = [None]*len(V)
    for cut in report.get('blocks', []):
        j = int(np.argmax(V@unit(cut['direction'])))
        if V[j]@unit(cut['direction']) >= 1-1e-6: seeds[j] = cut['seed']
    for j in range(len(V)):
        if seeds[j] is None: mine = np.flatnonzero(sets[j]); seeds[j] = int(mine[np.argmax(mesh.face_normals[mine]@-V[j])])
    owner = np.full(len(mesh.faces), -1)
    for j, a in enumerate(plan['areas']): owner[a['face_ids']] = j
    return mesh, s, V, poses, seeds, sets, owner


def mixed(weights, colors):
    """Face colours: each face the mix of the directions' colours by weights (direction x face); pale where none."""
    total = weights.sum(0); out = np.tile(PALE, (weights.shape[1], 1)); on = total > 1e-9
    out[on] = (weights[:, on].T@np.asarray(colors))/total[on, None]
    return out


def flip(data, size):
    return Image.frombytes('RGB', size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def border(mesh, label, P):
    """Scene.draw's line segments on the edges between faces of different labels, lifted off the surface, at P."""
    a, b = mesh.face_adjacency.T; cut = label[a] != label[b]
    q = placed((mesh.vertices+OFFSET*mesh.vertex_normals)[mesh.face_adjacency_edges[cut]].reshape(-1, 3), P)
    return [(q[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(q), CHUNK)]


def upright(mesh):
    """The upright pose on the table (world from object)."""
    P = np.eye(4); P[:3, :3] = UPRIGHT
    P[:3, 3] = -placed(mesh.vertices, P).min(0)*[0, 0, 1]-placed(mesh.centroid[None], P)[0]*[1, 1, 0]
    return P


def figure(run, path):
    mesh, s, V, poses, seeds, sets, owner = covers(run); k = len(sets); w = unit(s['working_direction_world'])
    colors = np.asarray([AREA_COLORS[j % len(AREA_COLORS)] for j in range(k)]); view = station_view(w)
    tiles = []; scene = Scene(mesh, s, TILE, TILE)
    try:
        scene.view = view
        for j, P in enumerate(poses):
            paint = np.tile(PALE, (len(mesh.faces), 1)); paint[sets[j]] = colors[j]
            tools = [(placed(mesh.triangles_center[seeds[j]], P), w, False)]
            scene.fit(np.vstack([placed(mesh.vertices, P), scene.points(P, tools)]), MARGIN)
            tiles.append(flip(scene.draw(P, paint, tools), scene.size))
    finally:
        scene.close()
    size = (k*TILE, BUNNY); scene = Scene(mesh, s, *size)
    try:
        P = upright(mesh); scene.view = view; scene.fit(placed(mesh.vertices, P), MARGIN)
        bunny = flip(scene.draw(P, mixed(sets.astype(float), colors), []), size)
    finally:
        scene.close()
    sheet = Image.new('RGB', (k*TILE, TILE+BUNNY), 'white')
    for j, im in enumerate(tiles): sheet.paste(im, (j*TILE, 0))
    sheet.paste(bunny, (0, TILE))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    count = sets.sum(0); area = mesh.area_faces; total = area[sets.any(0)].sum()
    print('CHANGE TASK FIGURE', path, '|', k, 'covers | area reached by n directions:',
          {n: f'{100*area[count == n].sum()/total:.1f}%' for n in range(1, k+1) if (count == n).any()}, flush=True)
    return path


def ease(u):
    u = min(max(u, 0.), 1.); return u*u*(3-2*u)


def facing(m, c):
    """(psi, tau): the turn about z, then the tilt (towards up) about the horizontal axis across the view c, that
    bring the world direction m to face the camera; the tilt within TILT."""
    m = unit(m); psi = np.arctan2(c[1], c[0])-np.arctan2(m[1], m[0])
    tau = np.clip(np.arcsin(np.clip(c[2], -1, 1))-np.arcsin(np.clip(m[2], -1, 1)), -TILT, TILT)
    return float(np.angle(np.exp(1j*psi))), float(tau)


def turned(mesh, psi, tau, c):
    """The pose (world from object): upright, turned psi about z and tilted tau about the horizontal axis across the
    view c, about the object's centre (at the origin)."""
    h = unit(np.r_[c[:2], 0.]); R = (Rotation.from_rotvec(tau*unit(np.cross(h, [0., 0., 1.])))
                                     * Rotation.from_rotvec([0., 0., psi])).as_matrix()@UPRIGHT
    P = np.eye(4); P[:3, :3] = R; P[:3, 3] = -R@mesh.centroid; return P


def video(run, path):
    mesh, s, V, poses, seeds, sets, owner = covers(run); k = len(sets); n = len(mesh.faces)
    colors = np.asarray([AREA_COLORS[j % len(AREA_COLORS)] for j in range(k)]); c = station_view(unit(s['working_direction_world']))
    # The start: every face in the area of the direction that reaches it most squarely (Step2's area where none does).
    start = np.argmax(np.where(sets, -V@mesh.face_normals.T, -np.inf), 0); start[~sets.any(0)] = owner[~sets.any(0)]
    a, b = mesh.face_adjacency.T; d = np.linalg.norm(mesh.triangles_center[a]-mesh.triangles_center[b], axis=1)
    graph = coo_matrix((np.r_[d, d], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()

    # The beats: (psi, tau, labels) per frame.
    label = start.copy(); frames = []; psi = tau = None
    def turn(seconds, to):
        nonlocal psi, tau
        if psi is None: psi, tau = to; return
        dp = float(np.angle(np.exp(1j*(to[0]-psi)))); p0, t0 = psi, tau; m = max(int(round(seconds*FPS)), 1)
        for i in range(m): u = ease((i+1)/m); frames.append((p0+u*dp, t0+u*(to[1]-t0), label.copy()))
        psi, tau = p0+dp, to[1]
    def hold(seconds): frames.extend([(psi, tau, label.copy())]*int(round(seconds*FPS)))
    steps = []
    for j in range(k):
        moving = np.flatnonzero((start == j) & (owner != j))
        if len(moving): steps.append((j, moving))
    for number, (j, moving) in enumerate(steps):
        normal = (mesh.area_faces[moving, None]*mesh.face_normals[moving]).sum(0)
        turn(TURN, facing(UPRIGHT@normal, c))
        if number == 0: hold(START)
        when = np.zeros(len(moving))                       # each face's turn: the front from its new area reaches it
        for r in np.unique(owner[moving]):
            mine = owner[moving] == r; reach = dijkstra(graph, indices=np.flatnonzero(label == r), min_only=True)
            when[mine] = reach[moving[mine]]
        when = when/max(when.max(), 1e-12); m = int(round(MOVE*FPS))
        for i in range(m):
            now = moving[when <= ease((i+1)/m)]; label[now] = owner[now]; frames.append((psi, tau, label.copy()))
        hold(HOLD)
    turn(TURN, (psi, 0.)); m = int(round(SPIN*FPS)); p0 = psi
    for i in range(m): frames.append((p0+2*np.pi*ease((i+1)/m), 0., label.copy()))
    hold(END)

    scene = Scene(mesh, s, *FRAME); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{FRAME[0]}x{FRAME[1]}',
                                '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
                                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    try:
        scene.view = c; right = unit(np.cross([0., 0., 1.], c)); up = np.cross(c, right)
        r = float(np.linalg.norm(mesh.vertices-mesh.centroid, axis=1).max())
        scene.fit(r*np.array([right, -right, up, -up]), 1.04)                      # one frame: the bunny's sphere
        for p, t, lab in frames:
            P = turned(mesh, p, t, c)
            process.stdin.write(flip(scene.draw(P, colors[lab], [], None, border(mesh, lab, P)), FRAME).tobytes())
    finally:
        scene.close(); process.stdin.close(); code = process.wait()
    if code: raise RuntimeError('video encoding failed')
    area = mesh.area_faces; moved = start != owner
    print('CHANGE TASK VIDEO', path, '|', len(frames), 'frames', f'{len(frames)/FPS:.1f} s |',
          f'{100*area[moved].sum()/area.sum():.1f}% of the area changes area', flush=True)
    return path


def main():
    p = argparse.ArgumentParser(); p.add_argument('run', nargs='?', default='B+whole_shape_v2')
    p.add_argument('--video', action='store_true'); p.add_argument('--out'); a = p.parse_args()
    if a.video: video(a.run, a.out or HERE/'change_task.mp4')
    else: figure(a.run, a.out or HERE/'change_task_overlap.png')


if __name__ == '__main__':
    main()
