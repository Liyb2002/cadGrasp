"""A fixed work arm, a moving support: the KUKA holds its needle still, SLOPE down from the right, and a Franka Panda
pushes the support (the bunny in it) on the table and twists it about the vertical until the needle's line has passed
over the whole work area; no text, SECONDS long.  A schematic: the needle never touches -- the face its line meets
first is worked (a gold dashed line from the tip to it), the tip above the bunny's top so nothing ever hits it.

The bunny stands upright (original +Y up), its exit straight up; the work area is every face above the support whose
normal points up (light pink; deep pink once worked: the faces within BRUSH of the point the line meets).  The support
is a pedestal -- the convex hull of the bunny's lower part (up DEPTH of its height, its lowest point LIFT above the table) grown
by WALL, down to the table -- minus the bunny swept straight up, with FINS upright fins round it, gripped from
outside and above (the hand comes in TILT below level towards the axis, the fingers close across the fin).  The support's frame: its axis (the vertical
through the bunny's centre of mass) on the table.

The motion, in STAGES stages: the Panda grips the fin 45 degrees right of its own direction, twists the support
360/STAGES degrees about its axis while pushing it to and fro along the needle's horizontal direction -- the line
crosses the axis higher or lower, so it meets the bunny from the top of its far side down to the support's rim --
PASSES times, lets go, backs off and swings round the axis to the next fin.  The to and fro follows, for each turn, the heights at which the line
crossing the axis meets the bunny first (not the support, not the table).

The KUKA is B+03_kuka's LBR Med 14 at its station base (slanted.holder: moved towards the tip, IK with the most room
to the joint limits); the Panda (demo/code/franka.py: MuJoCo Menagerie's model, long slim fingertips, its IK) stands
FRANKA beyond the support, SIDE off the far side from the KUKA.  Every frame is checked: the needle and head off the bunny
and the support, the KUKA's links off them, the Panda's hulls off the bunny (its fingers may touch only the fin).

Run from the repository root:
  PYTHONPATH=.:whole_task_2/idea/easy_task/code:whole_task_2/idea/demo/code .venv/bin/python -B whole_task_2/idea/easy_task/code/push_twist.py
  ... --still 5.0      one frame at 5 s, as png (easy_task/code/push_twist_still.png)
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, Slerp
import manifold3d as md

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame, GOLD, LIGHT
from whole_task_2.step1.find_tasks_figure import OUTLINE_WIDTH_PX
from whole_task_2.step1.find_tasks_figure import dashed as dashes
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.combine import exact_sweep
import franka as F
from slanted import holder, inside, INFLATE

HERE = Path(__file__).resolve().parents[1]
SLOPE = 45.             # deg: the needle below horizontal
LIFT = .035             # m: the bunny's lowest point above the table
DEPTH = .5              # share of the bunny's height the pedestal reaches up (push_twist.mp4: .163, i.e. 25 mm)
WALL = .006             # m: the pedestal past the bunny's lower part
FINS = 4                # fins round the pedestal, evenly
FIN = (.04, .010, .025) # m: a fin's reach past the pedestal, thickness, height
FIN_MID = .052          # m: a fin's centre above the table
TILT = 60.              # deg: the hand's approach below level, in towards the axis
CLEAR = .02             # m: the needle tip above the bunny's top
BRUSH = .026            # m: the faces this close to the point the line meets are worked
STAGES = 4              # grips, each twisting 360/STAGES degrees
PASSES = 4              # to-and-fro passes in a stage (even: each stage starts and ends at the top)
FACE = 200.             # deg: the bunny's turn at the start (its face towards the camera)
FRANKA = .70            # m: the Panda's base beyond the support's mean place
SIDE = 30.              # deg: the Panda's direction from the support turned this far off the KUKA's reach, so it
                        #     comes in from the picture's left
OPEN, SHUT = .024, .010 # m: the fingers' gap, open and on a fin
BACK = .06              # m: the hand backs off this much between grips
TIMES = dict(come=.9, down=.25, close=.15, scan=3.5, open=.12, up=.18, move=.4, leave=.9)   # s
SECONDS = 20.
FPS = 30
TURN = 30.              # deg: the camera's azimuth from square to the KUKA's reach, towards the KUKA
RISE = .34              # the camera's height (before normalising)
SIZE = (1600, 900)
DASH = (.006, .004)     # m: the gold line's dashes and gaps
SUB = 8                 # brush stamps per frame


def frame(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


def spin(a):
    return Rotation.from_rotvec([0., 0., np.radians(a)]).as_matrix()


def ease(u):
    return u*u*(3-2*u)


def ramp(u, e=.12):
    """0 to 1 at constant speed but for a smooth start and stop over the first and last share e."""
    if u < e: return u*u/(2*e)/(1-e)
    if u > 1-e: return 1-(1-u)**2/(2*e)/(1-e)
    return (u-e/2)/(1-e)


# ---------------------------------------------------------------------------------------------
# The scene: the station, the bunny in its support (support frame: the axis on the table), the needle.

def station():
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    origin = np.r_[setup['work_point_m'][:2], setup['table_height_m']]               # the tabletop z = 0
    setup['base_position_world_m'] = (np.asarray(setup['base_position_world_m'])-origin).tolist()
    reach = -np.asarray(setup['base_position_world_m'])[:2]; reach /= np.linalg.norm(reach)
    s = np.radians(SLOPE); d = np.r_[np.cos(s)*reach, -np.sin(s)]; setup['working_direction_world'] = d.tolist()
    return setup, reach, d


def bunny_and_support(source):
    """B: support <- object (the bunny upright, its centre of mass over the axis, its lowest point LIFT up); the bunny
    there; the support mesh (support frame); the fins' grips (TCP frames, support frame: the approach TILT below
    level and in towards the axis, the fingers closing across the fin); the pedestal's top."""
    B = frame(np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]), np.zeros(3))               # original +Y up
    m = source.copy(); m.apply_transform(B); shift = -m.center_mass; shift[2] = LIFT-m.bounds[0, 2]
    B[:3, 3] = shift; m.apply_translation(shift)
    top = LIFT+DEPTH*float(m.extents[2]); low = m.vertices[m.vertices[:, 2] < top]
    grown = (low[:, None]+trimesh.creation.icosphere(2, WALL).vertices).reshape(-1, 3)
    pedestal = md.Manifold.hull_points(np.vstack([grown, grown*[1, 1, 0]])/G.SCALE).trim_by_plane([0, 0, -1.], -top/G.SCALE)
    outline = G.mesh(pedestal); parts = [pedestal]; grips = []
    for a in np.radians(45.+360./FINS*np.arange(FINS)):
        u, t = np.r_[np.cos(a), np.sin(a), 0.], np.r_[-np.sin(a), np.cos(a), 0.]
        edge = outline.ray.intersects_location([.4*u+[0, 0, FIN_MID]], [-u])[0]
        r = float(np.linalg.norm(edge[:, :2], axis=1).max())                           # the pedestal's side there
        fin = trimesh.creation.box(extents=[FIN[0]+.008, FIN[1], FIN[2]])
        fin.apply_transform(frame(np.c_[u, t, [0, 0, 1.]], (r+FIN[0]/2-.004)*u+[0, 0, FIN_MID]))
        parts.append(G.solid(fin))
        c = np.radians(TILT); z = -np.cos(c)*u-np.r_[0, 0, np.sin(c)]
        grips.append(frame(np.c_[np.cross(t, z), t, z], (r+FIN[0]/2)*u+[0, 0, FIN_MID]))
    inflated = trimesh.Trimesh(m.vertices+INFLATE*m.vertex_normals, m.faces, process=False)
    body = G.union(parts)-exact_sweep(inflated, np.r_[0, 0, 1.], top+.3)
    support = G.mesh(max(body.decompose(), key=lambda p: p.volume()))
    return B, m, support, grips, top


# ---------------------------------------------------------------------------------------------
# The line: where it meets the bunny for a turn psi of the support and the height z at which it crosses the axis.

class Line:
    def __init__(self, bunny, support, tip, d, reach):
        self.mesh = trimesh.util.concatenate([bunny, support]); self.nb = len(bunny.faces)
        self.tip, self.d, self.reach = tip, d, reach; self.run = 1/np.tan(np.radians(SLOPE))

    def place(self, psi, z):
        """The support's world frame: turned psi (deg) about its axis, the axis where the line crosses it at height z."""
        c = self.tip[:2]+(self.tip[2]-z)*self.run*self.reach; return frame(spin(psi), np.r_[c, 0.])

    def meet(self, psi, z):
        """Where the line first meets the bunny or the support (support frame) and whether it is the bunny; arrays."""
        psi, z = np.broadcast_arrays(np.asarray(psi, float), np.asarray(z, float)); psi, z = psi.ravel(), z.ravel()
        o, v = np.zeros((len(psi), 3)), np.zeros((len(psi), 3))
        for k in range(len(psi)):
            S = self.place(psi[k], z[k]); o[k] = S[:3, :3].T@(self.tip-S[:3, 3]); v[k] = S[:3, :3].T@self.d
        idx, ray, loc = self.mesh.ray.intersects_id(o, v, multiple_hits=False, return_locations=True)
        point = np.full((len(psi), 3), np.nan); face = np.full(len(psi), -1); point[ray] = loc; face[ray] = idx
        return point, face, (face >= 0) & (face < self.nb)

    def band(self, top):
        """lo(psi), hi(psi) on whole degrees: the lowest and highest crossing heights at which the line meets the bunny
        first, the run of them through the middle, smoothed over 15 degrees and kept 2 mm inside."""
        psi = np.arange(360.); zs = np.arange(0., top+.05, .001)
        _, _, ok = self.meet(np.repeat(psi, len(zs)), np.tile(zs, len(psi))); ok = ok.reshape(len(psi), len(zs))
        lo, hi = np.zeros(360), np.zeros(360)
        for k in range(360):
            on = np.flatnonzero(ok[k]); runs = np.split(on, np.flatnonzero(np.diff(on) > 1)+1)
            r = max(runs, key=len); lo[k], hi[k] = zs[r[0]], zs[r[-1]]
        kernel = np.ones(15)/15
        smooth = lambda x: np.convolve(np.r_[x[-7:], x, x[:7]], kernel, 'valid')
        return np.maximum(smooth(lo), lo)+.002, np.minimum(smooth(hi), hi)-.002


# ---------------------------------------------------------------------------------------------
# The timeline: the support's turn and height, the Panda's TCP, its fingers.

def timeline(line, lo, hi, grips, panda, q0):
    """Every frame: psi (deg), z, the TCP target (world; None in a joint-space move: free = (towards, share)), the
    fingers' gap, the fin held (or None)."""
    def band(psi):
        k = np.mod(psi, 360.); i = np.floor(k).astype(int); f = k-i
        return (1-f)*lo[i]+f*lo[(i+1) % 360], (1-f)*hi[i]+f*hi[(i+1) % 360]
    def height(psi, u):
        a, b = band(psi); return a+(b-a)*(.5+.5*np.cos(np.pi*PASSES*u))
    step = 360./STAGES; az = np.degrees(np.arctan2(*(panda.base[:2]-line.tip[:2])[::-1]))   # the Panda's side
    def fin(psi):
        """The fin nearest 45 degrees right of the Panda's direction (the grip's start), at turn psi."""
        want = az-45.; return int(np.argmin([abs((np.degrees(np.arctan2(g[1, 3], g[0, 3]))+psi-want+180) % 360-180) for g in grips]))
    def hand(psi, z, j, back=0.):
        T = line.place(psi, z)@grips[j]; T[:3, 3] -= back*T[:3, 2]; return T
    frames = []
    def hold(n, **f): frames.extend([dict(f) for _ in range(n)])
    n = lambda key: max(1, round(TIMES[key]*FPS))
    psi = FACE; z = height(psi, 0.); j = fin(psi)
    above = hand(psi, z, j, BACK); grip = hand(psi, z, j)
    for u in np.arange(1, n('come')+1)/n('come'): frames.append(dict(psi=psi, z=z, T=None, free=(above, ease(u)), gap=OPEN, held=None))
    for u in np.arange(1, n('down')+1)/n('down'): frames.append(dict(psi=psi, z=z, T=blend(above, grip, ease(u)), gap=OPEN, held=None))
    for u in np.arange(1, n('close')+1)/n('close'): frames.append(dict(psi=psi, z=z, T=grip, gap=OPEN+(SHUT-OPEN)*u, held=None))
    for k in range(STAGES):
        p0 = psi
        for u in np.arange(1, n('scan')+1)/n('scan'):
            psi = p0+step*ramp(u); z = height(psi, u); frames.append(dict(psi=psi, z=z, T=hand(psi, z, j), gap=SHUT, held=j))
        T = frames[-1]['T']
        for u in np.arange(1, n('open')+1)/n('open'): frames.append(dict(psi=psi, z=z, T=T, gap=SHUT+(OPEN-SHUT)*u, held=None))
        up = hand(psi, z, j, BACK)
        for u in np.arange(1, n('up')+1)/n('up'): frames.append(dict(psi=psi, z=z, T=blend(T, up, ease(u)), gap=OPEN, held=None))
        if k == STAGES-1:
            for u in np.arange(1, n('leave')+1)/n('leave'): frames.append(dict(psi=psi, z=z, T=None, free=(None, ease(u)), gap=OPEN, held=None))
            break
        j = fin(psi); above = hand(psi, z, j, BACK); grip = hand(psi, z, j)
        c = line.place(psi, z)[:3, 3]
        for u in np.arange(1, n('move')+1)/n('move'): frames.append(dict(psi=psi, z=z, T=swing(up, above, c, ease(u)), gap=OPEN, held=None))
        for u in np.arange(1, n('down')+1)/n('down'): frames.append(dict(psi=psi, z=z, T=blend(above, grip, ease(u)), gap=OPEN, held=None))
        for u in np.arange(1, n('close')+1)/n('close'): frames.append(dict(psi=psi, z=z, T=grip, gap=OPEN+(SHUT-OPEN)*u, held=None))
    hold(round(SECONDS*FPS)-len(frames), **frames[-1])
    return frames


def swing(A, B, c, u):
    """The TCP from A to B round the vertical through c: its angle and distance about it and its height in step, the
    rotation by slerp."""
    a, b = A[:3, 3]-c, B[:3, 3]-c; ta, tb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0]); tb = ta+(tb-ta+np.pi) % (2*np.pi)-np.pi
    t = (1-u)*ta+u*tb; r = (1-u)*np.linalg.norm(a[:2])+u*np.linalg.norm(b[:2])
    R = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix()
    return frame(R, c+[r*np.cos(t), r*np.sin(t), (1-u)*a[2]+u*b[2]])


def blend(A, B, u):
    """The TCP from A to B: straight, the rotation by slerp."""
    R = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix(); return frame(R, (1-u)*A[:3, 3]+u*B[:3, 3])


def joints(panda, frames, q0):
    """The Panda's joints for every frame: a joint-space move from q0 to the first grip's approach (its IK nearest
    the arm's plain posture: joints 3 and 5 straight, 7 as in q0) and back to q0 at the end; on the TCP's path each
    frame from the last (its IK, the null space drawn to the last with joints 3 and 5 halfway to straight)."""
    def roomy(T):
        found = [p for p in (panda.solve(T, [s]) for s in [q0, *np.random.default_rng(0).uniform(panda.lower, panda.upper, (200, 7))]) if p is not None]
        if not found: raise ValueError(f'no IK for the TCP at {T[:3, 3].round(3)}')
        return min(found, key=lambda p: abs(p[2])+abs(p[4])+abs(p[6]-q0[6]))
    q = q0; out = []; first = None
    for k, f in enumerate(frames):
        if f['T'] is None:
            target, u = f['free']
            if target is not None:
                first = roomy(target) if first is None else first; p = q0+(first-q0)*u
            else:
                last = last if k and frames[k-1]['T'] is None else q; p = last+(q0-last)*u
        else:
            rest = q.copy(); rest[[2, 4]] *= .5; p = panda.solve(f['T'], [q], rest=rest)
            if p is None: raise ValueError(f'frame {k}: no IK for the TCP at {f["T"][:3, 3].round(3)}')
        out.append(p); q = p
    return out


# ---------------------------------------------------------------------------------------------

def main(still=None, name='push_twist_half'):
    source = load_shape(ROOT/'objects/B/mesh.stl'); setup, reach, d = station()
    B, bunny, support, grips, rim = bunny_and_support(source)
    tip = np.r_[0., 0., bunny.bounds[1, 2]+CLEAR]
    line = Line(bunny, support, tip, d, reach); lo, hi = line.band(bunny.bounds[1, 2])
    mean = line.place(0., float(np.mean([lo, hi])))[:3, 3]
    side = spin(SIDE)[:2, :2]@reach; panda = F.Panda(np.r_[mean[:2]+FRANKA*side, 0.], np.arctan2(-side[1], -side[0]))
    q0 = panda.ready.copy()
    frames = timeline(line, lo, hi, grips, panda, q0); Q = joints(panda, frames, q0)
    arm, links, q_kuka, _ = holder(setup, tip, d)

    # The work: the line's point every SUB-th of a frame (the support moves only while held), the faces it works.
    work = (bunny.face_normals[:, 2] > 0) & (bunny.triangles_center[:, 2] > rim); done = np.zeros(len(bunny.faces), bool); painted = []; hits = []
    centres = bunny.triangles_center
    for k, f in enumerate(frames):
        prev = frames[max(k-1, 0)]; us = np.arange(1, SUB+1)/SUB
        point, _, ok = line.meet(prev['psi']+(f['psi']-prev['psi'])*us, prev['z']+(f['z']-prev['z'])*us)
        for p in point[ok]: done |= work & (np.linalg.norm(centres-p, axis=1) < BRUSH)
        painted.append(done.copy()); hits.append((point[-1], bool(ok[-1])))
    share = bunny.area_faces[done].sum()/bunny.area_faces[work].sum()

    # Checks on dense samples: the needle and head, the KUKA, the Panda against the bunny and the support.
    L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    sb = trimesh.sample.sample_surface_even(bunny, 30000, seed=0)[0]; ss = trimesh.sample.sample_surface_even(support, 30000, seed=0)[0]
    tool = bad_tool = bad_kuka = 0; bad_panda = []
    hulls = [Delaunay(arm.collisions[i]) for i in range(1, 8)]
    for k in range(0, len(frames), 3):
        S = line.place(frames[k]['psi'], frames[k]['z']); P = np.vstack([sb, ss])@S[:3, :3].T+S[:3, 3]
        bad_tool += bool(inside(P, tip, tip-L*d, N).any() or inside(P, tip-L*d, tip-(L+.07)*d, H).any())
        bad_kuka += any(h.find_simplex((P-links[i+1][:3, 3])@links[i+1][:3, :3]).max() >= 0 for i, h in enumerate(hulls))
        Fp = panda.frames(Q[k], frames[k]['gap']); moving = [i for i in range(len(Fp)) if panda.level[i] > 0]
        others = [i for i in moving if i not in panda.fingers]
        if (F.pierced([panda.pieces[i] for i in others], Fp[others], F.Cloud(np.vstack([sb, ss])@S[:3, :3].T+S[:3, 3]))
                or F.pierced([panda.pieces[i] for i in panda.fingers], Fp[panda.fingers], F.Cloud(sb@S[:3, :3].T+S[:3, 3]))):
            bad_panda.append(k)
    print(f'{len(frames)} frames; work area {100*bunny.area_faces[work].sum()/bunny.area:.1f}% of the surface, worked {100*share:.2f}%;',
          f'line height {1e3*lo.min():.0f}-{1e3*hi.max():.0f} mm, tip {1e3*tip[2]:.0f} mm; support {1e6*support.volume:.0f} cm3;',
          f'frames with needle/head hits {bad_tool}, KUKA hits {bad_kuka}, Panda hits {len(bad_panda)} {bad_panda[:8]}', flush=True)

    # The film.
    reach_az = np.degrees(np.arctan2(-reach[1], -reach[0])); camera = reach_az-90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE]; view /= np.linalg.norm(view)
    span = [line.place(f['psi'], f['z']) for f in frames[::10]]
    fit = np.vstack([*(np.vstack([bunny.vertices[::5], support.vertices[::20]])@S[:3, :3].T+S[:3, 3] for S in span),
                     tip, tip-(L+.07)*d+[0, 0, .05], *(f['T'][:3, 3]+[0, 0, .12] for f in frames[::10] if f['T'] is not None)])
    table = trimesh.creation.box(bounds=[[*(fit[:, :2].min(0)-.25), -.02], [*(fit[:, :2].max(0)+.25), 0.]])
    scene = Scene(source, setup, *SIZE, arm=True)
    try:
        g = scene.r; scene.view = view; scene.fit(fit, 1.15); scene.ctx.line_width = 1.6*OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]
        table_shape = g.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), smooth=False, roughness=.85)
        support_shape = g.Shape(scene.ctx, scene.shader, support.vertices, support.faces, BLUE, metallic=.05, roughness=.5)
        panda_shapes = [g.Shape(scene.ctx, scene.shader, v, f, c, metallic=.15, roughness=.45) for v, f, c in panda.visuals()]
        needle = [(part, tool_frame(tip, d)) for part in scene.tools[False]]

        def picture(k):
            f = frames[k]; S = line.place(f['psi'], f['z']); W = S@B
            colors = np.tile(g.GRAY, (len(source.faces), 1)); colors[work] = LIGHT; colors[painted[k]] = g.PINK
            shapes = [(table_shape, np.eye(4)), (support_shape, S)]+needle
            shapes += list(zip(panda_shapes, panda.visual_frames(Q[k], f['gap'])))
            point, ok = hits[k]; end = S[:3, :3]@point+S[:3, 3] if ok else tip+.3*d
            segments = [(np.asarray(dashes([(tip, end)], *DASH)).reshape(-1, 3), GOLD)]
            return scene.draw(W, colors, [], links=links, shapes=shapes, segments=segments)
        if still is not None:
            k = min(int(round(still*FPS)), len(frames)-1); path = HERE/'code/push_twist_still.png'
            Image.frombytes('RGB', SIZE, picture(k)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path); print('STILL', path); return
        path = HERE/f'{name}.mp4'
        ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS),
                                   '-i', '-', '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-an', str(path)],
                                  stdin=subprocess.PIPE)
        try:
            for k in range(len(frames)): ffmpeg.stdin.write(picture(k))
        finally:
            ffmpeg.stdin.close(); ffmpeg.wait()
        print('VIDEO', path, f'{len(frames)/FPS:.1f} s')
    finally:
        scene.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--still', type=float)
    p.add_argument('--depth', type=float, default=DEPTH, help="share of the bunny's height the pedestal reaches up")
    p.add_argument('--name', default='push_twist_half', help='the video: easy_task/<name>.mp4 (a new variant, the earlier kept)')
    a = p.parse_args(); DEPTH = a.depth; main(a.still, a.name)
