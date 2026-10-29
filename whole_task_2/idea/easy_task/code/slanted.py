"""The easy task at a 45 degree needle: one needle place and one support serve six bunny poses; no text.

The needle on the KUKA points 45 degrees down, away from the arm's base.  Its tip hovers GAP before a fixed
world point P (it does not touch the bunny); the support holds the bunny from the other side on the same
line, at a fixed world point S one chord C further along the needle.  Both P and S are the same in every
pose; only the bunny turns.  Two supports, one figure each:
  rod    (slanted_rod.png): a thin round-ended rod (radius RADIUS), its end's centre at S, running on along
         the line down to the table;
  block  (slanted_block.png): one big block wrapping the line of action and standing on the table:
         everything within BLOCK of the line from BACK before S on through the table, and everything below
         that down to the table, minus every posed bunny grown by INFLATE (its largest piece), so each pose
         sits in what is left of its own imprint;
  slot   (slanted_slot.png): the same block, SLOT about the line and SLOT_BACK back, minus each posed bunny's
         whole sweep back along the needle (step4's exact_sweep): a slot the bunny slides into along the push
         and that holds it every other way.  Each pose reports its contact area and how far from the way
         back along the needle it could still move; below 90 degrees it is held.

Pose k turns object side k (+X, -X, +Y, -Y, +Z, -Z) towards the needle (original +Y up as far as it goes,
+Z for the Y sides) and then spins it about the needle line.  For one pose, every line along the needle on a
GRID whose rod section crosses the bunny once is a candidate: the needle line enters at P and the rod's round
end rests on the bunny with its centre S on the line (the contact force passes through S); C is the one chord, within TOL,
that every pose has, chosen so that the worst horizontal distance from a pose's centre of mass to its S is
smallest; each pose takes its spin and line with the centre of mass most nearly above S.  Then the needle
force and the weight both pass through the one support contact.  The pink work area is the faces within
PATCH of P that face the needle.

The arm is B+03_kuka's KUKA LBR Med 14 at its station base, the table at the station's height (the drawn
tabletop just covers the bunnies and the support).  The needle tip sits over the station's work point,
the arm moved towards it along its reach by at most SLIDE where IK holds the needle with the most room to
its joint limits.  Every pose is checked on dense samples: the arm links against bunny, support and table;
the needle and head against bunny and support; the support against the bunny; the bunny LIFT above the table.

One camera for every panel, across the arm's reach and turned TURN towards the arm, frames all poses and
the head; the rest of the arm leaves the image.  The push direction is marked one of two ways:
  line  (slanted_<support>.png): a gold dashed line, drawn over everything, the line of action from the
        needle tip through the bunny to S (rod) or on through the block to the table (block);
  cone  (slanted_<support>_cone.png): the gold dashed cone of similar push directions, CONE about the needle
        from its root to its tip, drawn at CONE_OPACITY like the other idea figures.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/easy_task/code/slanted.py --support block --mark cone
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation
import manifold3d as md

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame, kuka, arm_links, GOLD
from whole_task_2.step1.find_tasks_figure import task_outline, OUTLINE_WIDTH_PX, CONE_OPACITY, Opacity, cone, tubes
from whole_task_2.step1.find_tasks_figure import dashed as dashes
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.combine import exact_sweep

HERE = Path(__file__).resolve().parents[1]
SLOPE = 45.         # deg: the needle below horizontal
LIFT = .05          # m: every pose's lowest vertex above the table
RADIUS = .004       # m: the rod
BLOCK = .03         # m: the block's radius about the line of action
BACK = .03          # m: how far back from S, inside the bunnies, the block starts
INFLATE = .0003     # m: the bunnies grown by this before they are cut out, so no surfaces coincide
CONTACT = .001      # m: support faces this close to a bunny hold it
SLOT = .07          # m: the slot's radius about the line of action
SLOT_BACK = .05     # m: how far back from S the slot's walls reach
PATCH = .018        # m: the work area's radius about P
GAP = .012          # m: the needle tip before P
GRID = .001         # m: line spacing across the needle
TOL = .0005         # m: chord tolerance
SPIN = 15.          # deg: spin step about the needle line
SLIDE = .15         # m: how far the arm may move towards the needle tip
TURN = 30.          # deg: camera azimuth past square to the arm's reach, towards the arm (the work areas face it)
RISE = .35          # camera height over its horizontal distance
SIZE = (1000, 900)  # px: one panel
DASH, SPACE, WIDTH = 16, 10, 5   # px: the line of action
CONE = 25.          # deg: the cone of similar push directions about the needle
COLUMNS, GAP_PX = 3, 24
SIDES = dict(zip(['+X', '-X', '+Y', '-Y', '+Z', '-Z'], [s*a for a in np.eye(3) for s in (1, -1)]))


def facing(axis, d):
    """Object side `axis` towards the needle (-d); original +Y (+Z for the Y sides) as near world up as it goes."""
    up = np.eye(3)[1] if abs(axis[1]) < .5 else np.eye(3)[2]
    a = -d; w = np.array([0, 0, 1.])-a[2]*a; w /= np.linalg.norm(w)
    o = up-(up@axis)*axis; o /= np.linalg.norm(o)
    return np.c_[a, w, np.cross(a, w)]@np.c_[axis, o, np.cross(axis, o)].T


def basis(u):
    e1 = np.cross(u, [1., 0, 0] if abs(u[0]) < .9 else [0, 1., 0]); e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def crossings(mesh, u):
    """Every line along u on a GRID whose rod section (the lines within RADIUS) crosses the mesh once: the entry,
    and the centre of the rod's round end resting on the mesh, on that line."""
    E = np.c_[basis(u)[0], basis(u)[1]]; p = mesh.vertices@E
    g = np.mgrid[p[:, 0].min():p[:, 0].max():GRID, p[:, 1].min():p[:, 1].max():GRID]
    shape = g.shape[1:]; o = g.reshape(2, -1).T@E.T-u
    loc, ray, _ = mesh.ray.intersects_location(o, np.tile(u, (len(o), 1)), multiple_hits=True)
    n = np.bincount(ray, minlength=len(o)); t = (loc-o[ray])@u
    t = t[np.lexsort((t, ray))]; first = np.r_[0, np.cumsum(n)[:-1]]; once = n == 2
    enter = np.where(once, t[np.minimum(first, len(t)-1)], np.nan)
    k = round(RADIUS/GRID); leave = np.pad(np.where(once, t[np.minimum(first+1, len(t)-1)], np.nan).reshape(shape), k,
                                           constant_values=np.nan)
    rest = np.full(shape, -np.inf)
    for i, j in np.argwhere(np.hypot(*np.mgrid[-k:k+1, -k:k+1]) <= k)-k:
        lift = (RADIUS**2-(GRID*np.hypot(i, j))**2)**.5
        rest = np.fmax(rest, leave[k+i:k+i+shape[0], k+j:k+j+shape[1]]+lift)
    ok = binary_erosion(once.reshape(shape), np.hypot(*np.mgrid[-k:k+1, -k:k+1]) <= k).ravel()
    return o[ok]+enter[ok, None]*u, o[ok]+rest.ravel()[ok, None]*u


def poses(mesh, d):
    """Per side: (centre-of-mass offset, rotation, entry, exit) in the object frame, all with one chord; the chord."""
    com = mesh.center_mass; options = {}
    for name, axis in SIDES.items():
        options[name] = []
        for s in np.arange(0, 360, SPIN):
            R = Rotation.from_rotvec(np.radians(s)*d).as_matrix()@facing(axis, d)
            entry, exit = crossings(mesh, R.T@d)
            options[name].append((R, entry, exit, np.linalg.norm(exit-entry, axis=1),
                                  np.linalg.norm(((exit-com)@R.T)[:, :2], axis=1)))

    def best(name, c):
        found = []
        for R, entry, exit, chord, off in options[name]:
            near = np.abs(chord-c) < TOL
            if near.any(): k = np.argmin(np.where(near, off, np.inf)); found.append((off[k], R, entry[k], exit[k]))
        return min(found, key=lambda f: f[0]) if found else (np.inf,)
    top = min(max(o[3].max() for o in side) for side in options.values())
    c = min(np.arange(.01, top, .001), key=lambda c: max(best(name, c)[0] for name in SIDES))
    return {name: best(name, c) for name in SIDES}, c


def holder(setup, tip, d):
    """The arm, moved towards the tip along its reach in 1 cm steps up to SLIDE, holding the needle with the
    most room to its joint limits: (arm, links, joints, slide in m)."""
    base = np.asarray(setup['base_position_world_m']); unit = -base[:2]/np.linalg.norm(base[:2]); best = None
    for k in range(round(SLIDE/.01)+1):
        arm = kuka(dict(setup, base_position_world_m=(base+np.r_[k*.01*unit, 0.]).tolist()))
        try: links, q = arm_links(arm, tip, d)
        except ValueError: continue
        room = np.min(np.minimum(q-arm.lower, arm.upper-q)/(arm.upper-arm.lower))
        if best is None or room > best[0]: best = (room, arm, links, q, k*.01)
    if best is None: raise ValueError('The arm cannot hold the needle anywhere on the slide')
    return best[1:]


def dashed(draw, a, b):
    """A dashed line from pixel a to pixel b."""
    length = np.linalg.norm(b-a); unit = (b-a)/length
    for s in np.arange(0, length, DASH+SPACE):
        draw.line([tuple(a+s*unit), tuple(a+min(s+DASH, length)*unit)], fill=tuple(int(255*c) for c in GOLD), width=WIDTH)


def block(S, d, bunnies, radius, back, slot):
    """Everything within radius of the line of action from back before S on through the table, and everything
    below that down to the table, minus every bunny grown by INFLATE (slot: minus its whole sweep back along the
    needle, the way it slides in and out); its largest piece."""
    u, v = basis(d); a = np.linspace(0, 2*np.pi, 97)[:-1]
    ring = radius*(np.cos(a)[:, None]*u+np.sin(a)[:, None]*v)
    far = S+(S[2]+radius)/-d[2]*d                                      # the whole ring under the table here
    top = np.vstack([S-back*d+ring, far+ring])
    cut = md.Manifold.hull_points(np.vstack([top, top*[1, 1, 0]])/G.SCALE).trim_by_plane([0, 0, 1.], 0.)
    for m in bunnies:
        grown = trimesh.Trimesh(m.vertices+INFLATE*m.vertex_normals, m.faces, process=False)
        cut = cut-(exact_sweep(grown, -d, back+radius+.1) if slot else G.solid(grown))
    return G.mesh(max(cut.decompose(), key=lambda p: p.volume()))


def freedom(support, bunny, d):
    """Contact area of the support on the bunny (faces within CONTACT), and the widest angle from the way back
    along the needle (-d) of any direction the bunny can still translate in without pressing into a contact
    (by more than 1 degree): below 90 the bunny can only leave back towards the needle, it is held."""
    near = cKDTree(trimesh.sample.sample_surface_even(bunny, 400000, seed=0)[0]).query(support.triangles_center)[0] < CONTACT
    k = np.arange(4000)+.5; z = 1-2*k/4000; phi = k*np.pi*(3-5**.5)
    ways = np.c_[np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z]
    free = ways[(ways@support.face_normals[near].T > -np.sin(np.radians(1))).all(1)]
    return support.area_faces[near].sum(), np.degrees(np.arccos(np.clip(free@-d, -1, 1)).max()) if len(free) else 0.


def inside(points, a, b, radius):
    """points within radius of the segment ab."""
    ab = b-a; s = np.clip((points-a)@ab/(ab@ab), 0, 1)
    return np.linalg.norm(points-a-s[:, None]*ab, axis=1) < radius


def main(support, mark):
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    # Station frame moved so the tabletop is z = 0 and the work point is over the origin.
    station = np.r_[setup['work_point_m'][:2], setup['table_height_m']]
    setup['base_position_world_m'] = (np.asarray(setup['base_position_world_m'])-station).tolist()
    reach = -np.asarray(setup['base_position_world_m'])[:2]; reach /= np.linalg.norm(reach)
    slope = np.radians(SLOPE); d = np.r_[np.cos(slope)*reach, -np.sin(slope)]
    setup['working_direction_world'] = d.tolist()
    camera = np.degrees(np.arctan2(reach[1], reach[0]))+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE]; view /= np.linalg.norm(view)

    chosen, C = poses(source, d)
    low = min(((source.vertices-exit)@R[2]).min() for _, R, _, exit in chosen.values())
    S = np.r_[(C+GAP)*d[:2], LIFT-low]                                      # needle tip over the origin
    tip = S-(C+GAP)*d
    foot = S+(S[2]/np.sin(slope)+RADIUS)*d
    placed = {}
    for name, (off, R, entry, exit) in chosen.items():
        W = np.eye(4); W[:3, :3] = R; W[:3, 3] = S-R@exit; m = source.copy(); m.apply_transform(W)
        area = (np.linalg.norm(source.triangles_center-entry, axis=1) < PATCH) & (source.face_normals@(R.T@d) < 0)
        placed[name] = (W, m, area, GAP+C-np.linalg.norm(exit-entry), off)
    if support == 'rod':
        held = trimesh.creation.capsule(np.linalg.norm(foot-S), RADIUS, count=[48, 48])
        held.apply_transform(tool_frame((S+foot)/2, d)); end = S; spread = [foot[:2]]
    else:
        radius, back = (BLOCK, BACK) if support == 'block' else (SLOT, SLOT_BACK)
        held = block(S, d, [m for _, m, *_ in placed.values()], radius, back, support == 'slot')
        end = S+S[2]/-d[2]*d; spread = [held.vertices[:, :2]]
    xy = np.vstack([m.vertices[:, :2] for _, m, *_ in placed.values()]+spread)
    table = trimesh.creation.box(bounds=[[*(xy.min(0)-.02), -.015], [*(xy.max(0)+.02), 0.]])

    arm, links, q, slide = holder(setup, tip, d)
    L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    held_samples = trimesh.sample.sample_surface_even(held, 6000 if support == 'rod' else 40000, seed=0)[0]
    fixed = np.vstack([held_samples, trimesh.sample.sample_surface_even(table, 20000, seed=0)[0]])
    contact = {}
    for name, (W, m, *_) in placed.items():
        bunny = trimesh.sample.sample_surface_even(m, 60000, seed=0)[0]; samples = np.vstack([bunny, fixed])
        hits = [i for i in range(1, 8)
                if Delaunay(arm.collisions[i]).find_simplex((samples-links[i][:3, 3])@links[i][:3, :3]).max() >= 0]
        tool = np.vstack([bunny, held_samples])
        clash = dict(needle=inside(tool, tip, tip-L*d, N).sum(), head=inside(tool, tip-L*d, tip-(L+.07)*d, H).sum(),
                     support=(inside(bunny, S, foot, RADIUS-.0003) if support == 'rod' else held.contains(bunny)).sum())
        if hits or any(clash.values()) or m.bounds[0, 2] < LIFT-1e-9:
            raise ValueError(f'{name}: KUKA links {hits}, samples in {clash}, lowest {1e3*m.bounds[0, 2]:.1f} mm')
        contact[name] = freedom(held, m, d)

    panels = []
    fit = np.vstack([*(m.vertices for _, m, *_ in placed.values()), table.vertices, [tip, tip-(L+.07)*d]])
    for name, (W, m, area, *_) in placed.items():
        scene = Scene(source, setup, *SIZE, arm=True)
        try:
            r = scene.r
            scene.view = view; scene.ctx.line_width = OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]; scene.fit(fit, 1.04)
            shapes = [(r.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), roughness=.85), np.eye(4)),
                      (r.Shape(scene.ctx, scene.shader, held.vertices, held.faces, BLUE, metallic=.05, roughness=.5), np.eye(4))]
            shapes += [(part, tool_frame(tip, d)) for part in scene.tools[False]]
            colors = np.tile(r.GRAY, (len(source.faces), 1)); colors[area] = r.PINK
            outline = task_outline(source, area, W)
            if mark == 'cone':
                gold = r.Shape(scene.ctx, scene.shader, *tubes(dashes(cone(tip-L*d, d, L, CONE)), .0005), GOLD,
                               metallic=.1, roughness=.5)
                compositor = Opacity(scene)
                try: data = compositor.draw(W, colors, shapes, links, [], cones=[(gold, np.eye(4))],
                                            cone_opacity=CONE_OPACITY, segments=outline)
                finally: compositor.close()
            else:
                data = scene.draw(W, colors, [], links=links, shapes=shapes, segments=outline)
            ends = np.c_[tip, end].T@scene.vp[:3, :3].T+scene.vp[:3, 3]
        finally: scene.close()
        panel = Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if mark == 'line': dashed(ImageDraw.Draw(panel), *np.c_[(ends[:, 0]+1)/2*SIZE[0], (1-ends[:, 1])/2*SIZE[1]])
        panels.append(panel)
    rows = -(-len(panels)//COLUMNS)
    out = Image.new('RGB', (COLUMNS*SIZE[0]+(COLUMNS+1)*GAP_PX, rows*SIZE[1]+(rows+1)*GAP_PX), 'white')
    for k, p in enumerate(panels): out.paste(p, (GAP_PX+(k % COLUMNS)*(SIZE[0]+GAP_PX), GAP_PX+(k//COLUMNS)*(SIZE[1]+GAP_PX)))
    path = HERE/f'slanted_{support}{"_cone" if mark == "cone" else ""}.png'; out.save(path)
    print('RENDERED', path, f'chord {1e3*C:.0f} mm; S {1e3*S[2]:.0f} mm above the table;',
          f'support {1e6*held.volume:.1f} cm3, watertight {held.is_watertight};',
          f'arm {1e3*slide:.0f} mm in; joints {np.degrees(q).round(1).tolist()} deg')
    for name, (W, m, area, gap, off) in placed.items():
        print(f'  {name}: centre of mass {1e3*off:.1f} mm off the vertical through S, needle gap {1e3*gap:.1f} mm,',
              f'work area {100*source.area_faces[area].sum()/source.area:.2f}%, lowest {1e3*m.bounds[0, 2]:.0f} mm,',
              f'contact {1e4*contact[name][0]:.2f} cm2, free up to {contact[name][1]:.0f} deg from the way back')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--support', choices=['rod', 'block', 'slot'], default='slot')
    parser.add_argument('--mark', choices=['line', 'cone'], default='cone'); args = parser.parse_args()
    main(args.support, args.mark)
