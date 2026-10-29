"""Easy mode (idea figure): the needle straight down, so the simplest tray holds the object.

When the needle presses straight down, every load on the object is vertical: its weight and the push.  The object
then just sits in a plain tray on the table, a flat floor and a low rim.

Drawn: the bunny upright (+Y up, as in step1/find_tasks_figure.py), resting on the floor of a round TRAY on the
table.  Pink is the small patch the needle touches: the faces within PATCH of the tip, which Step1's saved screen
(data/B/step1_reach/screen.npz) says the needle reaches straight down.  The KUKA at station 07's base holds the
needle straight down there (real IK).  An illustration, not a Step3-5 support.  No text.

  PYTHONPATH=. .venv/bin/python -B subdivide_task/idea/subdivision/code/easy_mode.py
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from subdivide_task.step1.common import ROOT, load_shape, sha256, write_json
from subdivide_task.step1 import plan_reach as P, reach as R
from subdivide_task.step1.subdivide import rotation
from subdivide_task.step1.reach_video import Scene, kuka, work_point, tool_frame
from subdivide_task.step1.find_tasks_figure import picture, task_outline
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.render_independent import BLUE
from subdivide_task.step5.video.prepare import matrices

HERE = Path(__file__).parent
OUTPUT = HERE.parent/'easy_mode.png'
CONFIG = ROOT/'robots_config/random/07_kuka.json'
SCREEN = ROOT/'subdivide_task/data/B/step1_reach/screen.npz'
UP = np.array([0., 1., 0.])                 # the bunny's up (a Y-up mesh)
TURN_DEG = -60.                             # the bunny turned about the vertical, for the view
VIEW = np.array([.35, 1., .45])             # camera direction (towards the eye)
HIND = .6                                   # the needle touches the back (beyond this share of x: the hindquarters,
                                            # away from the ears), the face turned most between up and the camera
PATCH = .008                                # pink: faces within this of the tip, turned like its face (60 deg)
TRAY = (.004, .004, .014, .006)             # the tray: floor and rim thickness, rim height over the floor, gap to the object


def screen(mesh, shape, s):
    """Step1's saved screen (axes x faces), checked against the key P.screen would make."""
    policy = json.loads((ROOT/'subdivide_task/step1/reach_policy.json').read_text())
    axes = R.sphere_axes(policy['tool_axis_spacing_deg'])
    pixel = policy['screen_pixel_fraction_of_diagonal']*float(np.linalg.norm(mesh.extents))
    samples = int(policy['screen_samples_per_pixel']*mesh.area/pixel**2)
    key = json.dumps(dict(shape=sha256(shape), reach=sha256(ROOT/'subdivide_task/step1/reach.py'), length=s['needle_length_m'],
                          radius=s['head_radius_m'], needle_radius=s['needle_radius_m'], incidence=s['max_incidence_deg'],
                          axes=len(axes), pixel=pixel, samples=samples), sort_keys=True)
    with np.load(SCREEN) as z:
        if str(z['key']) != key: raise ValueError('the saved screen is for another object or tool')
        return axes, z['reach'].copy()


def disc(radius, low, high, xy, sections=64):
    m = trimesh.creation.cylinder(radius, high-low, sections=sections); m.apply_translation([xy[0], xy[1], (low+high)/2])
    return G.solid(m)


def tray(body):
    """A round tray under body: its floor's top at body's lowest point, its rim TRAY[3] clear of what body has below
    the rim's top, about the middle of that.  (mesh, centre xy, inner radius)"""
    floor, wall, rim, gap = TRAY; low = float(body.bounds[0][2]); V = body.vertices[body.vertices[:, 2] <= low+rim, :2]
    xy = (V.min(0)+V.max(0))/2; inner = float(np.linalg.norm(V-xy, axis=1).max())+gap
    value = disc(inner+wall, low-floor, low, xy, 128)+(disc(inner+wall, low, low+rim, xy, 128)-disc(inner, low-.001, low+rim+.001, xy, 128))
    return G.mesh(value), xy, inner


def contact(mesh, ids, M, view):
    """The face of ids turned most between up and the camera (world normals)."""
    b = np.array([0., 0., 1.])+view; return int(ids[np.argmax(mesh.face_normals[ids]@M.T@(b/np.linalg.norm(b)))])


def render(height=1400):
    _, shape, _, s = P.setup('B', CONFIG); mesh = load_shape(shape)
    s['working_direction_world'] = [0., 0., -1.]                     # easy mode: the needle straight down
    a = np.array([0., 0., -1.])
    edge = float(json.loads(CONFIG.read_text())['task_constraint']['surface_workspace']['edge_m'])
    point, _ = work_point(s, edge/2); point = np.asarray(point, float)
    axes, reach = screen(mesh, shape, s)
    # The needle along the screened axis nearest straight down (object frame: -UP), on the back; pink around it.
    M = rotation(UP, TURN_DEG); view = VIEW/np.linalg.norm(VIEW)
    hind = (mesh.triangles_center[:, 0]-mesh.bounds[0][0])/mesh.extents[0] >= HIND      # display landmark: a +X hindquarters
    k0 = int(np.argmax(axes@-UP)); face = contact(mesh, np.flatnonzero(reach[k0] & hind), M, view)
    tip0 = mesh.triangles_center[face]
    patch = (np.linalg.norm(mesh.triangles_center-tip0, axis=1) <= PATCH) & (mesh.face_normals@mesh.face_normals[face] >= .5)
    # The object: the tip at the work point, resting in the tray; the tray on the table.
    t = point-M@tip0
    world = mesh.copy(); world.apply_transform(np.r_[np.c_[M, t], [[0, 0, 0, 1]]])
    support, centre, inner = tray(world); table = float(support.bounds[0][2])
    # The arm holds the needle (real IK at the station's base, the needle straight down).
    arm = kuka(s); q = arm.solve(point, -a); links = matrices(arm.forward(q, geometry=True)[2]); links[:, :3, 3] -= point
    hulls = [ConvexHull(v) for v in arm.collisions]; inside = 0
    for i in range(2, 8):
        local = (np.vstack([world.vertices, support.vertices])-point-links[i][:3, 3])@links[i][:3, :3]
        inside += int(np.all(local@hulls[i].equations[:, :3].T+hulls[i].equations[:, 3] <= 0, axis=1).sum())
    # Draw, everything shifted by the work point (the scene's lights are about the origin).
    W = np.eye(4); W[:3, :3] = M; W[:3, 3] = t-point
    scene = Scene(mesh, s, int(height*1.1), height, arm=True); r = scene.r; scene.view = view
    colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[patch] = r.PINK
    shape_ = lambda v, f, color, **k: r.Shape(scene.ctx, scene.shader, v, f, color, **k)
    try:
        slab = trimesh.creation.box(bounds=[np.r_[-.25, -.25, table-point[2]-.02], np.r_[.25, .25, table-point[2]]])
        shapes = [(shape_(slab.vertices, slab.faces, (.91, .91, .91), roughness=.85), np.eye(4)),
                  (shape_(support.vertices-point, support.faces, BLUE, metallic=.05, roughness=.55), np.eye(4))]
        shapes += [(part, tool_frame(np.zeros(3), a)) for part in scene.tools[False]]
        L = s['needle_length_m']
        fit = list(world.vertices-point)+list(support.vertices-point)+[np.zeros(3), -(L+.07)*a]
        image = picture(scene, W, colors, shapes, fit, links, [], 1., None, segments=task_outline(mesh, patch, W))
    finally:
        scene.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True); image.save(OUTPUT)
    write_json(HERE/'easy_mode.json', dict(
        output=str(OUTPUT.relative_to(ROOT)), sha256=sha256(OUTPUT), source_sha256=sha256(Path(__file__)), screen_sha256=sha256(SCREEN),
        station=str(CONFIG.relative_to(ROOT)), needle='straight down (station 07 base and window, its direction replaced)',
        work_point_m=point.tolist(), up_object=UP.tolist(), turn_deg=TURN_DEG,
        contact_face=face, needle_axis_tilt_from_straight_down_deg=float(np.degrees(np.arccos(np.clip(axes[k0]@-UP, -1, 1)))),
        patch=dict(faces=int(patch.sum()), area_cm2=float(mesh.area_faces[patch].sum()*1e4), radius_mm=PATCH*1e3),
        tray=dict(floor_wall_rim_gap_mm=[x*1e3 for x in TRAY], inner_radius_mm=inner*1e3, centre_xy_world_m=centre.tolist(),
                  volume_cm3=float(support.volume*1e6), centre_of_mass_to_centre_mm=float(np.linalg.norm(world.center_mass[:2]-centre)*1e3),
                  rule='flat floor under the lowest point, rim TRAY[3] clear of the object below its top; illustration, not certified'),
        arm_joints_deg=np.degrees(q).tolist(), arm_vertices_hit=inside,
        arm_check='bunny and tray vertices inside the convex hulls of KUKA links 2-7'))
    return OUTPUT


if __name__ == '__main__':
    print(render())
