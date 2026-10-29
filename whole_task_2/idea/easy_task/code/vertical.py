"""The easiest task: one small pink work area on top, worked by a vertical needle on the KUKA, and one thin
pillar holding the bunny from below, both on the vertical through the centre of mass; no text.

The bunny stands upright (original +Y up) with its lowest vertex LIFT above the table.  The work area is the
upward faces within PATCH of the top of the vertical through the centre of mass; the needle tip hovers GAP
above that point, pointing straight down, and does not touch the bunny.  The pillar (radius RADIUS) rises
from the table to the bottom of that vertical, its top hidden inside the bunny.

The arm is B+03_kuka's KUKA LBR Med 14 at its station base, the table at the station's height (the drawn
tabletop is 2*TABLE wide).  The bunny's centre of mass sits on the line from the station's work point
towards the base, at most SLIDE in, where the arm holds the needle by IK with the most room to its joint
limits; its links are checked against dense samples of the bunny, pillar and tabletop.  The camera
looks across the arm's reach, turned TURN towards the far side, and frames the bunny, the head and the
wrist; the rest of the arm leaves the image.

Run from the repository root:  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/easy_task/code/vertical.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame, kuka, arm_links
from whole_task_2.step1.find_tasks_figure import task_outline, OUTLINE_WIDTH_PX
from whole_task_2.step3.render_independent import BLUE

HERE = Path(__file__).resolve().parents[1]
LIFT = .12          # m: the bunny's lowest vertex above the table
RADIUS = .004       # m: the pillar
PATCH = .018        # m: the work area's radius about the point under the needle
GAP = .012          # m: the needle tip above the work area
DOWN = np.array([0., 0., -1.])
TABLE = .10         # m: half the tabletop, small enough for its edges to show
FACE = 250.         # deg: camera azimuth in the upright bunny's frame (its face toward the camera)
SLIDE = .15         # m: how far towards the base the bunny may move from the work point
TURN = 25.          # deg: camera azimuth past square to the arm's reach, towards the far side
SIZE = (1200, 1500)


def first_hit(mesh, origins, direction):
    """z of the first surface point along direction from each origin."""
    hits, rays, _ = mesh.ray.intersects_location(origins, np.tile(direction, (len(origins), 1)))
    return np.array([(hits[rays == k, 2].max() if direction[2] < 0 else hits[rays == k, 2].min())
                     for k in range(len(origins))])


def spin(degrees):
    a = np.radians(degrees); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def holder(setup, tip):
    """The arm, moved towards the bunny along its reach in 1 cm steps up to SLIDE, holding the needle with the
    most room to its joint limits: (arm, links, joints, slide in m)."""
    base = np.asarray(setup['base_position_world_m']); unit = -base[:2]/np.linalg.norm(base[:2]); best = None
    for k in range(round(SLIDE/.01)+1):
        arm = kuka(dict(setup, base_position_world_m=(base+np.r_[k*.01*unit, 0.]).tolist()))
        try: links, q = arm_links(arm, tip, DOWN)
        except ValueError: continue
        room = np.min(np.minimum(q-arm.lower, arm.upper-q)/(arm.upper-arm.lower))
        if best is None or room > best[0]: best = (room, arm, links, q, k*.01)
    if best is None: raise ValueError('The arm cannot hold the needle anywhere on the slide')
    return best[1:]


def main():
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    setup['working_direction_world'] = DOWN.tolist()
    # Station frame moved so the tabletop is z = 0 and the work point is over the origin.
    station = np.r_[setup['work_point_m'][:2], setup['table_height_m']]
    setup['base_position_world_m'] = (np.asarray(setup['base_position_world_m'])-station).tolist()
    reach = -np.asarray(setup['base_position_world_m'])[:2]; reach = np.degrees(np.arctan2(reach[1], reach[0]))
    camera = reach+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), .45]; view /= np.linalg.norm(view)

    W = np.eye(4); W[:3, :3] = spin(camera-FACE)@[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # original +Y up
    placed = source.copy(); placed.apply_transform(W)
    shift = -placed.center_mass; shift[2] = LIFT-placed.bounds[0, 2]                   # COM over the origin
    W[:3, 3] = shift; placed.apply_translation(shift)
    com = placed.center_mass; top, bottom = com[2]+1, com[2]-1
    point = np.r_[com[:2], first_hit(placed, [[*com[:2], top]], DOWN)[0]]
    area = (np.linalg.norm(placed.triangles_center-point, axis=1) < PATCH) & (placed.face_normals[:, 2] > 0)
    tip = point-GAP*DOWN
    ring = np.linspace(0, 2*np.pi, 33)[:-1]; ring = com[:2]+RADIUS*np.c_[np.cos(ring), np.sin(ring)]
    seat = first_hit(placed, np.c_[ring, np.full(len(ring), bottom)], -DOWN)
    pillar = trimesh.creation.cylinder(RADIUS, segment=[[*com[:2], 0.], [*com[:2], seat.max()+1e-4]], sections=96)
    table = trimesh.creation.box(bounds=[[-TABLE, -TABLE, -.015], [TABLE, TABLE, 0.]])

    arm, links, q, slide = holder(setup, tip)
    samples = np.vstack([trimesh.sample.sample_surface_even(m, n, seed=0)[0]
                         for m, n in ((placed, 60000), (pillar, 4000), (table, 20000))])
    hits = [i for i in range(1, 8)
            if Delaunay(arm.collisions[i]).find_simplex((samples-links[i][:3, 3])@links[i][:3, :3]).max() >= 0]
    if hits: raise ValueError(f'KUKA links {hits} hit the bunny, pillar or table')

    scene = Scene(source, setup, *SIZE, arm=True)
    try:
        r = scene.r; L = setup['needle_length_m']
        scene.view = view; scene.ctx.line_width = OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]
        scene.fit(np.vstack([placed.vertices, table.vertices, [tip, tip-(L+.07)*DOWN, *links[6:, :3, 3]]]), 1.06)
        shapes = [(r.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), roughness=.85), np.eye(4)),
                  (r.Shape(scene.ctx, scene.shader, pillar.vertices, pillar.faces, BLUE, metallic=.05, roughness=.5), np.eye(4))]
        shapes += [(part, tool_frame(tip, DOWN)) for part in scene.tools[False]]
        colors = np.tile(r.GRAY, (len(source.faces), 1)); colors[area] = r.PINK
        data = scene.draw(W, colors, [], links=links, shapes=shapes, segments=task_outline(source, area, W))
    finally: scene.close()
    out = HERE/'vertical.png'
    Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(out)
    print('RENDERED', out, f'pillar {1e3*seat.min():.1f}-{1e3*seat.max():.1f} mm tall, radius {1e3*RADIUS:.0f} mm;',
          f'work area {100*placed.area_faces[area].sum()/placed.area:.2f}% of the surface;',
          f'needle tip {1e3*GAP:.0f} mm above it, {1e3*tip[2]:.1f} mm above the table; com {1e3*com[2]:.1f} mm;',
          f'bunny {1e3*slide:.0f} mm in from the work point; joints {np.degrees(q).round(1).tolist()} deg;',
          f'arm clear of {len(samples)} samples')


if __name__ == '__main__': main()
