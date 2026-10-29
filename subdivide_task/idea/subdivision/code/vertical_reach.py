"""Why a work arm with the needle always straight down is a poor choice (idea figure): it misses much of the object.

The bunny takes any pose: --euler gives turns about the world x, y and z axes, in degrees, applied after
easy_mode.py's upright pose.  The needle always points straight down, and the arm (the KUKA rising and falling with
the object) brings it anywhere.  Pink is every face the vertical needle reaches, by Step1's exact check
(step1/reach.py certify: needle line, needle body and head clear, the face turned to the needle).  Grey is the rest.
One needle is drawn, touching a pink face (the KUKA at station 07's base, real IK).  The bunny rests on the table.
No text.

  PYTHONPATH=. .venv/bin/python -B subdivide_task/idea/subdivision/code/vertical_reach.py [--euler 0 0 0] [--output name.png]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from subdivide_task.step1.common import ROOT, load_shape, sha256, write_json
from subdivide_task.step1 import plan_reach as P, reach as R
from subdivide_task.step1.subdivide import rotation
from subdivide_task.step1.reach_video import Scene, kuka, work_point, tool_frame
from subdivide_task.step1.find_tasks_figure import picture, task_outline
from subdivide_task.step5.video.prepare import matrices
from easy_mode import CONFIG, UP, TURN_DEG, VIEW

HERE = Path(__file__).parent
WORKERS = 4
FLAT = .8                                   # the drawn needle touches a pink face this level at least (world normal z),
                                            # the one nearest the centre of mass seen from above


def reached(shape, mesh, s, d, workers=WORKERS):
    """Faces the needle along d (object frame) reaches, by Step1's exact check."""
    eps = json.loads((ROOT/'subdivide_task/step1/reach_policy.json').read_text())['needle_start_fraction_of_length']*s['needle_length_m']
    ids = np.flatnonzero(-(mesh.face_normals@d) > R.facing(s['max_incidence_deg']))
    jobs = [(int(f), [(0, 0, d)], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'], eps, s['max_incidence_deg'])
            for f in ids]
    out = np.zeros(len(mesh.faces), bool)
    with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(shape),)) as pool:
        for face, pose, *_ in pool.map(P._certify, jobs, chunksize=16): out[face] = pose is not None
    return out


def render(euler=(0., 0., 0.), output=HERE.parent/'vertical_reach.png', height=1400):
    _, shape, _, s = P.setup('B', CONFIG); mesh = load_shape(shape)
    s['working_direction_world'] = [0., 0., -1.]                     # the needle always straight down
    a = np.array([0., 0., -1.])
    edge = float(json.loads(CONFIG.read_text())['task_constraint']['surface_workspace']['edge_m'])
    point, _ = work_point(s, edge/2); point = np.asarray(point, float)
    M = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()@rotation(UP, TURN_DEG); view = VIEW/np.linalg.norm(VIEW)
    paint = reached(shape, mesh, s, M.T@a)
    # One needle on a level pink face; the bunny there, resting on the table.
    level = paint & (mesh.face_normals@M.T[:, 2] >= FLAT); ids = np.flatnonzero(level if level.any() else paint)
    face = int(ids[np.argmin(np.linalg.norm(((mesh.triangles_center[ids]-mesh.center_mass)@M.T)[:, :2], axis=1))])
    t = point-M@mesh.triangles_center[face]
    world = mesh.copy(); world.apply_transform(np.r_[np.c_[M, t], [[0, 0, 0, 1]]]); table = float(world.bounds[0][2])
    arm = kuka(s); q = arm.solve(point, -a); links = matrices(arm.forward(q, geometry=True)[2]); links[:, :3, 3] -= point
    hulls = [ConvexHull(v) for v in arm.collisions]; inside = 0
    for i in range(2, 8):
        local = (world.vertices-point-links[i][:3, 3])@links[i][:3, :3]
        inside += int(np.all(local@hulls[i].equations[:, :3].T+hulls[i].equations[:, 3] <= 0, axis=1).sum())
    # Draw, everything shifted by the work point (the scene's lights are about the origin).
    W = np.eye(4); W[:3, :3] = M; W[:3, 3] = t-point
    scene = Scene(mesh, s, int(height*1.1), height, arm=True); r = scene.r; scene.view = view
    colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[paint] = r.PINK
    try:
        slab = trimesh.creation.box(bounds=[np.r_[-.25, -.25, table-point[2]-.02], np.r_[.25, .25, table-point[2]]])
        shapes = [(r.Shape(scene.ctx, scene.shader, slab.vertices, slab.faces, (.91, .91, .91), roughness=.85), np.eye(4))]
        shapes += [(part, tool_frame(np.zeros(3), a)) for part in scene.tools[False]]
        fit = list(world.vertices-point)+[np.zeros(3), -(s['needle_length_m']+.07)*a]
        image = picture(scene, W, colors, shapes, fit, links, [], 1., None, segments=task_outline(mesh, paint, W))
    finally:
        scene.close()
    output = Path(output); image.save(output); area = mesh.area_faces
    record = dict(output=str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output), sha256=sha256(output), source_sha256=sha256(Path(__file__)),
                  station=str(CONFIG.relative_to(ROOT)), needle='always straight down, no tilt',
                  euler_xyz_world_deg=list(map(float, euler)), rotation_world_object=M.tolist(),
                  reached_area_fraction=float(area[paint].sum()/area.sum()), reached_faces=int(paint.sum()),
                  check='step1/reach.py certify on every face turned to the needle: needle line, body and head clear',
                  contact_face=face, arm_joints_deg=np.degrees(q).tolist(), arm_vertices_hit=inside,
                  arm_check='bunny vertices inside the convex hulls of KUKA links 2-7')
    write_json(HERE/(output.stem+'.json'), record)
    print(output, 'reached %.1f%% of the surface' % (100*record['reached_area_fraction']), flush=True)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--euler', type=float, nargs=3, default=(0., 0., 0.), metavar=('X', 'Y', 'Z'))
    p.add_argument('--output', default='vertical_reach.png', help='file name in idea/subdivision/')
    a = p.parse_args(); render(a.euler, HERE.parent/a.output)
