"""The comfortable place of a station's work: where its arm does the most of the job, with the most room.

The job at a work point p is every needle tip in the work cube around p (the configured edge; the largest
drawn edge when unbounded), with the needle anywhere in the tilt cone, also backed off its own length along
its axis, each tip solved on one joint branch: from the arm's joints at p with the nominal needle.  A work
point is first judged by the share of its job the arm does (a weak arm may do all of it nowhere), then, among
those doing all of it, by its room: the half-edge of the largest cube whose whole job the arm does.  The joint
limits, the reach and the floor all bound the room, so the work point with the most room keeps the work far
from every one of them: not folded at a joint limit, not stretched at the end of the reach.

The base stands at the origin facing p.  p is searched by its distance from the base, its height and the
bearing of the needle from the base->p line, on a grid of the arm's reach plus the tool, then refined around the best; the
base must stand off the table (a square of table_half_size_m around p).

  python -m robots_config.place robots_config/random/03_kuka.json [--write]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from pathlib import Path

import numpy as np

from robots_config.constraints import load_config

NEEDLE = .1             # m, the needle's length: how far it backs off along its axis
RING = 6                # needle axes on the rim of the tilt cone, besides the nominal one
LEVELS = 6              # bisection steps of the room
GRID = 8                # coarse steps across the reach, in distance and in height
BEARINGS = (-90., -45., 0., 45., 90.)   # deg, the needle's horizontal bearing from base->p
UP = np.array([0., 0., 1.])

_arm = None


def arm_of(config):
    f = config['feasibility']; module = f['arm_module'].removesuffix('.py').replace('/', '.')
    return importlib.import_module(module).Arm(np.zeros(3), np.eye(3), tool=f['tcp_transform']['translation_flange_m'][2])


def _init(config):
    global _arm
    _arm = arm_of(config)


def facing(w):
    """The base rotation with its +X along the horizontal direction w."""
    w = np.array([w[0], w[1], 0.]); w /= np.linalg.norm(w)
    return np.column_stack([w, np.cross(UP, w), UP])


def job(a, tilt, half):
    """(tip offsets, needle axes, back-offs) of the work cube of half-edge `half`, hardest first."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)*half if half else []
    e = np.cross(a, UP if abs(a[2]) < .9 else [1., 0., 0.]); e /= np.linalg.norm(e); f = np.cross(a, e)
    t = np.radians(tilt); phi = 2*np.pi*np.arange(RING)/RING
    axes = [np.cos(t)*a+np.sin(t)*(np.cos(p)*e+np.sin(p)*f) for p in phi]+[a]
    return [(c, d, b) for b in (NEEDLE, 0.) for d in axes for c in [*corners, np.zeros(3)]]


def room(task):
    """(share of the job done, room) at one work point (room 0 unless all of it), and the joints at p."""
    p, a, tilt, half, top, R = task; _arm.rotation = R
    try: q0 = _arm.solve(p, -a)
    except ValueError: return (0., 0.), None
    def share(half, every=False):
        tips = job(a, tilt, half); done = 0
        for c, d, b in tips:
            try: _arm.solve(p+c-b*d, -d, q0); done += 1
            except ValueError:
                if not every: return 0.
        return done/len(tips)
    whole = share(half, every=True)
    if whole < 1: return (whole, 0.), q0
    low, high = half, top
    for _ in range(LEVELS):
        middle = (low+high)/2; low, high = (middle, high) if share(middle) == 1 else (low, middle)
    return (1., low), q0


def place(config, workers=22):
    """The comfortable work point, base rotation and its record, for a schema-1 station config."""
    task, f = config['task_constraint'], config['feasibility']
    a = np.asarray(task['working_direction_world'], float); tilt = float(task['cone_half_angle_deg'])
    edge = task['surface_workspace']['edge_m']
    if edge is None: edge = max(e for e in config['draw']['edges_m'] if e is not None)
    reach = float(config['robot_specification']['reach_m']); table = float(f['table_half_size_m'])
    u = np.array([a[0], a[1], 0.]); u = u/np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1., 0., 0.])
    foot = float(np.linalg.norm(arm_of(config).collisions[0][:, :2], axis=1).max())

    def point(r, z, bearing):
        c, s = np.cos(np.radians(-bearing)), np.sin(np.radians(-bearing))
        w = np.array([c*u[0]-s*u[1], s*u[0]+c*u[1], 0.])         # base->p: the needle's bearing turned back
        return r*w+z*UP, facing(w)

    def search(pool, candidates):
        ok = [(r, z, b) for r, z, b in candidates if r > 0 and z >= 0 and np.abs(point(r, z, b)[0][:2]).max() >= table+foot]
        tasks = [(p, a, tilt, edge/2, reach/2, R) for p, R in (point(r, z, b) for r, z, b in ok)]
        return [(h, c, q) for c, (h, q) in zip(ok, pool.map(room, tasks))]

    span = reach+f['tcp_transform']['translation_flange_m'][2]; step = span/GRID; turn = 45.
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(config,)) as pool:
        found = search(pool, [(r, z, b) for r in step*np.arange(1, GRID+1) for z in step*np.arange(GRID+1) for b in BEARINGS])
        best = max(found, key=lambda x: x[0])
        for _ in range(2):
            step /= 2; turn /= 2; r0, z0, b0 = best[1]
            found = search(pool, [(r0+i*step, z0+j*step, np.clip(b0+k*turn, -90, 90))
                                  for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
            best = max([best, *found], key=lambda x: x[0])
        (done, h), (r, z, b), q = best
        p, R = point(r, z, b)
        current = np.asarray(task['world_position_constraint']['center_world_m'], float)
        R0 = np.asarray(f['base_pose_world']['rotation_matrix'], float)
        (done_before, before), q_before = list(pool.map(room, [(current, a, tilt, edge/2, reach/2, R0)]))[0]
    record = dict(job_share=done, room_half_edge_m=h, distance_m=float(r), height_m=float(z), needle_bearing_deg=float(b),
                  joints_at_work_point_deg=np.degrees(q).round(2).tolist(), job_cube_edge_m=edge, needle_back_off_m=NEEDLE,
                  previous=dict(center_world_m=current.tolist(), job_share=done_before, room_half_edge_m=before,
                                joints_at_work_point_deg=None if q_before is None else np.degrees(q_before).round(2).tolist()),
                  method='work point doing the largest share of the job (tips in the work cube, the needle anywhere in the '
                         'tilt cone and backed off its length, on one joint branch from the work point), then with the most '
                         'room (the largest such cube done whole); grid over distance, height and needle bearing, refined '
                         'twice (robots_config.place)')
    return p, R, record


def placed(config, workers=22):
    """The config with the comfortable work point and the base facing it."""
    p, R, record = place(config, workers)
    w = config['task_constraint']['world_position_constraint']
    w.update(center_world_m=p.tolist(), base_to_center_distance_m=float(np.linalg.norm(p)),
             rule='the comfortable work point of robots_config.place; Step1 slides the work point forward from here')
    config['feasibility']['base_pose_world'].update(rotation_matrix=R.tolist(), provenance='base +X toward the comfortable work point')
    config['feasibility']['placement'] = record
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='+', type=Path); parser.add_argument('--write', action='store_true')
    parser.add_argument('--workers', type=int, default=22)
    args = parser.parse_args()
    for path in args.configs:
        config = placed(json.loads(path.read_text()), args.workers); r = config['feasibility']['placement']
        print(path.name, 'job', round(r['previous']['job_share'], 3), '->', round(r['job_share'], 3),
              'room', round(r['previous']['room_half_edge_m']*1000), '->', round(r['room_half_edge_m']*1000), 'mm;',
              'distance', round(r['distance_m'], 3), 'height', round(r['height_m'], 3), 'bearing', r['needle_bearing_deg'],
              'joints', r['joints_at_work_point_deg'], flush=True)
        if args.write: path.write_text(json.dumps(config, indent=2)+'\n'); load_config(path)
