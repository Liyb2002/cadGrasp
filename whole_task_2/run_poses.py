"""Step1 -> Step3 -> Step4 keep-out -> Step4 combine -> Step5 -> evaluation for one object at N tabletop
poses, on a station from robots_config/, as an entry around the unchanged core (step1/plan_reach.py,
step3/tool_membranes.py, step4/keepout.py, step4/combine.py, step5/combined.py, evaluation/combine.py).
The run is <object>+<N>pose: data under data/<run>/, figures (and Step5's rNN/ packages) under output/<run>/,
as run_station.py's.

  python -m whole_task_2.run_poses --poses 8 --config robots_config/random/06_so101.json
  python -m whole_task_2.run_poses --poses 8 --config robots_config/random/06_so101.json --start step4_keepout --until step4_keepout

What the entry sets, everything else is the core's:
  - poses: N object sides spread evenly on the sphere (Fibonacci), each turned toward the robot
    horizontally and lowered onto the table as the core places its six (+Y up, or -Z/+Z for sides near
    +Y/-Y), the up direction made perpendicular to the side.  Task ids S1..SN.
  - the work arm: the config's arm module (feasibility.arm_module: KUKA or SO101).  An arm with fewer than
    seven joints sits behind the core's seven-joint interface, its joints then zeros, and is drawn with its
    own meshes.
  - the work cube: the config's surface_workspace edge (unbounded: the core's default, half the object's
    longest edge), placed as the core places it; the work point is where the arm reaches every tip within
    that cube (half its edge each way) instead of within half the object (plan.json's work_point_rule
    text is the core's).
  - the object's mesh for the run name.
The core validation tests two defaults of the six-side station literally (six poses; cube edge half the
object); the entry tests their meaning for the run instead (the N poses as placed; the configured edge).
The patches are set when this module loads, so the spawned workers of Step3 and Step4 have them too.
"""
import argparse
import importlib
import json
import shutil
import time
from pathlib import Path
import numpy as np

from robots_config.constraints import load_config
from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, object_of, load_shape, write_json
from whole_task_2.step1 import plan_reach as P, reach_video as V

_kuka, _scene, _work_point, _setup, _surface = V.kuka, V.Scene.__init__, V.work_point, P.setup, P.surface_workspace
EDGE = None                     # m, the configured work cube edge (None: the core's default)
REPLACED = {'six_tabletop_poses_match': 'tabletop_poses_match',
            'workspace_cube_edge_is_half_longest_object_aabb_edge': 'workspace_cube_edge_is_configured_edge'}


class Padded:
    """An arm with fewer than seven joints behind step1.arm's and plan.npz's seven joints: its own, then zeros."""
    dof = 7

    def __init__(self, arm):
        self.arm = arm; self.n = len(arm.lower); pad = np.zeros(7-self.n)
        self.lower, self.upper = np.r_[arm.lower, pad], np.r_[arm.upper, pad]
        self.collisions, self.base = arm.collisions, arm.base

    def forward(self, q, geometry=False):
        return self.arm.forward(np.asarray(q)[:self.n], geometry)

    def solve(self, point, outward, previous=None):
        q = self.arm.solve(point, outward, None if previous is None else np.asarray(previous)[:self.n])
        return np.r_[q, np.zeros(7-self.n)]

    def visuals(self):
        return self.arm.visuals()


def work_arm(setup):
    """The config's work arm at the station's base, carrying the head: flange 70 mm behind the needle root."""
    module = setup.get('arm_module')
    if module is None: return _kuka(setup)
    Arm = importlib.import_module(module.removesuffix('.py').replace('/', '.')).Arm
    arm = Arm(np.asarray(setup['base_position_world_m'], float), np.asarray(setup['base_rotation_world'], float),
              tool=setup['needle_length_m']+.07)
    return arm if len(arm.lower) == 7 else Padded(arm)


def scene(self, mesh, setup, width, height, arm=False):
    """The core's studio; with an arm, the config's arm drawn from its own meshes."""
    _scene(self, mesh, setup, width, height, arm)
    if arm and setup.get('arm_module'):
        self.arm = [(i, self.r.Shape(self.ctx, self.shader, v, f, tuple(colour), metallic=.25, roughness=.4))
                    for i, v, f, colour in work_arm(setup).visuals()]


def work_point(setup, radius, *args, **options):
    """The core's work point; with a configured cube, every tip within the cube instead of half the object."""
    return _work_point(setup, radius if EDGE is None else EDGE/2, *args, **options)


def shape(path):
    """A run's mesh is its object's: objects/<object>+<station>/x -> objects/<object>/x."""
    path = Path(path)
    if not path.exists() and path.parent.parent == ROOT/'objects': path = ROOT/'objects'/object_of(path.parent.name)/path.name
    return load_shape(path)


def setup(object_name='B', config_path=None, target_path=None):
    """The core's setup for the run's object, with the config's arm module."""
    mesh, path, target, s = _setup(object_of(object_name), config_path, target_path)
    s['arm_module'] = load_config(config_path)['feasibility'].get('arm_module') if config_path else None
    return mesh, path, target, s


def surface_workspace(mesh, reference_pose, size_fraction=.5, base_position_world=(0., 0., 0.)):
    """The core's work cube, its edge the configured one when there is one."""
    if EDGE is not None:
        R = np.asarray(reference_pose['rotation_world_object'], float)
        size_fraction = EDGE/float(np.ptp(mesh.vertices@R.T, axis=0).max())
    return _surface(mesh, reference_pose, size_fraction, base_position_world)


def sides(n):
    """n object directions spread evenly on the sphere (Fibonacci)."""
    k = np.arange(n)+.5; z = 1-2*k/n; r = np.sqrt(1-z*z); phi = k*np.pi*(3-5**.5)
    return np.c_[r*np.cos(phi), r*np.sin(phi), z]


def placed(mesh, views, s):
    """Each side toward the robot horizontally, then lowered onto the table (workspace.tabletop_placements,
    with the up direction made perpendicular to the side)."""
    front = -np.asarray(s['working_direction_world'], float); front[2] = 0
    front = np.array([0., 1., 0.]) if np.linalg.norm(front) < 1e-9 else front/np.linalg.norm(front)
    basis = np.column_stack([np.cross(front, [0., 0., 1.]), front, [0., 0., 1.]])
    center = mesh.bounds.mean(0); point = np.asarray(s['work_point_m']); out = []
    for side in np.asarray(views, float):
        up = np.array([0., 1., 0.]) if abs(side[1]) < .5 else np.array([0., 0., -np.sign(side[1])])
        up = up-(up@side)*side; up /= np.linalg.norm(up)
        R = basis@np.column_stack([np.cross(side, up), side, up]).T
        shift = point-R@center; shift[2] = s['table_height_m']-(mesh.vertices@R.T)[:, 2].min()
        out.append(dict(rotation_world_object=R.tolist(), translation_world_m=shift.tolist()))
    return out


V.kuka = work_arm; V.Scene.__init__ = scene; V.work_point = work_point
P.load_shape = shape; P.setup = setup; P.surface_workspace = surface_workspace; P.placed = placed


def validate(name, n):
    """The core validation; its two six-side defaults tested for the run instead."""
    try: return P.validate(name)
    except ValueError: pass
    work = ROOT/'whole_task_2/data'/name/'step1_reach'; v = json.loads((work/'validation.json').read_text())
    report = json.loads((work/'plan.json').read_text()); s = report['setup']
    failed = [k for k, ok in v['checks'].items() if not ok]
    if set(failed)-set(REPLACED): raise ValueError('Step1 validation failed: '+', '.join(failed))
    expected = placed(shape(ROOT/'objects'/name/'mesh.stl'), P.VIEWS, s)
    b = np.asarray(s['surface_workspace']['bounds_world_m'])
    new = dict(tabletop_poses_match=bool(report['task_count'] == n and [t['id'] for t in report['tasks']] == list(P.SIDES) and
                   all(np.allclose(t['placement'][k], e[k], atol=1e-12) for t, e in zip(report['tasks'], expected) for k in e)),
               workspace_cube_edge_is_configured_edge=bool(EDGE is None or np.allclose(b[1]-b[0], EDGE, atol=1e-12, rtol=0)))
    checks = {k: ok for k, ok in v['checks'].items() if k not in REPLACED}|new
    v.update(status='passed' if all(checks.values()) else 'failed', checks=checks, checks_replaced_by_entry=REPLACED,
             entry='whole_task_2/run_poses.py')
    write_json(work/'validation.json', v)
    if v['status'] != 'passed': raise ValueError('Step1 validation failed: '+json.dumps(new))
    print('STEP1 VALIDATION PASSED (entry)', flush=True); return v


STAGES = ('step1', 'step3', 'step4_keepout', 'step4_combine', 'step5', 'evaluation')


def run(n, config, object_name='B', workers=6, start='step1', until='step3', combine_workers=22):
    global EDGE
    config = (ROOT/config).resolve(); name = f'{object_name}+{n}pose'; output = DEFAULT_OUTPUT
    EDGE = load_config(config)['task_constraint'].get('surface_workspace', {}).get('edge_m')
    P.VIEWS = sides(n); P.SIDES = tuple(f'S{k+1}' for k in range(n))
    # The tool-axis screen depends on the object and the tool only: the object's own screen is a valid cache (key checked).
    work = ROOT/'whole_task_2/data'/name/'step1_reach'; work.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'whole_task_2/data'/object_name/'step1_reach/screen.npz'
    if cache.exists() and not (work/'screen.npz').exists(): shutil.copy2(cache, work/'screen.npz')
    for stage in STAGES[STAGES.index(start):STAGES.index(until)+1]:
        began = time.monotonic()
        if stage == 'step1':
            P.plan(name, config, None, output_root=output); validate(name, n)
        elif stage == 'step3':
            from whole_task_2.step3.tool_membranes import run as step3
            step3(output, name, workers)
        elif stage == 'step4_keepout':
            from whole_task_2.step4 import keepout
            from whole_task_2.step4.render_keepout import render
            keepout.run(name, workers); render(name, output/name/'step4/keepout.png')
        elif stage == 'step4_combine':
            from whole_task_2.step4.combine import run as combine
            combine(name, combine_workers)
        elif stage == 'step5':
            from whole_task_2.step5.combined import run as step5
            step5(name, output)
        else:
            from whole_task_2.evaluation.combine import run as evaluate
            evaluate(name, 40)
        print(name, stage, round(time.monotonic()-began), 's', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses', type=int, required=True); p.add_argument('--config', required=True)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--combine-workers', type=int, default=22)
    p.add_argument('--start', default='step1', choices=STAGES); p.add_argument('--until', default='step3', choices=STAGES)
    a = p.parse_args(); run(a.poses, a.config, a.object, a.workers, a.start, a.until, a.combine_workers)
