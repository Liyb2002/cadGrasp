"""Text-free LBR Med assembly: grounded pivoting, Step6 insertion, static loads.

Robot poses follow the upstream seven-axis model and joint limits. The object
rolls on supporting vertices; its trajectory never leaves or enters the floor.
The assembly choreography is kinematic; the final load checks use free bodies.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from cases import random_cases
from equilibrium import solve
from scene import SHAPE, read, build
from ground_motion import GroundRoll
from kuka import Arm
from video import project, annotate

WIDTH, HEIGHT = 1280, 800
TEST_START = 29.
ARMS_CLEAR = 27.
PARK_A = np.array([-.05, -.10, .42])
PARK_B = np.array([.15, .10, .50])


def smooth(value):
    value = float(np.clip(value, 0., 1.))
    return value * value * (3 - 2 * value)


def mix(first, second, amount):
    return (1 - amount) * np.asarray(first) + amount * np.asarray(second)


def unit(value):
    value = np.asarray(value, float)
    return value / np.linalg.norm(value)


def motion_time(t):
    # Slow B's free withdrawal to 3 s and A's to 6 s to respect joint speeds.
    return float(np.interp(t, [0, 17, 20, 21, 27, 29], [0, 17, 19, 20, 22, 24]))


class Workflow:
    def __init__(self, model, seed=20260920, test_seconds=4.):
        self.model = model
        self.test_seconds = test_seconds
        self.duration = TEST_START + 10 * test_seconds + 2
        with np.load(SHAPE / 'object_geometry.npz') as geometry:
            self.object = trimesh.Trimesh(geometry['vertices_m'], geometry['faces'], process=False)
            work_ids = geometry['work_face_ids'].copy()
        self.com = model.body_ipos[model.body('object').id].copy()
        self.path = read(SHAPE / 'baseline_trajectory.json')
        if not (self.path['passed'] and self.path['continuous_sweep_verified']):
            raise ValueError('A verified Step6 insertion path is required')
        if self.path.get('path_kind') == 'piecewise_rigid' or np.linalg.norm(self.path['motion'][3:]) > 1e-14:
            raise ValueError('This B/pose_2 presentation expects its saved straight insertion path')
        self.withdrawal = (np.asarray(self.path['motion'][:3]) * self.path['length_scale_m']
                           * self.path['final_withdrawal_amount'])
        # Replay the original whole-solid sweep, not a sampled animation check.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'slides/baseline_algo'))
        from step5_connect_support import solids, belt_geometry, rigid_path
        with np.load(SHAPE / 'support_geometry.npz') as geometry:
            data = {key: geometry[key].copy() for key in geometry.files}
        assert rigid_path.replay(belt_geometry.Scene(self.object), solids.unpack_parts(data), self.path)
        self.support = trimesh.Trimesh(data['union_vertices_m'], data['union_faces'], process=False)
        # Choose an actual stable side placement, instead of rotating through the floor.
        poses, probabilities = self.object.compute_stable_poses(sigma=0, n_samples=1, threshold=0)
        low = [i for i, pose in enumerate(poses)
               if np.ptp(trimesh.transform_points(self.object.vertices, pose), axis=0)[2]
               < .7 * self.object.extents[2]]
        if not low:
            raise ValueError('No stable lying pose found')
        self.roll = GroundRoll(self.object.vertices, poses[max(low, key=lambda i: probabilities[i]), :3, :3])
        self.initial = self.roll.pose(0.)
        self.arm_times = None
        self.arm_q = None
        # A's visual tool touches a non-working side face; B touches the rear frame.
        centers = self.object.triangles_center
        candidates = np.flatnonzero((self.object.face_normals[:, 0] < -.45)
                                   & (centers[:, 2] > .05) & (centers[:, 2] < .17))
        candidates = np.setdiff1d(candidates, work_ids)
        # Keep the real 100 mm tool and robot wrist above the floor throughout.
        clearance = np.full(len(candidates), np.inf)
        for fraction in np.linspace(0, 1, 101):
            transform = self.roll.pose(fraction)
            flange = centers[candidates] + .135 * self.object.face_normals[candidates]
            clearance = np.minimum(clearance, trimesh.transform_points(flange, transform)[:, 2])
        safe = candidates[clearance > .10]
        if not len(safe):
            raise ValueError('No side pushing point keeps the wrist above the floor')
        index = safe[np.argmin(np.linalg.norm(centers[safe] - [-.01, -.01, .12], axis=1))]
        self.grip_a = centers[index]
        self.normal_a = self.object.face_normals[index]
        centers = self.support.triangles_center
        direction = unit(self.withdrawal)
        candidates = np.flatnonzero((self.support.face_normals @ direction > .6)
                                   & (centers[:, 2] > .075))
        index = candidates[np.argmax(centers[candidates] @ direction)]
        self.grip_b = centers[index]
        self.normal_b = self.support.face_normals[index]
        self.cases = random_cases(10, seed)
        self.results = []
        gravity_case = dict(self.cases[0], id='arms_clear_gravity', force_body_mg=[0., 0., 0.])
        for case in [gravity_case] + self.cases:
            result, _ = solve(model, case)
            if result['status'] != 'equilibrium_feasible':
                raise ValueError(f'{case["id"]}: static equilibrium unresolved; refusing a stationary success video')
            if case is not gravity_case:
                self.results.append(result)
        print('Step6 sweep replayed; gravity and 10 independent random loads balance both free bodies.', flush=True)

    def object_pose(self, t):
        return self.roll.pose(smooth((motion_time(t) - 4) / 5))

    def prepare_arms(self, fps=30):
        if self.arm_times is not None and getattr(self, 'arm_fps', None) == fps:
            return
        arms = [Arm('A'), Arm('B')]
        self.arms = arms
        self.arm_fps = None
        self.arm_times = np.arange(round(ARMS_CLEAR*fps)+1) / fps
        self.arm_q = np.empty((len(self.arm_times), 2, 7))
        for i, t in enumerate(self.arm_times):
            state = self.state(t)
            for j, (point, normal) in enumerate(((state['a'], state['normal_a']), (state['b'], state['normal_b']))):
                released = 21. if j == 0 else 17.
                returned = 27. if j == 0 else 20.
                try:
                    if t > released:
                        # In free space, return through joint space; do not force
                        # an unnecessary Cartesian wrist orientation singularity.
                        first = self.arm_q[round(released*fps),j]
                        self.arm_q[i,j] = mix(first, self.arm_q[0,j], smooth((t-released)/(returned-released)))
                        _,_,transforms = arms[j].forward(self.arm_q[i,j],geometry=True)
                        floor = min((v@r.T+p)[:,2].min() for v,(r,p) in zip(arms[j].collisions[1:],transforms[1:]))
                        if floor < -1e-6: raise ValueError('Robot intersects the floor during withdrawal')
                    else:
                        self.arm_q[i,j] = arms[j].solve(point, normal, None if i == 0 else self.arm_q[i-1,j])
                except ValueError as error:
                    raise ValueError(f'Arm {j}, t={t:.3f}: {error}') from error
            if i % (5*fps) == 0:
                print(f'LBR Med IK {t:.0f} / {ARMS_CLEAR:g} s', flush=True)
        velocities = np.abs(np.diff(self.arm_q, axis=0)) * fps
        for j, arm in enumerate(arms):
            ratio = float(np.max(velocities[:,j] / arm.velocity))
            if ratio > 1.0:
                raise ValueError(f'Robot joint speed exceeds model limits: {ratio:.3f}')
        self.arm_fps = fps
        print('Both LBR Med joint trajectories satisfy position, orientation and speed limits.', flush=True)

    def state(self, t):
        transform = self.object_pose(t)
        clock = t
        t = motion_time(t)
        support = self.withdrawal * (1 - smooth((t - 11) / 5))
        point_a = transform[:3, :3] @ self.grip_a + transform[:3, 3]
        normal_a = transform[:3, :3] @ self.normal_a
        point_b = self.grip_b + support
        a = mix(PARK_A, point_a, smooth((t - 2) / 2)) if t < 4 else point_a.copy()
        if t >= 19:
            a += .075 * normal_a * smooth(t - 19)
        if t >= 20:
            a = mix(point_a + .075 * normal_a, PARK_A, smooth((t - 20) / 2))
            normal_a = unit(mix(normal_a, [0., 0., 1.], smooth((t - 20) / 2)))
        b = point_b.copy()
        if t >= 16:
            b += .075 * self.normal_b * smooth(t - 16)
        if t >= 17:
            b = mix(point_b + .075 * self.normal_b, PARK_B, smooth((t - 17) / 2))
        normal_b = unit(mix(self.normal_b, [0., 0., 1.], smooth((t - 17) / 2)))
        if getattr(self, 'arm_fps', None):
            index = min(round(clock*self.arm_fps),len(self.arm_q)-1)
            if clock > 21.:
                a, axis, _, _ = self.arms[0].forward(self.arm_q[index,0])
                normal_a = -axis
            if clock > 17.:
                b, axis, _, _ = self.arms[1].forward(self.arm_q[index,1])
                normal_b = -axis
        load_index = None if clock < TEST_START else min(9, int((clock - TEST_START) / self.test_seconds))
        return dict(object_pose=transform, support_translation=support,
                    a=a, b=b, normal_a=normal_a, normal_b=normal_b,
                    working_area_visible=t >= 9, arm_a_holding=4 <= t < 19,
                    arm_b_holding=t < 16, load_index=load_index,
                    camera_close=smooth((t - 22) / 2))


def set_pose(model, data, name, transform):
    body = model.body(name).id
    address = model.jnt_qposadr[model.body_jntadr[body]]
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    data.qpos[address:address + 3] = transform[:3, 3]
    data.qpos[address + 3:address + 7] = quaternion[[3, 0, 1, 2]]


class Presentation:
    def __init__(self, workflow):
        self.workflow = workflow
        self.model, _ = build(robots=True)
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.azimuth, self.camera.elevation = 85, -27
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.model.geom_rgba[self.model.geom('object_visual').id] = [.52, .55, .58, 1]
        self.model.geom_rgba[self.model.geom('floor').id] = [.88, .9, .92, 1]
        self.model.vis.headlight.diffuse[:] = [.5, .5, .5]
        self.model.vis.headlight.ambient[:] = [.3, .3, .3]
        self.model.vis.headlight.specular[:] = [.1, .1, .1]
        self.robot_addresses = [[self.model.jnt_qposadr[self.model.joint(f'robot_{name}_joint_{i}').id]
                                for i in range(1,8)] for name in ('A','B')]

    def frame(self, renderer, t):
        state = self.workflow.state(t)
        set_pose(self.model, self.data, 'object', state['object_pose'])
        support = np.eye(4)
        support[:3, 3] = state['support_translation']
        set_pose(self.model, self.data, 'support', support)
        index = min(round(t*self.workflow.arm_fps), len(self.workflow.arm_q)-1)
        for addresses, q in zip(self.robot_addresses, self.workflow.arm_q[index]):
            self.data.qpos[addresses] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.model.geom_rgba[self.model.geom('working_area_visual').id, 3] = float(state['working_area_visible'])
        close = state['camera_close']
        action = smooth((motion_time(t)-2)/2) * (1-smooth((motion_time(t)-16)/2))
        distance = float(mix(1.9, .95, action))
        focus = mix([.04, .025, .25], [.035, .035, .12], action)
        self.camera.lookat[:] = mix(focus, [.03, .008, .095], close)
        self.camera.distance = float(mix(distance, .48, close))
        renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        image = Image.fromarray(renderer.render())
        if state['load_index'] is not None:
            result = self.workflow.results[state['load_index']]
            point = np.array(result['force_point_world_m'])
            force = np.array(result['force_world_N'])
            start = project(renderer.scene, point - force*.025, WIDTH, HEIGHT)
            end = project(renderer.scene, point, WIDTH, HEIGHT)
            annotate(ImageDraw.Draw(image), start, end)
        return np.asarray(image)


def render(workflow, path, fps=30):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'slides/baseline_algo'))
    from step5_connect_support.video import mp4_writer
    workflow.prepare_arms(fps)
    display = Presentation(workflow)
    frames = round(workflow.duration * fps)
    with mujoco.Renderer(display.model, height=HEIGHT, width=WIDTH) as renderer, mp4_writer(path, fps) as writer:
        for frame in range(frames):
            writer.append_data(display.frame(renderer, frame / fps))
            if frame % (fps * 5) == 0:
                print(f'Video {frame / fps:.0f} / {workflow.duration:.0f} s', flush=True)
    return frames
