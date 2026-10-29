"""Offline SO101 pick-and-place planner for B. No robot communication or rendering is imported."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot import JOINTS, Robot, points, rigid
from collision import World
from search import PlanningFailure, edge_samples, grasp_candidates, rrt_connect

HERE = Path(__file__).resolve().parent
PLANNER_VERSION = '2.0'
DEFAULTS = dict(seed=2601026, grasp_samples=8000, grasp_rolls=36, max_grasps=350, ik_random_seeds=3,
                contact_cone_half_angle_deg=20., contact_normal_tolerance_deg=20.,
                contact_directions_per_sample=5, contact_pair_limit=3000,
                refine_multistart=12, refine_max_nfev=100, refine_max_variants=3,
                position_tolerance_m=.0004, orientation_tolerance_deg=.5,
                edge_joint_deg=1., edge_step_m=.004, cartesian_step_m=.004,
                rrt_step_rad=.25, rrt_iterations=3000, rrt_seconds=12., shortcut_tries=50,
                maximum_joint_speed_deg_s=30., maximum_joint_acceleration_deg_s2=90.,
                maximum_jaw_speed_deg_s=30., maximum_jaw_acceleration_deg_s2=90., sample_hz=20.,
                pregrasp_distance_m=.035, grip_seconds=.5)


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def write(path, data):
    Path(path).write_text(json.dumps(plain(data), ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_pose(package, task):
    path = Path(package)
    info = json.loads((path / 'info.json' if path.is_dir() else path).read_text())
    row = next((r for r in info['tasks'] if r['id'] == task), None)
    if row is None:
        raise ValueError(f'Task {task} is not present in {package}')
    # Both Step5 and real_so101 packages declare millimetres in their transforms.
    if 'mm' not in info.get('units', ''):
        raise ValueError('Expected a system package explicitly declaring mm units')
    p = rigid(row['world_from_object']).copy()
    p[:3, 3] *= .001
    p[2, 3] += float(row.get('object_lift_mm', 0.)) * .001
    return p, row


def load_request(path):
    path = Path(path).resolve()
    request = json.loads(path.read_text())
    if request.get('schema') != 'bunny_movement_request_v1':
        raise ValueError('Expected bunny_movement_request_v1')
    directory = path.parent
    request['planner'] = dict(DEFAULTS, **request.get('planner', {}))
    for name in ('source', 'target'):
        spec = request[name]
        if 'package' in spec:
            pose, row = package_pose(directory / spec['package'], spec['task'])
            spec['world_from_object'] = pose.tolist()
            spec.setdefault('exit_direction_world', row.get('exit_direction_world', [0, 0, 1]))
            spec['system_source'] = dict(package=spec['package'], task=spec['task'])
            spec.setdefault('travel_m', float(row.get('exit_length_mm', 50.)) * .001)
            if spec.get('include_support'):
                package = directory / spec['package']
                package = package if package.is_dir() else package.parent
                support_pose = rigid(row['world_from_support']).copy()
                support_pose[:3, 3] *= .001
                mesh = os.path.relpath(package / 'support.stl', directory)
                obstacle = dict(name=name + '_fixture', mesh=mesh, mesh_units='mm',
                                world_from_mesh=support_pose.tolist(), supports_bunny=True)
                request.setdefault('obstacles', []).append(obstacle)
        rigid(spec['world_from_object'])
        direction = np.asarray(spec.get('exit_direction_world', [0, 0, 1]), float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-10:
            raise ValueError('Exit direction must be a finite nonzero world vector')
        spec['exit_direction_world'] = (direction / np.linalg.norm(direction)).tolist()
        if spec.get('travel_m', .05) <= 0:
            raise ValueError('Exit/entry travel must be positive')
    policy = request['planner']
    for key in ('grasp_samples', 'grasp_rolls', 'max_grasps', 'contact_directions_per_sample',
                'refine_max_nfev', 'refine_max_variants'):
        if not isinstance(policy[key], int) or policy[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    for key in ('refine_multistart', 'contact_pair_limit'):
        if not isinstance(policy[key], int) or policy[key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    if not 0 <= policy['contact_cone_half_angle_deg'] < 90 or not 0 < policy['contact_normal_tolerance_deg'] < 90:
        raise ValueError('Contact cone and normal tolerances must be below 90 degrees')
    for half_extents in request['gripper'].get('contact_patch_half_extents_m', {}).values():
        if np.shape(half_extents) != (3,) or not np.isfinite(half_extents).all() or np.min(half_extents) <= 0:
            raise ValueError('Measured contact patch half extents must be three positive lengths in metres')
    if any(not np.isfinite(policy[k]) or policy[k] <= 0 for k in
           ('position_tolerance_m', 'orientation_tolerance_deg', 'edge_joint_deg', 'edge_step_m',
            'cartesian_step_m', 'maximum_joint_speed_deg_s', 'maximum_joint_acceleration_deg_s2',
            'maximum_jaw_speed_deg_s', 'maximum_jaw_acceleration_deg_s2', 'sample_hz')):
        raise ValueError('Planner tolerances, speeds, accelerations and resolutions must be positive')
    mesh = trimesh.load(directory / request['object']['mesh'], force='mesh', process=True)
    unit = request['object'].get('mesh_units')
    if unit not in ('m', 'mm'):
        raise ValueError('Explicit object mesh_units (m or mm) are required')
    mesh.vertices *= .001 if unit == 'mm' else 1.
    if not mesh.is_watertight:
        raise ValueError('B must be a watertight triangle mesh')
    expected = request['object'].get('sha256')
    if expected and digest(directory / request['object']['mesh']) != expected:
        raise ValueError('B mesh hash differs from the system output')
    robot = Robot(request['robot'], directory, request['gripper'], policy)
    return request, robot, mesh, directory


class Planner:
    def __init__(self, request, robot, body, directory):
        self.request, self.robot, self.body = request, robot, body
        self.directory = directory
        self.policy = request['planner']
        self.rng = np.random.default_rng(self.policy['seed'])
        self.world = World(robot, body, request, directory)
        self.a, self.b = [rigid(request[n]['world_from_object']) for n in ('source', 'target')]
        self.start = np.radians(request['robot']['initial_joints_deg'])
        self.park = np.radians(request['robot'].get('park_joints_deg', request['robot']['initial_joints_deg']))
        if self.start.shape != (5,) or self.park.shape != (5,):
            raise ValueError('initial_joints_deg and park_joints_deg need five model angles')
        self.failures = Counter()
        self.refine_rng = np.random.default_rng(self.policy['seed'] + 991)
        self.last_edge_failure = None
        self.report = dict(planner_version=PLANNER_VERSION, sampling={}, refinement_seeds=0,
                           failed_attempts=[], candidates_checked=0)

    def check(self, x, body_pose, grasp):
        return self.world.check(x[:5], x[5], body_pose(x[:5]) if callable(body_pose) else body_pose, grasp)

    def edge(self, a, b, body_pose, grasp):
        r = self.robot
        reach = r.reach_bound + np.linalg.norm(self.body.extents)
        self.last_edge_failure = None
        for x in [a, *edge_samples(a, b, np.radians(self.policy['edge_joint_deg']), reach, self.policy['edge_step_m'])]:
            reason = self.check(x, body_pose, grasp)
            if reason:
                self.last_edge_failure = dict(reason=reason, joints_model_deg=np.degrees(x).tolist(),
                                              pad_contact=self.world.last_contact_failure)
                return False
        return True

    def line(self, q, first, last, jaw, body_pose, grasp, phase='cartesian'):
        out = [q]
        n = max(1, int(np.ceil(np.linalg.norm(last[:3, 3] - first[:3, 3]) / self.policy['cartesian_step_m'])))
        for u in np.arange(1, n + 1) / n:
            t = first.copy()
            t[:3, 3] = (1 - u) * first[:3, 3] + u * last[:3, 3]
            p = self.robot.ik(t, [q])
            if p is None:
                raise PlanningFailure('cartesian_ik', phase=phase, fraction=float(u), target_gripper=t)
            if not self.edge(np.r_[q, jaw], np.r_[p, jaw], body_pose, grasp):
                raise PlanningFailure('cartesian_collision', phase=phase, fraction=float(u), **self.last_edge_failure)
            out.append(p)
            q = p
        return out

    def access_line(self, q, actual, jaw, body_pose, grasp, endpoint, phase):
        """Try the tool-axis approach, then the endpoint's exit corridor; all motions retain full IK checks."""
        directions = [-actual[:3, 2], np.asarray(self.request[endpoint]['exit_direction_world'])]
        directions += [np.asarray(v, float) for v in self.request[endpoint].get('approach_directions_world', [])]
        distances = self.policy.get('pregrasp_distances_m', [self.policy['pregrasp_distance_m']])
        seen, errors = [], []
        for direction in directions:
            norm = np.linalg.norm(direction)
            if not np.isfinite(direction).all() or norm < 1e-10:
                raise ValueError('Approach directions must be nonzero finite world vectors')
            direction = direction / norm
            if any(np.allclose(direction, old) for old in seen):
                continue
            seen.append(direction)
            for distance in distances:
                if not np.isfinite(distance) or distance <= 0:
                    raise ValueError('Pregrasp distances must be positive')
                target = actual.copy()
                target[:3, 3] += direction * distance
                try:
                    path = self.line(q, actual, target, jaw, body_pose, grasp, phase)
                    return path, dict(direction_world=direction, distance_m=distance)
                except PlanningFailure as exc:
                    errors.append(dict(failure=str(exc), detail=exc.detail))
        raise PlanningFailure(errors[-1]['failure'], phase=phase, alternatives=errors)

    def free(self, a, b, jaw, body_pose, grasp):
        path = rrt_connect(a, b,
            lambda q: self.check(np.r_[q, jaw], body_pose, grasp) is None,
            lambda x, y: self.edge(np.r_[x, jaw], np.r_[y, jaw], body_pose, grasp),
            self.robot.lower, self.robot.upper, self.rng, self.policy)
        if path is None:
            raise PlanningFailure('rrt_connect')
        return path

    def attempt(self, grasp):
        r, p = self.robot, self.policy
        g = np.asarray(grasp['object_from_gripper'])
        jaw = grasp['jaw_rad']
        initial_jaw = np.radians(self.request['robot']['initial_jaw_deg'])
        opened = min(r.jaw_limits[1], jaw + np.radians(r.gripper['open_margin_deg']))
        if opened <= jaw:
            raise PlanningFailure('no_jaw_opening_margin')
        ta = self.a @ g
        qa = r.ik(ta, [grasp.get('source_seed', self.start), self.start], self.rng, p['ik_random_seeds'])
        if qa is None:
            raise PlanningFailure('source_ik')
        if self.world.check(qa, jaw, self.a, grasp):
            raise PlanningFailure('source_configuration_collision')
        # Attachment uses the attained FK, so the object never teleports to an ideal grasp transform.
        actual_a = r.frames(qa)[0]
        attachment = np.linalg.inv(actual_a) @ self.a
        held = lambda q: r.frames(q)[0] @ attachment
        tb = self.b @ np.linalg.inv(attachment)
        qb = r.ik(tb, [qa, self.start], self.rng, p['ik_random_seeds'])
        if qb is None:
            raise PlanningFailure('target_ik')
        placed_b = held(qb)
        if self.world.check(qb, jaw, placed_b, grasp):
            raise PlanningFailure('target_configuration_collision')
        # Solve cheap Cartesian pieces before spending time on free-space RRT searches.
        reverse_a, source_access = self.access_line(qa, actual_a, opened, self.a, grasp, 'source', 'approach')
        out_a = actual_a.copy()
        out_a[:3, 3] += np.asarray(self.request['source']['exit_direction_world']) * self.request['source'].get('travel_m', .05)
        extraction = self.line(qa, actual_a, out_a, jaw, held, grasp, 'extract')
        actual_b = r.frames(qb)[0]
        in_b = actual_b.copy()
        in_b[:3, 3] += np.asarray(self.request['target']['exit_direction_world']) * self.request['target'].get('travel_m', .05)
        insertion_reverse = self.line(qb, actual_b, in_b, jaw, held, grasp, 'insert')
        retreat, target_access = self.access_line(qb, actual_b, opened, placed_b, grasp, 'target', 'retreat')
        if not self.edge(np.r_[qa, opened], np.r_[qa, jaw], self.a, grasp):
            raise PlanningFailure('closing_collision', phase='close', **self.last_edge_failure)
        if not self.edge(np.r_[qb, jaw], np.r_[qb, opened], placed_b, grasp):
            raise PlanningFailure('release_collision', phase='release', **self.last_edge_failure)
        if not self.edge(np.r_[self.start, initial_jaw], np.r_[self.start, opened], self.a, grasp):
            raise PlanningFailure('initial_jaw_adjustment_collision')
        if not self.edge(np.r_[self.park, opened], np.r_[self.park, initial_jaw], placed_b, grasp):
            raise PlanningFailure('park_jaw_adjustment_collision')
        first = self.free(self.start, reverse_a[-1], opened, self.a, grasp)
        carrying = self.free(extraction[-1], insertion_reverse[-1], jaw, held, grasp)
        last = self.free(retreat[-1], self.park, opened, placed_b, grasp)
        phases = [
            ('prepare_gripper', [self.start, self.start], (initial_jaw, opened), self.a, False),
            ('approach_free', first, opened, self.a, False),
            ('approach', reverse_a[::-1], opened, self.a, False),
            ('close', [qa, qa], (opened, jaw), self.a, False),
            ('extract', extraction, jaw, held, True),
            ('carry', carrying, jaw, held, True),
            ('insert', insertion_reverse[::-1], jaw, held, True),
            ('release', [qb, qb], (jaw, opened), placed_b, False),
            ('retreat', retreat, opened, placed_b, False),
            ('return', last, opened, placed_b, False),
            ('reset_gripper', [self.park, self.park], (opened, initial_jaw), placed_b, False),
        ]
        return self.retime(phases), dict(grasp=grasp, gripper_from_object=attachment,
                                        source_access=source_access, target_access=target_access,
                                        actual_source_world_from_object=self.a,
                                        actual_target_world_from_object=placed_b, jaw_open_deg=np.degrees(opened))

    def refine_grasp(self, grasp):
        """Compatibility wrapper returning the first valid multistart branch."""
        return next(self.refine_grasps(grasp), None)

    def refine_grasps(self, grasp):
        """Yield distinct source IK branches; later path/replay failures resume the remaining seeds."""
        r = self.robot
        initial = np.asarray(grasp['object_from_gripper'])
        fixed, moving = np.asarray(grasp['contacts_object_m'])
        axis = moving - fixed
        axis /= np.linalg.norm(axis)
        def pose(angle):
            g = initial.copy()
            g[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix() @ initial[:3, :3]
            g[:3, 3] = fixed - g[:3, :3] @ r.fixed_pad
            return g
        def residual(x):
            wanted = self.a @ pose(x[5])
            attained = r.frames(x[:5])[0]
            return np.r_[attained[:3, 3] - wanted[:3, 3],
                         .12 * Rotation.from_matrix(wanted[:3, :3] @ attained[:3, :3].T).as_rotvec()]
        half_bin = np.pi / self.policy['grasp_rolls']
        seed = self.start.copy()
        local = points((self.a @ initial)[:3, 3], np.linalg.inv(r.base))
        seed[0] = np.clip(-np.arctan2(local[1], local[0] - .0388353), r.lower[0], r.upper[0])
        starts = [np.r_[np.clip(seed, r.lower, r.upper), 0.],
                  np.r_[np.clip(self.start, r.lower, r.upper), 0.]]
        if 'source_seed' in grasp:
            starts.append(np.r_[np.clip(grasp['source_seed'], r.lower, r.upper), 0.])
        starts.extend(np.r_[q, self.refine_rng.uniform(-half_bin, half_bin)] for q in
                      self.refine_rng.uniform(r.lower, r.upper, (self.policy['refine_multistart'], 5)))
        found = []
        for index, start in enumerate(starts):
            self.report['refinement_seeds'] += 1
            fit = least_squares(residual, start,
                bounds=(np.r_[r.lower, -half_bin], np.r_[r.upper, half_bin]),
                max_nfev=self.policy['refine_max_nfev'], ftol=1e-9, xtol=1e-9, gtol=1e-9)
            e = residual(fit.x)
            if (np.linalg.norm(e[:3]) > self.policy['position_tolerance_m'] or
                    np.linalg.norm(e[3:]) / .12 > np.radians(self.policy['orientation_tolerance_deg'])):
                continue
            if any(np.linalg.norm(fit.x[:5] - old[:5]) < .02 and abs(fit.x[5] - old[5]) < .002 for old in found):
                continue
            found.append(fit.x.copy())
            yield dict(grasp, object_from_gripper=pose(fit.x[5]), source_seed=fit.x[:5],
                       refine_seed_index=index, roll_refinement_deg=float(np.degrees(fit.x[5])))
            if len(found) >= self.policy['refine_max_variants']:
                return

    def retime(self, phases):
        p = self.policy
        vmax = np.radians([p['maximum_joint_speed_deg_s']] * 5 + [p['maximum_jaw_speed_deg_s']])
        amax = np.radians([p['maximum_joint_acceleration_deg_s2']] * 5 + [p['maximum_jaw_acceleration_deg_s2']])
        frames = []
        now = 0.
        for name, path, jaw, pose, attached in phases:
            jaw_path = np.linspace(*jaw, len(path)) if isinstance(jaw, tuple) else np.full(len(path), jaw)
            x = np.column_stack([path, jaw_path])
            if not frames:
                frames.append(dict(t=now, q=x[0], object_pose=pose(x[0, :5]) if callable(pose) else pose,
                                   phase=name, attached=attached))
            for a, b in zip(x, x[1:]):
                delta = np.abs(b - a)
                # Exact maxima of quintic smoothstep: v=1.875 delta/T; a=(10/sqrt(3)) delta/T^2.
                duration = max(float((1.875 * delta / vmax).max()),
                               float(np.sqrt((10 / np.sqrt(3)) * delta / amax).max()), 1 / p['sample_hz'])
                if name in ('close', 'release'):
                    duration = max(duration, p['grip_seconds'])
                n = max(1, int(np.ceil(duration * p['sample_hz'])))
                duration = n / p['sample_hz']
                for k in range(1, n + 1):
                    u = k / n
                    s = u ** 3 * (10 - 15 * u + 6 * u * u)
                    q = a + s * (b - a)
                    frames.append(dict(t=now + k / p['sample_hz'], q=q,
                        object_pose=pose(q[:5]) if callable(pose) else pose, phase=name, attached=attached))
                now += duration
        return frames

    def plan(self):
        p, r = self.policy, self.robot
        began = time.monotonic()
        for name, pose in (('source', self.a), ('target', self.b)):
            if not self.world.has_support(pose):
                raise PlanningFailure(name + ' has no nearby table/declared support; include its fixture geometry')
        initial = self.world.check(self.start, np.radians(self.request['robot']['initial_jaw_deg']), self.a)
        if initial:
            raise PlanningFailure('Initial state is invalid: ' + initial)
        candidates = grasp_candidates(self.body, r, self.a, self.b, self.rng, p, self.report['sampling'])
        print('GRASPS', len(candidates), flush=True)
        attempts = 0
        for grasp in candidates:
            self.report['candidates_checked'] += 1
            g = np.asarray(grasp['object_from_gripper'])
            if not all(self.world.hand_clear(t @ g, grasp['jaw_rad'], t, grasp) for t in (self.a, self.b)):
                self.failures['hand_geometry'] += 1
                continue
            refined = False
            for variant in self.refine_grasps(grasp):
                refined = True
                g = np.asarray(variant['object_from_gripper'])
                if not all(self.world.hand_clear(t @ g, variant['jaw_rad'], t, variant) for t in (self.a, self.b)):
                    self.failures['hand_geometry_after_refinement'] += 1
                    continue
                if attempts >= p['max_grasps']:
                    break
                attempts += 1
                try:
                    frames, result = self.attempt(variant)
                    # The exact timed frames are part of candidate acceptance, not a one-shot check after search.
                    candidate_validation = verify(self.request, r, self.body, self.directory, frames, result)
                    if candidate_validation['status'] != 'passed':
                        raise PlanningFailure('retimed_replay', errors=candidate_validation['errors'][:8],
                            pad_contacts=candidate_validation.get('pad_contact_failures', [])[:8])
                    self.report.update(status='passed', stage_failures=dict(self.failures),
                                       seconds=time.monotonic() - began, attempts=attempts)
                    print('FOUND', attempts, 'grasp attempts;', len(frames), 'frames', flush=True)
                    return frames, dict(result, failures_before_success=dict(self.failures), grasps_tried=attempts)
                except PlanningFailure as exc:
                    self.failures[str(exc)] += 1
                    if len(self.report['failed_attempts']) < 100:
                        self.report['failed_attempts'].append(dict(attempt=attempts, failure=str(exc),
                            detail=exc.detail, contacts_object_m=variant['contacts_object_m']))
                if attempts % 10 == 0:
                    print('SEARCH', attempts, dict(self.failures), flush=True)
            if not refined:
                self.failures['five_axis_grasp_orientation'] += 1
            if attempts >= p['max_grasps']:
                break
        self.report.update(status='failed', stage_failures=dict(self.failures),
                           seconds=time.monotonic() - began, attempts=attempts)
        raise PlanningFailure('No feasible single-grasp transfer in the search budget: ' + str(dict(self.failures)))


def verify(request, robot, body, directory, frames, result):
    """Rebuild collision geometry and replay FK; also check between saved frames. Does not solve IK."""
    world = World(robot, body, request, directory)
    policy = request['planner']
    attachment = rigid(result['gripper_from_object'])
    grasp = result['grasp']
    errors = []
    contact_failures = []
    worst_rigid = 0.
    previous = None
    reach = robot.reach_bound + np.linalg.norm(body.extents)
    for i, f in enumerate(frames):
        q, pose = np.asarray(f['q']), rigid(f['object_pose'])
        if f['attached']:
            fk = robot.frames(q[:5])[0] @ attachment
            worst_rigid = max(worst_rigid, float(np.abs(fk - pose).max()))
            if not np.allclose(fk, pose, atol=1e-8):
                errors.append((i, 'broken_rigid_attachment'))
        else:
            expected = (request['source']['world_from_object'] if f['phase'] in
                        ('prepare_gripper', 'approach_free', 'approach', 'close') else
                        result['actual_target_world_from_object'])
            if not np.allclose(pose, expected, atol=1e-8):
                errors.append((i, 'unattached_bunny_moved'))
        reason = world.check(q[:5], q[5], pose, grasp)
        if reason:
            errors.append((i, reason))
            if world.last_contact_failure:
                contact_failures.append(dict(frame=i, phase=f['phase'], jaw_deg=float(np.degrees(q[5])),
                                             **world.last_contact_failure))
        if previous is not None:
            dt = f['t'] - previous['t']
            if dt <= 0:
                errors.append((i, 'nonpositive_time_step'))
            attached = f['attached']
            if attached and not previous['attached']:
                if not np.allclose(robot.frames(np.asarray(previous['q'])[:5])[0] @ attachment,
                                   previous['object_pose'], atol=1e-8):
                    errors.append((i, 'attachment_teleport'))
            for x in edge_samples(np.asarray(previous['q']), q, np.radians(policy['edge_joint_deg']), reach, policy['edge_step_m']):
                bp = robot.frames(x[:5])[0] @ attachment if attached else pose
                reason = world.check(x[:5], x[5], bp, grasp)
                if reason:
                    errors.append((i, 'between_frames:' + reason))
                    if world.last_contact_failure:
                        contact_failures.append(dict(frame=i, phase=f['phase'], between_frames=True,
                            jaw_deg=float(np.degrees(x[5])), **world.last_contact_failure))
                    break
        previous = f
        if len(errors) > 20:
            break
    times = np.array([f['t'] for f in frames])
    q = np.array([f['q'] for f in frames])
    dt = np.diff(times)
    if len(frames) < 2 or (dt <= 0).any():
        raise PlanningFailure('Invalid saved time axis')
    velocity = np.diff(np.degrees(q), axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
    speed = np.array([policy['maximum_joint_speed_deg_s']] * 5 + [policy['maximum_jaw_speed_deg_s']])
    accel = np.array([policy['maximum_joint_acceleration_deg_s2']] * 5 + [policy['maximum_jaw_acceleration_deg_s2']])
    if (np.abs(velocity).max(0) > speed + 1e-5).any():
        errors.append(('all', 'speed_limit'))
    if len(acceleration) and (np.abs(acceleration).max(0) > accel + 1e-4).any():
        errors.append(('all', 'acceleration_limit'))
    for name, f in [('source', frames[0]), ('target', frames[-1])]:
        wanted = rigid(request[name]['world_from_object'])
        actual = rigid(f['object_pose'])
        if not world.has_support(actual):
            errors.append((name, 'missing_placement_support'))
        if np.linalg.norm(actual[:3, 3] - wanted[:3, 3]) > policy['position_tolerance_m']:
            errors.append((name, 'placement_position'))
        if Rotation.from_matrix(wanted[:3, :3] @ actual[:3, :3].T).magnitude() > np.radians(policy['orientation_tolerance_deg']):
            errors.append((name, 'placement_orientation'))
    if not np.allclose(q[0, :5], np.radians(request['robot']['initial_joints_deg']), atol=1e-8):
        errors.append(('source', 'initial_joints'))
    if not np.isclose(q[0, 5], np.radians(request['robot']['initial_jaw_deg']), atol=1e-8):
        errors.append(('source', 'initial_jaw'))
    park = request['robot'].get('park_joints_deg', request['robot']['initial_joints_deg'])
    if not np.allclose(q[-1], np.radians([*park, request['robot']['initial_jaw_deg']]), atol=1e-8):
        errors.append(('target', 'final_park_state'))
    return dict(status='passed' if not errors else 'failed', errors=errors, frames=len(frames),
        pad_contact_failures=contact_failures,
        duration_s=float(times[-1]), collision_configurations_checked=world.calls,
        maximum_joint_speed_deg_s=np.abs(velocity).max(0),
        maximum_joint_acceleration_deg_s2=np.abs(acceleration).max(0) if len(acceleration) else np.zeros(6),
        maximum_rigid_attachment_matrix_error=worst_rigid,
        method='FK replay; FCL mesh contacts; all saved frames and subdivided joint-space edges',
        continuous_collision_certificate=False, grasp_force_and_payload_certified=False,
        physical_robot_executed=False, station_geometry_scope=request.get('station_geometry_scope', 'supplied obstacles only'))


def export(folder, request, robot, frames, result, validation, source_path):
    folder.mkdir(parents=True, exist_ok=True)
    q = np.array([f['q'] for f in frames])
    np.savez_compressed(folder / 'trajectory.npz', times_s=[f['t'] for f in frames],
        joints_model_deg=np.degrees(q), joints_hardware_deg=robot.hardware_degrees(q),
        world_from_object=np.array([f['object_pose'] for f in frames]),
        world_from_gripper=np.array([robot.frames(f['q'][:5])[0] for f in frames]),
        phase=np.array([f['phase'] for f in frames]), attached=np.array([f['attached'] for f in frames]))
    directory = Path(source_path).resolve().parent
    sources = [Path(source_path).resolve(), directory / request['object']['mesh'], *robot.asset_paths]
    sources += [directory / o['mesh'] for o in request.get('obstacles', []) if 'mesh' in o]
    for n in ('source', 'target'):
        if 'package' in request[n]:
            path = directory / request[n]['package']
            sources.append(path / 'info.json' if path.is_dir() else path)
    sources += [HERE / n for n in ('bunny_movement.py', 'robot.py', 'collision.py', 'search.py')]
    manifest = {os.path.relpath(path.resolve(), directory): digest(path) for path in sources}
    program = dict(schema='bunny_movement_program_v1', status='model_verified', planner_version=PLANNER_VERSION,
        source_request_sha256=digest(source_path), joint_order=[*JOINTS, 'gripper'],
        input_sha256=manifest,
        joint_limits_model_deg=np.degrees(np.vstack([np.c_[robot.lower, robot.upper], robot.jaw_limits])),
        units=dict(time='s', joints='deg', transforms='m'),
        hardware_mapping='model_deg = hardware_signs * hardware_deg + hardware_zero_offsets_deg',
        physical_robot_executed=False, trajectory_sha256=digest(folder / 'trajectory.npz'),
        solution=plain(result), execution_contract=dict(
            initial_state='Measured arm AND jaw must match the first waypoint; otherwise replan.',
            close='Stop after close and confirm that B is held before extract.',
            release='Confirm that B is supported before opening; otherwise stop.',
            station='Use measured carrier base registration, jaw pad calibration, payload and parked work-arm geometry.'),
        waypoints=[dict(time_s=f['t'], positions_model_deg=np.degrees(f['q']),
                        positions_hardware_deg=robot.hardware_degrees(f['q']), phase=f['phase'],
                        object_attached=f['attached']) for f in frames])
    write(folder / 'program.json', program)
    write(folder / 'validation.json', dict(validation, trajectory_sha256=digest(folder / 'trajectory.npz'),
                                           program_sha256=digest(folder / 'program.json')))
    write(folder / 'result.json', dict(status='passed', result=result, validation=validation))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=HERE / 'example_request.json')
    parser.add_argument('--output', type=Path, default=HERE / 'example_output')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    planner = None
    try:
        request, robot, body, directory = load_request(args.request)
        if args.validate_only:
            program = json.loads((args.output / 'program.json').read_text())
            old_validation = json.loads((args.output / 'validation.json').read_text())
            if old_validation.get('program_sha256') != digest(args.output / 'program.json'):
                raise PlanningFailure('Program changed after validation')
            if program['source_request_sha256'] != digest(args.request):
                raise PlanningFailure('Request changed after planning')
            if program['trajectory_sha256'] != digest(args.output / 'trajectory.npz'):
                raise PlanningFailure('Trajectory changed after planning')
            for path, expected in program['input_sha256'].items():
                if digest(directory / path) != expected:
                    raise PlanningFailure('Input geometry or planner source changed: ' + path)
            with np.load(args.output / 'trajectory.npz') as z:
                n = len(z['times_s'])
                if (z['joints_model_deg'].shape != (n, 6) or z['joints_hardware_deg'].shape != (n, 6) or
                        z['world_from_object'].shape != (n, 4, 4) or z['world_from_gripper'].shape != (n, 4, 4) or
                        z['phase'].shape != (n,) or z['attached'].shape != (n,) or len(program['waypoints']) != n):
                    raise PlanningFailure('Inconsistent saved trajectory array sizes')
                hardware = robot.hardware_degrees(np.radians(z['joints_model_deg']))
                if not np.allclose(hardware, z['joints_hardware_deg'], atol=1e-8):
                    raise PlanningFailure('Saved hardware mapping does not match the model joints')
                for i, waypoint in enumerate(program['waypoints']):
                    if (not np.isclose(waypoint['time_s'], z['times_s'][i]) or
                            not np.allclose(waypoint['positions_model_deg'], z['joints_model_deg'][i], atol=1e-8) or
                            not np.allclose(waypoint['positions_hardware_deg'], hardware[i], atol=1e-8) or
                            waypoint['phase'] != str(z['phase'][i]) or waypoint['object_attached'] != bool(z['attached'][i]) or
                            not np.allclose(robot.frames(np.radians(z['joints_model_deg'][i, :5]))[0], z['world_from_gripper'][i], atol=1e-8)):
                        raise PlanningFailure('Program/trajectory/FK mismatch at frame ' + str(i))
                frames = [dict(t=float(t), q=np.radians(q), object_pose=b, phase=str(phase), attached=bool(attached))
                          for t, q, b, phase, attached in zip(z['times_s'], z['joints_model_deg'], z['world_from_object'], z['phase'], z['attached'])]
            validation = verify(request, robot, body, directory, frames, program['solution'])
            write(args.output / 'validation.json', dict(validation,
                trajectory_sha256=digest(args.output / 'trajectory.npz'), program_sha256=digest(args.output / 'program.json')))
            if validation['status'] != 'passed':
                raise PlanningFailure(str(validation['errors']))
            write(args.output / 'result.json', dict(status='passed', result=program['solution'], validation=validation))
        else:
            # A failed rerun must not leave an older program at the advertised output path.
            for name in ('program.json', 'trajectory.npz', 'validation.json', 'result.json'):
                (args.output / name).unlink(missing_ok=True)
            planner = Planner(request, robot, body, directory)
            frames, result = planner.plan()
            write(args.output / 'search_report.json', planner.report)
            validation = verify(request, robot, body, directory, frames, result)
            if validation['status'] != 'passed':
                write(args.output / 'validation.json', validation)
                raise PlanningFailure('Replay failed: ' + str(validation['errors']))
            export(args.output, request, robot, frames, result, validation, args.request)
        print('VERIFIED', validation['frames'], 'frames;', round(validation['duration_s'], 2), 'seconds;',
              round(time.monotonic() - began, 1), 's computation', flush=True)
        return 0
    except (PlanningFailure, ValueError, OSError, KeyError) as exc:
        if planner is not None:
            planner.report.update(status='failed', stage_failures=dict(planner.failures))
            write(args.output / 'search_report.json', planner.report)
        write(args.output / 'result.json', dict(status='failed', error=str(exc), physical_robot_executed=False,
                                               detail=getattr(exc, 'detail', {})))
        print('FAILED:', exc, file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
