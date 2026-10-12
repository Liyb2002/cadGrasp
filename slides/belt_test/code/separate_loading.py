"""One-arm, separate-body loading choreography for the belt_test concept film."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import trimesh

import separate_scene
import fixture_geometry as F


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3-2*t)


def shifted(T, delta):
    T = T.copy(); T[:3, 3] += delta
    return T


def blend(A, B, u):
    u = float(smooth(u))
    T = np.eye(4)
    T[:3, :3] = Slerp([0., 1.], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))([u]).as_matrix()[0]
    T[:3, 3] = (1-u)*A[:3, 3]+u*B[:3, 3]
    return T


def tool_frame(point, normal, closing_axis):
    z = -np.asarray(normal); z /= np.linalg.norm(z)
    x = np.asarray(closing_axis)-z*np.dot(z, closing_axis); x /= np.linalg.norm(x)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, np.cross(z, x), z]); T[:3, 3] = point
    return T


def object_grasp(mesh):
    normal = np.array([.5, -.3, .75]); normal /= np.linalg.norm(normal)
    hits, _, _ = mesh.ray.intersects_location([mesh.center_mass], [normal], multiple_hits=True)
    surface = min(hits, key=lambda q:np.linalg.norm(q-mesh.center_mass))
    point = surface-normal*.012
    first = np.cross([0., 0., 1.], normal); first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    candidates = []
    for angle in np.linspace(0., np.pi, 36, endpoint=False):
        axis = first*np.cos(angle)+second*np.sin(angle)
        hit, rays, _ = mesh.ray.intersects_location([point, point], [axis, -axis], multiple_hits=True)
        if len(np.unique(rays)) != 2:
            continue
        distance = [min(np.linalg.norm(q-point) for q in hit[rays==i]) for i in range(2)]
        gap = sum(distance)/2+.004
        if .010 < gap < .060:
            center = point+axis*(distance[0]-distance[1])/2
            candidates.append((abs(distance[0]-distance[1]), gap, center, axis))
    if not candidates:
        raise RuntimeError('No object grip section fits the illustrated jaw stroke')
    _, gap, point, axis = min(candidates, key=lambda row:row[0])
    return tool_frame(point, normal, axis), gap


def stable_parking(mesh, grasp, xy):
    poses, probability = mesh.compute_stable_poses(sigma=0., n_samples=1, threshold=0.)
    options = []
    for i, T in enumerate(poses):
        if (T[:3, :3]@(-grasp[:3, 2]))[2] < .25:
            continue
        options.append((float(probability[i]), i, T.copy()))
    if not options:
        raise RuntimeError('No stable parking pose admits an above-floor grasp')
    _, index, T = max(options, key=lambda row:row[0])
    # Face the approaching wrist toward the robot on the positive-X side.
    outward = T[:3, :3]@(-grasp[:3, 2])
    yaw = -np.arctan2(outward[1], outward[0])
    T = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])@T
    points = trimesh.transform_points(mesh.vertices, T)
    T[:2, 3] += np.asarray(xy)-(points[:, :2].max(0)+points[:, :2].min(0))/2
    T[2, 3] -= points[:, 2].min()
    return T, int(index)


class Sequence:
    def __init__(self):
        from shared_workflow import kuka, ROBOT_BASE
        self.cases, self.base, self.obj, self.blue, self.meta = separate_scene.build()
        object_tool, object_gap = object_grasp(self.obj)
        rod = np.asarray(self.meta['blue_grip_rod_axis']); rod /= np.linalg.norm(rod)
        # The jaws close perpendicular to the rod. The approach may be
        # oblique to the rod, keeping the wrist above all three dock poses.
        normal = np.array([.5, -.6, .6]); normal /= np.linalg.norm(normal)
        blue_tool = tool_frame(self.meta['blue_grip_local'], normal, np.cross(rod, normal))
        self.grasps = {'object':object_tool, 'blue':blue_tool}
        self.gaps = {'object':float(object_gap), 'blue':.009}
        self.object_park, self.lying_index = stable_parking(self.obj, object_tool, [-.04, -.16])
        self.blue_park, self.blue_parking_index = stable_parking(self.blue, blue_tool, [.23, -.15])
        self.poses = {'object':self.object_park.copy(), 'blue':self.blue_park.copy()}
        self.hand = tool_frame([.33, -.04, .48], [0., 0., 1.], [1., 0., 0.])
        self.hand_park = self.hand.copy()
        self.gap = .07
        self.segments = []
        self.transport_rows = {}
        self.time = 0.
        self.story_times = []
        self._hold(1., 'ready')
        for i, case in enumerate(self.cases):
            dock = case['transform']
            blue_pre = shifted(dock, case['direction']*.03)
            object_pre = case['object_pre']
            blue_waypoints = [shifted(self.blue_park,[0,0,.28])]
            if i==2:
                blue_waypoints.append(shifted(self.blue_park,[-.17,0,.28]))
            blue_waypoints += [shifted(blue_pre,case['blue_departure']*(.35 if i==1 else .28)),blue_pre]
            self._transport('blue',self.blue_park,dock,blue_waypoints,i,'install_blue')
            self.story_times.append(self.time-.6)
            object_outer = shifted(object_pre, [-.06,0,.02]) if i==0 else object_pre
            waypoints = [shifted(self.object_park,[0,0,.28]),
                         shifted(object_outer,[0,0,.28]),object_outer]
            if i==0:
                waypoints.append(object_pre)
            self._transport('object',self.object_park,dock,waypoints,i,'install_object')
            self.story_times.append(self.time-.6)
            self._hold(3., 'task', case=i)
            self.story_times.append(self.time-1.5)
            # Reverse loading order: never leave an unsupported object on a
            # point contact while the single arm is busy carrying the module.
            self._transport('object', dock, self.object_park,
                [object_pre, shifted(object_pre, [0, 0, .28]),
                 shifted(self.object_park, [0, 0, .28])], i, 'remove_object')
            self.story_times.append(self.time-.6)
            self._transport('blue', dock, self.blue_park,
                [blue_pre, shifted(blue_pre, [0, 0, .28]),
                 shifted(self.blue_park, [0, 0, .28])], i, 'remove_blue')
            self.story_times.append(self.time-.6)
        self._hold(1., 'finished')
        self.duration = self.time
        self.arm = kuka.Arm('B'); self.arm.base = ROBOT_BASE.copy()
        self.arm_times = None; self.time_scale = 1.

    def _add(self, seconds, label, hand=None, gap=None, moving=None, pose=None, case=None):
        end_poses = {key:T.copy() for key, T in self.poses.items()}
        if moving is not None:
            end_poses[moving] = pose.copy()
        end_hand = self.hand.copy() if hand is None else hand.copy()
        if moving is not None:
            end_hand = pose@self.grasps[moving]
        end_gap = self.gap if gap is None else gap
        self.segments.append(dict(start=self.time, end=self.time+seconds, label=label,
            case=case, moving=moving, poses0={k:T.copy() for k,T in self.poses.items()},
            poses1=end_poses, hand0=self.hand.copy(), hand1=end_hand,
            gap0=self.gap, gap1=end_gap))
        self.time += seconds; self.poses = end_poses; self.hand = end_hand; self.gap = end_gap

    def _hold(self, seconds, label, case=None):
        self._add(seconds, label, case=case)

    def _transport(self, body, start, end, waypoints, case, label):
        if label.startswith('remove_'):
            # The same unobstructed handoff can be executed in reverse. Keep
            # its IK branch too, instead of accumulating redundant-arm drift.
            forward = self.transport_rows[case, body]
            for index,source in enumerate(reversed(forward)):
                original=source['label'].removeprefix('install_'+body+'_')
                suffix={'grasp':'release','release':'grasp'}.get(original,'reverse_'+str(index)+'_'+original)
                self._add(source['end']-source['start'], label+'_'+suffix,
                          hand=source['hand0'], gap=source['gap0'], moving=source['moving'],
                          pose=source['poses0'][body], case=case)
                self.segments[-1]['reverse_source'] = source
            return
        first = len(self.segments)
        grasp = start@self.grasps[body]
        approach = shifted(grasp, -grasp[:3, 2]*.10)
        high = shifted(approach, [0, 0, max(0., .48-approach[2, 3])])
        self._add(1.5, label+'_approach_high', hand=high, gap=.07, case=case)
        self._add(1., label+'_approach', hand=approach, case=case)
        self._add(.8, label+'_reach', hand=grasp, case=case)
        self._add(.6, label+'_grasp', gap=self.gaps[body], case=case)
        for j, target in enumerate(waypoints+[end]):
            self._add(2. if j in (0, 1) else 1.4, label+f'_move_{j}',
                      moving=body, pose=target, case=case)
        self._add(.6, label+'_release', gap=.07, case=case)
        grasp = end@self.grasps[body]
        self._add(.8, label+'_retreat', hand=shifted(grasp, -grasp[:3, 2]*.10), case=case)
        self._add(1.5, label+'_clear', hand=self.hand_park, case=case)
        self.transport_rows[case, body] = self.segments[first:]

    def state(self, t):
        row = next((r for r in self.segments if t <= r['end']), self.segments[-1])
        u = np.clip((t-row['start'])/(row['end']-row['start']), 0., 1.)
        poses = {key:blend(row['poses0'][key], row['poses1'][key], u) for key in ('object', 'blue')}
        if row['moving'] is not None:
            hand = poses[row['moving']]@self.grasps[row['moving']]
        else:
            hand = blend(row['hand0'], row['hand1'], u)
        gap = (1-smooth(u))*row['gap0']+smooth(u)*row['gap1']
        return dict(**poses, hand=hand, point=hand[:3, 3], normal=-hand[:3, 2],
            gap=float(gap), work=row['case'], task=row['label']=='task',
            phase=row['label'], moving=row['moving'], time=float(t))

    def check_geometry(self):
        base = F.solid(self.base)
        object_solid, blue_solid = F.solid(self.obj), F.solid(self.blue)
        rows = []
        for t in np.linspace(0, self.duration, round(self.duration*10)+1):
            state = self.state(t)
            record = dict(time_s=float(t))
            for key, mesh, solid in (('object', self.obj, object_solid), ('blue', self.blue, blue_solid)):
                T = state[key].copy(); T[:3, 3] /= F.SCALE
                moving = solid.transform(T[:3, :])
                record[key+'_base_overlap_mm3'] = abs(float((moving^base).volume()))*F.SCALE**3*1e9
                record[key+'_min_z_mm'] = float(trimesh.transform_points(mesh.vertices, state[key])[:, 2].min())*1000
            relative = np.linalg.inv(state['object'])@state['blue']
            relative[:3, 3] /= F.SCALE
            record['object_blue_overlap_mm3'] = abs(float((object_solid^blue_solid.transform(relative[:3, :])).volume()))*F.SCALE**3*1e9
            rows.append(record)
        maxima = {k:max(r[k] for r in rows) for k in ('object_base_overlap_mm3','blue_base_overlap_mm3','object_blue_overlap_mm3')}
        floor_min = min(min(r['object_min_z_mm'], r['blue_min_z_mm']) for r in rows)
        report = dict(samples=rows, maxima=maxima, minimum_floor_height_mm=floor_min,
                      clear=max(maxima.values()) < .01 and floor_min > -.01)
        print('Shared-base full sequence geometry:', maxima, 'floor mm', floor_min, flush=True)
        if not report['clear']:
            failures = [r for r in rows if max(r[k] for k in maxima) > .01 or min(r['object_min_z_mm'],r['blue_min_z_mm']) < -.01]
            print('First conflicts:', failures[:8], flush=True)
            raise RuntimeError('Shared-base sequence has a sampled geometric collision')
        return report


    def prepare_arm(self, sample_hz=8, initial_seed=None):
        self.arm_times = np.linspace(0., self.duration, int(self.duration*sample_hz)+1)
        qs = []; previous = initial_seed; park_q = None
        for i, t in enumerate(self.arm_times):
            state = self.state(t); goal = state['hand']
            row = next((r for r in self.segments if t <= r['end']), self.segments[-1])
            reverse = row.get('reverse_source')
            if reverse is not None:
                source_t = reverse['end']-(t-row['start'])
                seed = np.array([np.interp(source_t, self.arm_times[:i], np.asarray(qs)[:, j]) for j in range(7)])
            else:
                seed = previous if previous is not None else self.arm.solve(goal[:3, 3], -goal[:3, 2])
            free_joint_return = row['label'].startswith('install_') and row['label'].endswith('_clear')
            reverse_joint_return = reverse is not None and reverse['label'].endswith('_clear')
            def residual(q):
                p, R, links = self.arm.forward(q, geometry=True)
                heights = np.array([(v@r.T+x)[:,2].min() for v,(r,x)
                                    in zip(self.arm.collisions[1:],links[1:])])
                error = Rotation.from_matrix(goal[:3, :3]@R.T).as_rotvec()
                return np.r_[p-goal[:3, 3], .15*error, 3*np.minimum(heights-.002,0.), .00001*(q-seed)]
            if free_joint_return:
                if 'clear_start_q' not in row:
                    row['clear_start_q'] = previous.copy()
                u = smooth((t-row['start'])/(row['end']-row['start']))
                q = (1-u)*row['clear_start_q']+u*park_q
            elif reverse_joint_return:
                q = seed
            else:
                fit = least_squares(residual, seed, bounds=(self.arm.lower,self.arm.upper),
                                    max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                q = fit.x
            p, R, transforms = self.arm.forward(q, geometry=True)
            pe = np.linalg.norm(p-goal[:3, 3])
            re = np.linalg.norm(Rotation.from_matrix(goal[:3, :3]@R.T).as_rotvec())
            floor = min((v@r.T+x)[:,2].min() for v,(r,x) in zip(self.arm.collisions[1:],transforms[1:]))
            if ((not free_joint_return and not reverse_joint_return) and (pe > 1e-4 or re > 1e-3)) or floor < -.001:
                raise RuntimeError(f'Full-orientation IK failed at {t:.3f} s ({state["phase"]}): position={pe}, angle={re}, floor={floor}, joints={np.rad2deg(q).round(2)}')
            qs.append(q); previous = q
            if park_q is None:
                park_q = q.copy()
            if i%160 == 0:
                print(f'KUKA full-pose IK {t:.0f}/{self.duration:.0f} s', flush=True)
        self.arm_q = np.asarray(qs)
        ratio = float(np.max(np.abs(np.diff(self.arm_q,axis=0))/np.diff(self.arm_times)[:,None]/self.arm.velocity))
        self.time_scale = max(1., ratio*1.05)
        if self.time_scale > 5.:
            raise RuntimeError(f'IK continuity needs excessive slowdown: {self.time_scale:.3f}')
        return dict(ik_sample_hz=sample_hz, full_tool_orientation=True,
                    unloaded_return_to_park_in_joint_space=True, reverse_unloading_reuses_loading_ik=True,
                    time_scale=self.time_scale,
                    maximum_joint_speed_ratio=ratio/self.time_scale)

    def joints(self, t):
        return np.array([np.interp(t,self.arm_times,self.arm_q[:,i]) for i in range(7)])
