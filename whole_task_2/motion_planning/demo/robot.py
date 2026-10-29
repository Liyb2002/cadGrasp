"""SO101 carrier kinematics, including its rotating jaw. All internal lengths are metres."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.optimize import least_squares, brentq
from scipy.spatial.transform import Rotation

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
LINKS = ('base_link', 'shoulder_link', 'upper_arm_link', 'lower_arm_link', 'wrist_link',
         'gripper_link', 'moving_jaw_so101_v1_link')


def transform(rotation=None, translation=None):
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return t


def points(p, t):
    return np.asarray(p) @ t[:3, :3].T + t[:3, 3]


def origin(element):
    if element is None:
        return np.eye(4)
    return transform(Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix(),
                     np.fromstring(element.get('xyz', '0 0 0'), sep=' '))


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-7)):
        raise ValueError('Expected a finite rigid 4x4 transform, with translation in metres')
    return t


class Robot:
    def __init__(self, config, directory, gripper, policy):
        self.config, self.policy = config, policy
        path = (Path(directory) / config['urdf']).resolve()
        self.asset_root = path.parent
        root = ET.parse(path).getroot()
        joints = {j.get('name'): j for j in root.findall('joint')}
        self.base = rigid(config['world_from_base'])
        self.origins = [origin(joints[n].find('origin')) for n in JOINTS]
        self.axes = [np.fromstring(joints[n].find('axis').get('xyz'), sep=' ') for n in JOINTS]
        self.lower = np.array([float(joints[n].find('limit').get('lower')) for n in JOINTS])
        self.upper = np.array([float(joints[n].find('limit').get('upper')) for n in JOINTS])
        self.tcp_local = origin(joints['gripper_frame_joint'].find('origin'))
        self.jaw_local = origin(joints['gripper'].find('origin'))
        self.jaw_axis = np.fromstring(joints['gripper'].find('axis').get('xyz'), sep=' ')
        self.jaw_limits = np.array([float(joints['gripper'].find('limit').get(k)) for k in ('lower', 'upper')])
        self.fixed_pad = np.asarray(gripper['fixed_contact_gripper_m'], float)
        at_zero = np.r_[gripper['moving_contact_at_zero_gripper_m'], 1.]
        self.moving_pad_local = (np.linalg.inv(self.jaw_local) @ self.tcp_local @ at_zero)[:3]
        self.gripper = gripper
        self.signs = np.asarray(config.get('hardware_signs', [1] * 6), float)
        self.offsets = np.asarray(config.get('hardware_zero_offsets_deg', [0] * 6), float)
        if self.signs.shape != (6,) or not np.isin(self.signs, [-1, 1]).all() or self.offsets.shape != (6,):
            raise ValueError('Six hardware signs (+1/-1) and six model zero offsets are required')
        self.parts = []
        cache = {}
        for link in root.findall('link'):
            name = link.get('name')
            if name not in LINKS:
                continue
            for collision in link.findall('collision'):
                spec = collision.find('geometry/mesh')
                file = (path.parent / spec.get('filename')).resolve()
                if file not in cache:
                    cache[file] = trimesh.load(file, force='mesh', process=True)
                mesh = cache[file].copy()
                mesh.vertices *= np.fromstring(spec.get('scale', '1 1 1'), sep=' ')
                mesh.apply_transform(origin(collision.find('origin')))
                contact = ('fixed' if file.name == 'wrist_roll_follower_so101_v1.stl' else
                           'moving' if name == LINKS[-1] else None)
                self.parts.append(dict(link=name, level=LINKS.index(name), mesh=mesh, contact=contact,
                                       table_vertices=mesh.convex_hull.vertices,
                                       name=f'{name}/{file.stem}'))
        self.reach_bound = sum(np.linalg.norm(o[:3, 3]) for o in self.origins) + .25
        self.asset_paths = [path, *cache]

    def frames(self, q, jaw=0., jacobian=False):
        t = self.base.copy()
        out = {LINKS[0]: t.copy()}
        axes, origins = [], []
        for i, (o, axis, angle) in enumerate(zip(self.origins, self.axes, q)):
            t = t @ o
            axes.append(t[:3, :3] @ axis)
            origins.append(t[:3, 3].copy())
            t = t @ transform(Rotation.from_rotvec(axis * angle).as_matrix())
            out[LINKS[i + 1]] = t.copy()
        tcp = t @ self.tcp_local
        out[LINKS[-1]] = t @ self.jaw_local @ transform(Rotation.from_rotvec(self.jaw_axis * jaw).as_matrix())
        if jacobian:
            axes, origins = np.asarray(axes), np.asarray(origins)
            return tcp, np.vstack([np.cross(axes, tcp[:3, 3] - origins).T, axes.T])
        return tcp, out

    def moving_pad(self, jaw):
        t = np.linalg.inv(self.tcp_local) @ self.jaw_local @ transform(Rotation.from_rotvec(self.jaw_axis * jaw).as_matrix())
        return points(self.moving_pad_local, t)

    def pad_frame(self, gripper_pose, jaw, which):
        """Physical pad coordinates. The moving pad's box rotates with the actual hinged finger."""
        if which == 'fixed':
            return gripper_pose @ transform(translation=self.fixed_pad)
        zero = np.linalg.inv(self.tcp_local) @ self.jaw_local
        current = zero @ transform(Rotation.from_rotvec(self.jaw_axis * jaw).as_matrix())
        return gripper_pose @ current @ transform(zero[:3, :3].T, self.moving_pad_local)

    def jaw_for_width(self, width):
        lo = max(self.jaw_limits[0], np.radians(-9.))
        hi = min(self.jaw_limits[1], np.radians(self.gripper['maximum_jaw_angle_deg']))
        f = lambda a: np.linalg.norm(self.moving_pad(a) - self.fixed_pad) - width
        if width > self.gripper['maximum_grasp_width_m'] or f(lo) > 0 or f(hi) < 0:
            raise ValueError('Contact span outside the configured SO101 jaw range')
        return brentq(f, lo, hi, xtol=1e-12)

    def ik(self, target, seeds, rng=None, multistart=0):
        """Full rigid pose IK. A five-axis arm may have no solution; never discard an orientation residual."""
        target = rigid(target)
        starts = list(seeds)
        if rng is not None and multistart:
            starts.extend(rng.uniform(self.lower, self.upper, (multistart, 5)))
        for seed in starts:
            def residual(q):
                t = self.frames(q)[0]
                return np.r_[t[:3, 3] - target[:3, 3],
                             .12 * Rotation.from_matrix(target[:3, :3] @ t[:3, :3].T).as_rotvec()]
            fit = least_squares(residual, np.clip(seed, self.lower + 1e-8, self.upper - 1e-8),
                                bounds=(self.lower, self.upper), max_nfev=90,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            t = self.frames(fit.x)[0]
            pos = np.linalg.norm(t[:3, 3] - target[:3, 3])
            ang = Rotation.from_matrix(target[:3, :3] @ t[:3, :3].T).magnitude()
            if pos <= self.policy['position_tolerance_m'] and ang <= np.radians(self.policy['orientation_tolerance_deg']):
                return fit.x
        return None

    def hardware_degrees(self, q6):
        """model_deg = hardware_signs * hardware_deg + hardware_zero_offsets_deg."""
        return (np.degrees(q6) - self.offsets) / self.signs
