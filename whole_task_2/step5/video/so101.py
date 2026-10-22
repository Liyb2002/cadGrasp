"""SO-101 work arm (TheRobotStudio SO-ARM100, Apache-2.0), with the interface of kuka.Arm.

The chain base -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll comes from
codes/simulation/assets/so101/so101_new_calib.urdf; the jaw stays at 0, closed on the needle holder.  The
needle is held like a pen, across the jaws: along the gripper frame's y axis, perpendicular to the
wrist-roll axis, so that wrist flex and wrist roll together set its direction (five joints for the tip
position and the needle direction).  Its tip is `tool` along it from the gripper frame's origin.
"""
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT

ASSETS = ROOT/'codes/simulation/assets/so101'
CHAIN = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
LINKS = ('base_link', 'shoulder_link', 'upper_arm_link', 'lower_arm_link', 'wrist_link', 'gripper_link')
JAW = 'moving_jaw_so101_v1_link'
NEEDLE = np.array([0., 1., 0.])                       # the needle, gripper frame
COLORS = {'3d_printed': (.93, .91, .86), 'sts3215': (.16, .16, .18)}


def origin(element):
    T = np.eye(4)
    if element is not None:
        T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        T[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return T


@lru_cache(None)
def description():
    root = ET.parse(ASSETS/'so101_new_calib.urdf').getroot()
    return {j.get('name'): j for j in root.findall('joint')}, {k.get('name'): k for k in root.findall('link')}


@lru_cache(None)
def parts(link, kind):
    """(link <- mesh, mesh, material) of every visual or collision mesh of a link."""
    _, links = description(); out = []
    for e in links[link].findall(kind):
        material = e.find('material')
        out.append((origin(e.find('origin')), trimesh.load(ASSETS/e.find('geometry/mesh').get('filename'), force='mesh'),
                    material.get('name') if material is not None else '3d_printed'))
    return out


def turn(axis, angle):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(axis*angle).as_matrix(); return T


class Arm:
    dof = len(CHAIN)

    def __init__(self, base, rotation, tool=.17):
        joints, _ = description()
        self.base = np.asarray(base, float).copy(); self.rotation = np.asarray(rotation, float).copy(); self.tool = tool
        self.origins = [origin(joints[n].find('origin')) for n in CHAIN]
        self.axes = [np.fromstring(joints[n].find('axis').get('xyz'), sep=' ') for n in CHAIN]
        self.lower = np.array([float(joints[n].find('limit').get('lower')) for n in CHAIN])
        self.upper = np.array([float(joints[n].find('limit').get('upper')) for n in CHAIN])
        self.frame = origin(joints['gripper_frame_joint'].find('origin'))          # gripper link <- gripper frame
        self.jaw = origin(joints['gripper'].find('origin'))                          # gripper link <- jaw at 0
        self.collisions = [np.vstack([trimesh.transform_points(m.vertices, T) for T, m, _ in parts(k, 'collision')])
                           for k in LINKS]
        self.collisions[-1] = np.vstack([self.collisions[-1], *[trimesh.transform_points(m.vertices, self.jaw@T)
                                                                 for T, m, _ in parts(JAW, 'collision')]])
        self.seeds = [np.zeros(5), np.array([0., -.6, .9, .6, 0.]), np.array([0., .6, -.9, -.6, 0.]),
                      np.array([0., -.6, .9, .6, 1.57]), np.array([0., -.6, .9, .6, -1.57])]
        self.seeds += list(np.random.default_rng(17).uniform(self.lower*.8, self.upper*.8, (24, 5)))

    def forward(self, q, geometry=False):
        T = np.eye(4); T[:3, :3] = self.rotation; T[:3, 3] = self.base
        transforms = [(T[:3, :3].copy(), T[:3, 3].copy())]; origins, axes = [], []
        for O, axis, angle in zip(self.origins, self.axes, q):
            T = T@O; origins.append(T[:3, 3].copy()); axes.append(T[:3, :3]@axis)
            T = T@turn(axis, angle); transforms.append((T[:3, :3].copy(), T[:3, 3].copy()))
        F = T@self.frame; d = F[:3, :3]@NEEDLE; tcp = F[:3, 3]+d*self.tool
        if geometry: return tcp, F[:3, :3], transforms
        axes, origins = np.asarray(axes), np.asarray(origins)
        return tcp, d, np.cross(axes, tcp-origins).T, np.cross(axes, d).T

    def floor(self, transforms):
        return min((v@R.T+p)[:, 2].min() for v, (R, p) in zip(self.collisions[1:], transforms[1:]))

    def solve(self, point, outward, previous=None):
        target = -np.asarray(outward, float); point = np.asarray(point, float); best = None
        for seed in ([previous] if previous is not None else self.seeds):
            def residual(q):
                tcp, axis, _, _ = self.forward(q)
                return np.r_[tcp-point, .18*(axis-target), 1e-5*(q-seed)]

            def jacobian(q):
                _, _, jp, ja = self.forward(q)
                return np.vstack([jp, .18*ja, 1e-5*np.eye(5)])
            fit = least_squares(residual, np.clip(seed, self.lower, self.upper), jac=jacobian, bounds=(self.lower, self.upper),
                                max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            tcp, axis, _, _ = self.forward(fit.x)
            error, alignment = np.linalg.norm(tcp-point), np.linalg.norm(axis-target)
            low = self.floor(self.forward(fit.x, geometry=True)[2])
            candidate = (error+alignment*.1+max(0., self.base[2]-.001-low)*100, fit.x, error, alignment, low)
            if best is None or candidate[0] < best[0]: best = candidate
            if error < 1e-5 and alignment < 1e-4 and low >= self.base[2]-.001: return fit.x
        if best[2] > 1e-4 or best[3] > 1e-3 or best[4] < self.base[2]-.001:
            raise ValueError(f'SO101 IK unresolved: position {best[2]:.4g} m, axis {best[3]:.4g}, floor {best[4]:.4g} m')
        return best[1]

    def visuals(self):
        """(transform index, vertices, faces, colour) of every visual mesh, in its link's frame."""
        out = [(i, trimesh.transform_points(m.vertices, T), m.faces, COLORS.get(name, COLORS['3d_printed']))
               for i, k in enumerate(LINKS) for T, m, name in parts(k, 'visual')]
        return out+[(len(LINKS)-1, trimesh.transform_points(m.vertices, self.jaw@T), m.faces, COLORS.get(name, COLORS['3d_printed']))
                    for T, m, name in parts(JAW, 'visual')]
