"""Pinned LBR Med 14 R820 meshes, seven-axis kinematics and bounded IK."""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh
import yaml

ASSETS = Path(__file__).resolve().parent / 'assets/lbr_med14_r820'
BASE_A = np.array([-.75, -.15, 0.])
BASE_B = np.array([.80, .28, 0.])
TOOL = .135  # link_7 -> upstream flange (.035 m) + a 100 mm tool


def numbers(value):
    return ' '.join(format(float(x), '.17g') for x in np.asarray(value).ravel())


@lru_cache(None)
def description():
    root = ET.parse(ASSETS / 'urdf/lbr_med14_r820_macro.xacro').getroot()[0]
    links = root.findall('link')[:8]
    joints = [joint for joint in root.findall('joint') if joint.get('type') == 'revolute']
    limits = yaml.safe_load((ASSETS / 'config/joint_limits.yaml').read_text())
    return links, joints, limits


@lru_cache(None)
def visual_parts(index):
    scene = trimesh.load(ASSETS / f'meshes/visual/link_{index}.dae', force='scene')
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        rgba = np.asarray(mesh.visual.material.baseColorFactor) / 255.
        parts.append((trimesh.transform_points(mesh.vertices, transform), mesh.faces, rgba))
    return parts


def add_robots(root):
    """Use actual upstream link origins and geometry, without scaling the robot."""
    links, joints, limits = description()
    asset, world = root.find('asset'), root.find('worldbody')
    for index in range(8):
        for part, (vertices, faces, _) in enumerate(visual_parts(index)):
            ET.SubElement(asset, 'mesh', name=f'med14_{index}_{part}', vertex=numbers(vertices),
                          face=' '.join(map(str, faces.ravel())))
    for name, base, yaw in (('A', BASE_A, 0.), ('B', BASE_B, np.pi)):
        body = ET.SubElement(world, 'body', name=f'robot_{name}_link_0', pos=numbers(base),
                             quat=numbers([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]))
        for index, link in enumerate(links):
            if index:
                joint = joints[index-1]
                body = ET.SubElement(body, 'body', name=f'robot_{name}_link_{index}',
                                     pos=joint.find('origin').get('xyz'))
                limit = limits[f'A{index}']
                ET.SubElement(body, 'joint', name=f'robot_{name}_joint_{index}', type='hinge',
                              axis=joint.find('axis').get('xyz'), limited='true',
                              range=numbers(np.deg2rad([limit['lower'], limit['upper']])))
            inertia = link.find('inertial')
            values = inertia.find('inertia')
            ET.SubElement(body, 'inertial', pos=inertia.find('origin').get('xyz'),
                          mass=inertia.find('mass').get('value'),
                          fullinertia=' '.join(values.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')))
            for part, (_, _, rgba) in enumerate(visual_parts(index)):
                # The upstream logos are dark material patches on grey rings.
                # Match the ring colour, preserving the actual mesh geometry.
                if index in (2, 4) and np.allclose(rgba[:3], .2):
                    rgba = np.array([.6, .6, .6, 1.])
                ET.SubElement(body, 'geom', name=f'robot_{name}_visual_{index}_{part}', type='mesh',
                              mesh=f'med14_{index}_{part}', pos=link.find('visual/origin').get('xyz'),
                              rgba=numbers(rgba), contype='0', conaffinity='0', group='2')
        ET.SubElement(body, 'geom', name=f'robot_{name}_tool_stem', type='capsule',
                      fromto='0 0 .035 0 0 .126', size='.006', rgba='.25 .28 .3 1',
                      contype='0', conaffinity='0', group='2')
        if name == 'A':
            # Open pushing palm; there is no enclosing gripper on the object.
            ET.SubElement(body, 'geom', name='robot_A_palm', type='box', pos='0 0 .132',
                          size='.019 .014 .003', rgba='.22 .25 .28 1',
                          contype='0', conaffinity='0', group='2')
        else:
            for side in (-1, 1):
                ET.SubElement(body, 'geom', name=f'robot_B_finger_{side}', type='box',
                              pos=f'{side*.014} 0 .115', size='.004 .01 .02',
                              rgba='.22 .25 .28 1', contype='0', conaffinity='0', group='2')


class Arm:
    def __init__(self, name):
        links, joints, limits = description()
        self.base = BASE_A.copy() if name == 'A' else BASE_B.copy()
        self.rotation = Rotation.from_euler('z', 0 if name == 'A' else np.pi).as_matrix()
        self.offsets = [np.fromstring(joint.find('origin').get('xyz'), sep=' ') for joint in joints]
        self.axes = [np.fromstring(joint.find('axis').get('xyz'), sep=' ') for joint in joints]
        self.lower = np.deg2rad([limits[f'A{i}']['lower'] for i in range(1,8)])
        self.upper = np.deg2rad([limits[f'A{i}']['upper'] for i in range(1,8)])
        self.velocity = np.deg2rad([limits[f'A{i}']['velocity'] for i in range(1,8)])
        self.collisions = []
        for index, link in enumerate(links):
            mesh = trimesh.load(ASSETS / f'meshes/collision/link_{index}.stl', force='mesh')
            offset = np.fromstring(link.find('collision/origin').get('xyz'), sep=' ')
            self.collisions.append(mesh.vertices + offset)

    def forward(self, q, geometry=False):
        rotation, point = self.rotation.copy(), self.base.copy()
        transforms = [(rotation.copy(), point.copy())]
        origins, axes = [], []
        for offset, axis, angle in zip(self.offsets, self.axes, q):
            point = point + rotation @ offset
            origins.append(point.copy())
            axes.append(rotation @ axis)
            rotation = rotation @ Rotation.from_rotvec(axis * angle).as_matrix()
            transforms.append((rotation.copy(), point.copy()))
        tcp = point + rotation[:, 2]*TOOL
        if geometry:
            return tcp, rotation, transforms
        axes, origins = np.asarray(axes), np.asarray(origins)
        position_jac = np.cross(axes, tcp - origins).T
        direction_jac = np.cross(axes, rotation[:, 2]).T
        return tcp, rotation[:, 2], position_jac, direction_jac

    def solve(self, point, outward, previous=None):
        target_axis = -np.asarray(outward)
        point = np.asarray(point)
        if previous is None:
            seeds = [np.deg2rad([0, 55, 0, -95, 0, 90, 0]),
                     np.deg2rad([0, 80, 90, 95, -90, 40, 0]),
                     np.deg2rad([90, 70, 0, -95, 0, 90, 0])]
            seeds += list(np.random.default_rng(17).uniform(self.lower*.7, self.upper*.7, (20,7)))
        else:
            seeds = [previous]
        best = None
        for seed in seeds:
            def residual(q):
                tcp, axis, _, _ = self.forward(q)
                return np.r_[tcp-point, .18*(axis-target_axis), .00001*(q-seed)]
            def jacobian(q):
                _, _, jp, ja = self.forward(q)
                return np.vstack([jp, .18*ja, .00001*np.eye(7)])
            fit = least_squares(residual, seed, jac=jacobian, bounds=(self.lower, self.upper),
                                max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            tcp, axis, _, _ = self.forward(fit.x)
            error = np.linalg.norm(tcp-point)
            alignment = np.linalg.norm(axis-target_axis)
            _, _, transforms = self.forward(fit.x, geometry=True)
            floor = min((v@r.T+p)[:,2].min() for v,(r,p) in zip(self.collisions[1:],transforms[1:]))
            candidate = (error + alignment*.1 + max(0., -.001-floor)*100, fit.x, error, alignment, floor)
            if best is None or candidate[0] < best[0]: best = candidate
            if error < 1e-5 and alignment < 1e-4 and floor >= -.001:
                return fit.x
        if best[2] > 1e-4 or best[3] > 1e-3 or best[4] < -.001:
            raise ValueError(f'LBR Med IK unresolved: position {best[2]:.4g} m, axis {best[3]:.4g}, floor {best[4]:.4g} m; q={np.rad2deg(best[1]).round(2).tolist()}')
        return best[1]
