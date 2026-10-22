"""Native LBR Med IK ported from codes/simulation/kuka.py; 120mm needle."""
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from codes.simulation.kuka import description,ASSETS,visual_parts

class Arm:
    def __init__(self,base,rotation,tool=.155):
        links, joints, limits = description()
        self.base = np.asarray(base).copy(); self.tool=tool
        self.rotation = np.asarray(rotation).copy()
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
        tcp = point + rotation[:, 2]*self.tool
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
            candidate = (error + alignment*.1 + max(0., self.base[2]-.001-floor)*100, fit.x, error, alignment, floor)
            if best is None or candidate[0] < best[0]: best = candidate
            if error < 1e-5 and alignment < 1e-4 and floor >= self.base[2]-.001:
                return fit.x
        if best[2] > 1e-4 or best[3] > 1e-3 or best[4] < self.base[2]-.001:
            raise ValueError(f'LBR Med IK unresolved: position {best[2]:.4g} m, axis {best[3]:.4g}, floor {best[4]:.4g} m; q={np.rad2deg(best[1]).round(2).tolist()}')
        return best[1]
