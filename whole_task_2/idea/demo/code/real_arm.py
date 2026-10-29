"""Measured SO101 adapter for the demo only. The Step1--Step5 sources stay unchanged."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from whole_task_2.step5.video.so101 import Arm, turn
from whole_task_2.step1 import arm as A
from whole_task_2.step1.reach import tip_contact


def prism(start, end, radius, sides=12):
    """The video planner's conservative circumscribed tool proxy, without importing the video stack."""
    a = np.arange(sides)*2*np.pi/sides; r = radius/np.cos(np.pi/sides)
    ring = np.c_[np.cos(a)*r, np.sin(a)*r]
    return np.vstack([np.c_[ring, np.full(sides, start)], np.c_[ring, np.full(sides, end)]])


def pieces(vertices):
    from scipy.spatial import ConvexHull
    result = []
    for v in vertices:
        h = ConvexHull(v); c = v.mean(0)
        result.append((h.equations[:, :3], h.equations[:, 3], c, float(np.linalg.norm(v-c, axis=1).max())))
    return result


def scene(mesh, pose):
    triangles = mesh.triangles @ pose[:3, :3].T + pose[:3, 3]; centre = triangles.mean(1)
    return triangles, cKDTree(centre), float(np.linalg.norm(triangles-centre[:, None], axis=2).max())


def meets(hulls, frames, geometry):
    """The core's exact clipped-triangle/hull collision check, with the same broad phase."""
    triangles, tree, rho = geometry
    for (n, d, c, r), T in zip(hulls, frames):
        rotation, p = T[:3, :3], T[:3, 3]; near = tree.query_ball_point(rotation@c+p, r+rho)
        if not near: continue
        X = (triangles[near]-p)@rotation; distance = X@n.T+d
        if (distance <= 0).all(2).any(): return True
        if any(A.clipped(X[k], n, d) for k in np.flatnonzero(~(distance > 0).all(1).any(1))): return True
    return False


def tool_frame(point, direction):
    z = -np.asarray(direction); x = np.cross(z, [1., 0, 0] if abs(z[0]) < .9 else [0, 1., 0]); x /= np.linalg.norm(x)
    T = np.eye(4); T[:3, 0] = x; T[:3, 1] = np.cross(z, x); T[:3, 2] = z; T[:3, 3] = point
    return T


def tcp_matrix(value):
    if isinstance(value, dict):
        T = np.eye(4)
        T[:3, :3] = value['rotation_flange_tcp']
        T[:3, 3] = value['translation_flange_m']
    else:
        T = np.asarray(value, float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1]) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9) or
            not np.isclose(np.linalg.det(T[:3, :3]), 1)):
        raise ValueError('TCP must be a rigid gripper_frame_from_process_tcp transform in metres')
    return T.copy()


class RealArm(Arm):
    def __init__(self, station):
        super().__init__(station['base_position_world_m'], station['base_rotation_world'])
        self.tcp = tcp_matrix(station['feasibility']['tcp_transform'])
        self.table = float(station['table_top_z_m'])
        self.velocity = np.full(5, np.radians(station['maximum_joint_speed_deg_s']))
        self.nominal = np.asarray(station['working_direction_world'], float)
        self.cone = float(station['cone_half_angle_deg'])
        self.cloud = None

    def forward(self, q, geometry=False):
        T = np.eye(4); T[:3, :3] = self.rotation; T[:3, 3] = self.base
        transforms = [(T[:3, :3].copy(), T[:3, 3].copy())]; origins, axes = [], []
        for O, axis, angle in zip(self.origins, self.axes, q):
            T = T @ O; origins.append(T[:3, 3].copy()); axes.append(T[:3, :3] @ axis)
            T = T @ turn(axis, angle); transforms.append((T[:3, :3].copy(), T[:3, 3].copy()))
        F = T @ self.frame @ self.tcp; p, d = F[:3, 3], F[:3, 2]
        if geometry: return p, F[:3, :3], transforms
        return p, d, np.cross(axes, p - np.asarray(origins)).T, np.cross(axes, d).T

    def sample(self, count=160000, seed=20261025):
        """An IK seed/ranking cloud only. A cloud match never certifies a stop."""
        rng = np.random.default_rng(seed)
        Q = rng.uniform(self.lower, self.upper, (count, 5))
        T = np.repeat(np.eye(4)[None], count, axis=0); T[:, :3, :3] = self.rotation; T[:, :3, 3] = self.base
        for O, axis, angles in zip(self.origins, self.axes, Q.T):
            T = T @ O
            from scipy.spatial.transform import Rotation
            R = np.repeat(np.eye(4)[None], count, axis=0)
            R[:, :3, :3] = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
            T = T @ R
        F = T @ self.frame @ self.tcp
        keep = (F[:, :3, 2] @ self.nominal >= np.cos(np.radians(self.cone + 8))) & (F[:, 2, 3] >= self.table)
        self.cloud = (F[keep, :3, 3].copy(), F[keep, :3, 2].copy(), Q[keep])
        if len(self.cloud[0]) < 10: raise ValueError('Too few FK seeds in the process cone')
        self.tree = cKDTree(np.c_[self.cloud[0], .12 * self.cloud[1]])
        return self.cloud

    def nearest(self, point, axis, k=1):
        return self.tree.query(np.concatenate([point, .12 * np.broadcast_to(axis, np.shape(point))], axis=-1), k=k)

    def solve(self, point, outward, previous=None):
        point, target = np.asarray(point, float), -np.asarray(outward, float)
        seeds = [] if previous is None else [np.asarray(previous, float)]
        if self.cloud is not None:
            _, ids = self.nearest(point, target, k=3)
            seeds += list(self.cloud[2][np.atleast_1d(ids)])
        seeds += self.seeds[:3]
        for seed in seeds:
            def fun(q):
                p, d, _, _ = self.forward(q)
                return np.r_[p - point, .18 * (d - target), 1e-6 * (q - seed)]
            def jac(q):
                _, _, jp, jd = self.forward(q)
                return np.vstack([jp, .18 * jd, 1e-6 * np.eye(5)])
            fit = least_squares(fun, np.clip(seed, self.lower, self.upper), jac=jac,
                                bounds=(self.lower, self.upper), max_nfev=90,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            p, d, _, _ = self.forward(fit.x)
            if (np.linalg.norm(p - point) < 1e-5 and np.linalg.norm(d - target) < 1e-4 and
                    self.floor(self.forward(fit.x, geometry=True)[2]) >= self.table - 1e-9):
                return fit.x
        raise ValueError('No SO101 IK above the measured fixed tabletop')


class SceneCheck(A.Check):
    """Reuse the exact triangle/link test with this adapter, without patching core globals."""
    def __init__(self, source, arm, station, support=None, support_pose=None):
        from scipy.spatial import ConvexHull
        self.mesh, self.arm, self.floor = source, arm, arm.table
        self.table = None                       # moving links checked against the infinite table plane below
        self.table_bounds = np.array([[-2, -2, self.floor - .05], [2, 2, self.floor]])
        self.work_bounds = None
        self.length = station['needle_length_m']; self.needle_radius = station['needle_radius_m']
        self.head_radius = station['head_radius_m']; self.placed = {}; self.hulls = []
        for v in arm.collisions:
            h = ConvexHull(v); c = v.mean(0)
            self.hulls.append((h.equations[:, :3], h.equations[:, 3], c, float(np.linalg.norm(v - c, axis=1).max())))
        self.support = None
        if support is not None:
            self.support = SceneCheck(support, arm, station)
            self.support_pose = np.asarray(support_pose)
            self.support_scene = scene(support, self.support_pose)

    def clear(self, q, object_pose, allowance=0.):
        p, d, _, _ = self.arm.forward(q)
        if self.arm.floor(self.arm.forward(q, geometry=True)[2]) < self.floor - 1e-9: return False
        if self.hits(q, object_pose): return False
        if self.support is not None:
            if self.support.hits(q, self.support_pose): return False
            hulls = pieces([prism(allowance, self.length, self.needle_radius),
                            prism(self.length, self.length + .07, self.head_radius)])
            T = tool_frame(p, d)
            if meets(hulls, [T, T], self.support_scene): return False
        return True

    def face(self, face, axis_object, pose):
        from whole_task_2.step1.reach import test_points
        d = pose[:3, :3] @ axis_object
        if not self.tool_above_table(face, axis_object, pose): return None
        allowance = float(tip_contact(self.needle_radius, -self.mesh.face_normals[face] @ axis_object))
        Q = []; previous = None
        for point in test_points(self.mesh)[face] @ pose[:3, :3].T + pose[:3, 3]:
            try: q = self.arm.solve(point, -d, previous)
            except ValueError: return None
            if not self.clear(q, pose, allowance): return None
            Q.append(q); previous = q
        return np.asarray(Q)
