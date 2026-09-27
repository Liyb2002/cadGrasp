"""Piecewise pivoting on exact supporting vertices: no lift and no floor slip."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


class GroundRoll:
    def __init__(self, vertices, initial_rotation):
        self.vertices = np.asarray(vertices, dtype=float)
        vector = Rotation.from_matrix(initial_rotation).as_rotvec()
        self.angle = float(np.linalg.norm(vector))
        self.axis = vector / self.angle
        points = self.vertices[ConvexHull(self.vertices).vertices]
        parallel = (points @ self.axis)[:, None] * self.axis
        # z(R(theta) v) = a cos(theta) + b sin(theta) + c.
        self.coefficients = np.c_[points[:, 2] - parallel[:, 2],
                                    np.cross(self.axis, points)[:, 2], parallel[:, 2]]
        a, b, c = (self.coefficients[:, None, k] - self.coefficients[None, :, k]
                   for k in range(3))
        radius = np.hypot(a, b)
        valid = (radius > 1e-12) & (abs(c) <= radius + 1e-14)
        phase = np.arctan2(b[valid], a[valid])
        offset = np.arccos(np.clip(-c[valid] / radius[valid], -1., 1.))
        roots = [0., self.angle]
        for sign in (-1, 1):
            for winding in (-1, 0, 1):
                values = phase + sign * offset + winding * 2 * np.pi
                roots.extend(values[(values > 1e-10) & (values < self.angle - 1e-10)])
        roots = np.unique(np.asarray(roots))
        self.segments = []
        for low, high in zip(roots[:-1], roots[1:]):
            if high - low < 1e-12:
                continue
            mid = .5 * (low + high)
            index = int(np.argmin(self.coefficients @ [np.cos(mid), np.sin(mid), 1.]))
            if self.segments and self.segments[-1]['index'] == index:
                self.segments[-1]['high'] = high
            else:
                self.segments.append(dict(low=low, high=high, index=index, pivot=points[index]))
        translation = np.zeros(3)
        for segment in self.segments:
            low, high, pivot = segment['low'], segment['high'], segment['pivot']
            segment['translation'] = translation.copy()
            segment['contact'] = self.rotation(low) @ pivot + translation
            translation += self.rotation(low) @ pivot - self.rotation(high) @ pivot
        self.ends = np.array([segment['high'] for segment in self.segments])
        self.verify_continuous_floor_contact()

    def rotation(self, angle):
        return Rotation.from_rotvec(angle * self.axis).as_matrix()

    def pose(self, fraction):
        """fraction=0 is lying; fraction=1 is the exact saved target pose."""
        angle = self.angle * (1 - np.clip(fraction, 0, 1))
        if angle == 0:
            return np.eye(4)
        segment = self.segments[min(np.searchsorted(self.ends, angle), len(self.segments)-1)]
        rotation = self.rotation(angle)
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = segment['contact'] - rotation @ segment['pivot']
        return result

    def verify_continuous_floor_contact(self):
        # Check analytic minima over each entire arc, not only animation frames.
        for segment in self.segments:
            delta = self.vertices - segment['pivot']
            parallel = (delta @ self.axis)[:, None] * self.axis
            a = delta[:, 2] - parallel[:, 2]
            b = np.cross(self.axis, delta)[:, 2]
            c = parallel[:, 2] + segment['contact'][2]
            low, high = segment['low'], segment['high']
            values = [a*np.cos(low)+b*np.sin(low)+c, a*np.cos(high)+b*np.sin(high)+c]
            phase = np.arctan2(b, a)
            for winding in range(-2, 3):
                theta = phase + winding*np.pi
                inside = (theta >= low) & (theta <= high)
                values.append((a*np.cos(theta)+b*np.sin(theta)+c)[inside])
            assert min(float(v.min()) for v in values if len(v)) > -1e-9
            assert abs(segment['contact'][2]) < 1e-9
