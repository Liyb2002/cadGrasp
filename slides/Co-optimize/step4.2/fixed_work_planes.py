"""Deterministic work-prism coefficients for fixed task/seat orientations.

The original inequalities and their order are retained. Only their offsets
depend on translation. Exit sweeps remain separate, direction-dependent data.
"""
import numpy as np


class FixedWorkPlanes:
    def __init__(self, access):
        self.local = []
        self.slices = []
        for work in access:
            local = tuple(p[:n].copy() for p, n in zip(work.planes, work.counts))
            for p in local:
                p.flags.writeable = False
            self.local.append(local)
            ends = np.cumsum([0]+[len(p) for p in local])
            self.slices.append(tuple(slice(a, b) for a, b in zip(ends[:-1], ends[1:])))
        self.flat = [np.vstack(p) for p in self.local]
        self.orientations = {}
        self.ground_orientations = {}
        self.inverse_rotations = {}
        self.statistics = dict(orientation_preparations=0, ground_orientation_preparations=0,
                               translation_updates=0, ground_translation_updates=0,
                               sampled_points=0)

    def inverse(self, matrix):
        key = matrix[:3, :3].tobytes()
        rotation = self.inverse_rotations.get(key)
        if rotation is None:
            rotation = np.linalg.inv(matrix[:3, :3])
            self.inverse_rotations[key] = rotation
        inverse = np.eye(4)
        inverse[:3, :3] = rotation
        inverse[:3, 3] = -rotation @ matrix[:3, 3]
        return inverse

    def transformed(self, k, inverse):
        key = (int(k), inverse[:3, :3].tobytes())
        normals = self.orientations.get(key)
        if normals is None:
            normals = self.flat[k][:, :3] @ inverse[:3, :3]
            self.orientations[key] = normals
            self.statistics['orientation_preparations'] += 1
        out = np.empty_like(self.flat[k])
        out[:, :3] = normals
        out[:, 3] = self.flat[k][:, :3] @ inverse[:3, 3] + self.flat[k][:, 3]
        self.statistics['translation_updates'] += 1
        return tuple(out[s] for s in self.slices[k])

    def on_ground(self, task, inverse, width):
        # All triangles of a task are updated in one matrix/vector operation.
        common = np.vstack(task.work_common)
        key = (task.index, common[:, :3].tobytes(), inverse[:3, :3].tobytes(), float(width))
        cached = self.ground_orientations.get(key)
        if cached is None:
            normals = common[:, :3] @ inverse[:3, :3]
            cube = .5*width*normals[:, 2]-.5*width*np.abs(normals).sum(axis=1)
            cached = normals, cube
            self.ground_orientations[key] = cached
            self.statistics['ground_orientation_preparations'] += 1
        normals, cube = cached
        planes = np.empty_like(common)
        planes[:, :3] = normals
        planes[:, 3] = common[:, :3] @ inverse[:3, 3] + common[:, 3]
        offsets = planes[:, 3] + cube
        self.statistics['ground_translation_updates'] += 1
        return tuple((planes[s], offsets[s]) for s in self.slices[task.index])
