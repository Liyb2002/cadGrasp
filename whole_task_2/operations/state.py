"""Separate the working world pose from the seat in the common material frame."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Layout:
    poses: np.ndarray             # world_from_object, at the fixed working head
    material: np.ndarray          # fixture_from_world, independently per task
    exits: np.ndarray            # unit directions in the working world

    def copy(self):
        return Layout(self.poses.copy(), self.material.copy(), self.exits.copy())

    def arrays(self):
        return dict(poses=self.poses, material=self.material, exits=self.exits)

    @classmethod
    def read(cls, archive):
        return cls(*(archive[k].copy() for k in ('poses', 'material', 'exits')))


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]
