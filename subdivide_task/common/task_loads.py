"""In-memory load interface for support search; no frozen-pose file reader."""
from dataclasses import dataclass
import numpy as np
from .wrenches import WrenchDomain

@dataclass
class TaskLoads:
    pose: str
    domain: WrenchDomain
    targets: np.ndarray
    scale: np.ndarray
    targets_7d: np.ndarray
    source_face_ids: np.ndarray
    parameters: np.ndarray
    work_face_index: np.ndarray
    geometry_screened: np.ndarray
    weight_per_sample: float

    def forward_nodes(self, indices):
        raw = self.domain.forward(self.work_face_index[indices], self.parameters[indices])
        scaled = raw*self.scale
        np.testing.assert_allclose(scaled, self.targets[indices], atol=1e-13, rtol=0)
        return np.concatenate((scaled, np.zeros(scaled.shape[:-1]+(1,))), axis=-1)

    @property
    def requires_geometry_review(self):
        return not bool(self.geometry_screened.all())

