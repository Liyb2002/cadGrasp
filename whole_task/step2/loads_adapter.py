"""Array/forward interface for future Step3/4; no legacy fixed-pose paths."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT
from .validate_loads import check_binding
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


def load_task(pose_id, output_root=DEFAULT_OUTPUT, object_name="B"):
    folder = Path(output_root)/object_name/"step2/step2.2"
    report = json.loads((folder/"loads.json").read_text())
    check_binding(report, folder)
    row = next((row for row in report["poses"] if row["id"] == pose_id), None)
    if row is None:
        raise ValueError(f"Unknown computed pose: {pose_id}")
    folder = folder/row["folder"]
    domain = WrenchDomain(json.loads((folder/"needs.json").read_text()))
    with np.load(folder/"samples.npz") as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    return TaskLoads(pose_id, domain, arrays["targets_scaled"], arrays["wrench_scale"],
        arrays["targets_7d"], arrays["source_face_id"], arrays["parameters"], arrays["work_face_index"],
        arrays["geometrically_screened"], float(arrays["weight_per_sample"]))
