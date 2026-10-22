"""File and shape helpers for the new whole-task pipeline."""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "whole_task_2/output"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def object_of(name):
    """The object of a run: B+01_kuka -> B (its data and figures stay under the run's name)."""
    return name.split("+", 1)[0]


def stage_directory(output_root, object_name, stage):
    if Path(object_name).name != object_name or object_name in (".", "..", ""):
        raise ValueError("Expected a single object directory name")
    path = Path(output_root) / object_name / "step1" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_shape(path):
    """Read only the shape. Keep STL triangle order as the face ID convention."""
    mesh = trimesh.load(path, force="mesh", process=False)
    original_triangles = mesh.triangles.copy()
    mesh.merge_vertices()
    if not np.array_equal(mesh.triangles, original_triangles):
        raise ValueError("Welding STL vertices changed the source triangles")
    if (not mesh.is_watertight or not mesh.is_winding_consistent
            or mesh.volume <= 0 or np.any(mesh.area_faces <= 0)):
        raise ValueError("Expected a closed, consistently outward-oriented shape")
    return mesh
