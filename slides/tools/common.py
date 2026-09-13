"""Shared paths, constants and small helpers for the object-library pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OBJ_DIR = REPO / "objects"

# The Passive Grippers (SIGGRAPH'22) test suite.
# https://github.com/milmillin/passive-gripper/tree/main/data/stl
#
# D5, D6 and D7 (apple, peach, orange) are excluded: they are near-spheres with
# no discrete stable placements -- 40 drops gave 36, 16 and 24 distinct resting
# directions -- so they have neither a well-defined T0 nor an edge to tip about.
PG_URL = "https://raw.githubusercontent.com/milmillin/passive-gripper/main/data/stl/{name}.stl"
PG_OBJECTS = [
    "A1-f", "A1-s", "A2", "A3", "A4", "A5",
    "B",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "D1", "D2", "D3", "D4", "D8",
]

DENSITY = 1000.0  # kg/m^3, uniform


def obj_path(name: str) -> Path:
    return OBJ_DIR / name


def read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo-order (w,x,y,z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo-order (w,x,y,z) quaternion."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def objects_with_meshes() -> list[str]:
    return [n for n in PG_OBJECTS if (obj_path(n) / "mesh.stl").exists()]


def figure_path(filename: str) -> str:
    """Keep optional tool figures inside slides, independent of the working directory."""
    directory = Path(__file__).resolve().parent / "figures"
    directory.mkdir(exist_ok=True)
    return str(directory / filename)
