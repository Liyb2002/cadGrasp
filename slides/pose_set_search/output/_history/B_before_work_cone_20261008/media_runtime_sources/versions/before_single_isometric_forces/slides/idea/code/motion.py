"""Save all rigid transforms used by the two videos, with path diagnostics."""
from pathlib import Path
import json
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA.parents[1]
sys.path.insert(0, str(ROOT / "slides/Co-optimize/helper_func"))
from co_common import I, S, material_volume

FPS = 24
DURATION = 16.
TRAVEL = .24


def smooth(value):
    u = np.clip(value, 0., 1.)
    return float(u * u * (3. - 2. * u))


def interpolate(a, b, u, center):
    R = Slerp([0., 1.], Rotation.from_matrix(np.array([a[:3, :3], b[:3, :3]])))([u]).as_matrix()[0]
    ca, cb = a[:3, :3] @ center + a[:3, 3], b[:3, :3] @ center + b[:3, 3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = ca * (1. - u) + cb * u - R @ center
    return T


def apply(vertices, T):
    return vertices @ T[:3, :3].T + T[:3, 3]


def world_mesh(mesh, T):
    other = mesh.copy()
    other.apply_transform(T)
    return other


def frame(mode, t, params, obj, support):
    native = np.asarray(params["world_transforms"])
    seated = np.asarray(params["seated_transforms"])
    relative = np.asarray(params["following_common_exit_fixture"])
    pose = 0 if t < 9.25 else 1
    if t < .75:
        lift, stage = TRAVEL, "pose1_before_insertion"
    elif t < 2.75:
        lift, stage = TRAVEL * (1. - smooth((t - .75) / 2.)), "pose1_insert"
    elif t < 4.25:
        lift, stage = 0., "pose1_seated"
    elif t < 6.25:
        lift, stage = TRAVEL * smooth((t - 4.25) / 2.), "pose1_exit"
    elif t < 9.25:
        lift, stage = TRAVEL, "change_pose"
    elif t < 9.5:
        lift, stage = TRAVEL, "pose2_before_insertion"
    elif t < 11.5:
        lift, stage = TRAVEL * (1. - smooth((t - 9.5) / 2.)), "pose2_insert"
    elif t < 13.:
        lift, stage = 0., "pose2_seated"
    elif t < 15.:
        lift, stage = TRAVEL * smooth((t - 13.) / 2.), "pose2_exit"
    else:
        lift, stage = TRAVEL, "pose2_after_exit"

    fixture_T = native[pose].copy() if mode == "pose_following_wrap" else np.eye(4)
    object_T = native[pose].copy() if mode == "pose_following_wrap" else seated[pose].copy()
    object_visible = True
    exit_vector = native[pose, :3, :3] @ relative if mode == "pose_following_wrap" else np.array([0., 0., 1.])
    object_T[:3, 3] += lift * exit_vector
    if 6.25 <= t < 9.25:
        u = smooth((t - 6.25) / 3.)
        if mode == "pose_following_wrap":
            object_visible = False
            fixture_T = interpolate(native[0], native[1], u, support.center_mass)
            # Raise the EMPTY fixture, rotate, then set it back down. All
            # displayed placements remain above the floor throughout the turn.
            lift_fraction = (smooth((t - 6.25) / .5) if t < 6.75 else
                             1. if t <= 8.75 else 1. - smooth((t - 8.75) / .5))
            minimum = apply(support.vertices, fixture_T)[:, 2].min()
            fixture_T[2, 3] += .025 * lift_fraction - minimum
            stage = "empty_fixture_reorientation"
        else:
            object_T = interpolate(seated[0], seated[1], u, obj.center_mass)
            object_T[2, 3] += TRAVEL
            safe_z = support.bounds[1, 2] + .035
            minimum = apply(obj.vertices, object_T)[:, 2].min()
            object_T[2, 3] += max(0., safe_z - minimum)
            stage = "object_reorientation_above_fixed_fixture"
    return dict(time_seconds=t, stage=stage, pose=pose + 1,
                support_transform=fixture_T.tolist(), object_transform=object_T.tolist(),
                object_visible=object_visible, lift_m=lift)


def main():
    params = json.loads((IDEA / "data/geometry.json").read_text())
    obj = trimesh.load(IDEA / "data/object_local.obj", force="mesh", process=False)
    summaries = {}
    for mode in ("pose_following_wrap", "fixed_seat_reuse"):
        support = trimesh.load(IDEA / f"data/{mode}.obj", force="mesh", process=False)
        frames = [frame(mode, n / FPS, params, obj, support) for n in range(int(FPS * DURATION))]
        min_floor = min(apply(support.vertices, np.asarray(row["support_transform"]))[:, 2].min() for row in frames)
        if min_floor < -1e-9:
            raise RuntimeError("Fixture crosses floor during display")
        if mode == "fixed_seat_reuse":
            if not all(np.array_equal(row["support_transform"], np.eye(4)) for row in frames):
                raise RuntimeError("The fixed fixture moved")
        checks = []
        # Geometric replay of displayed paths complements the complete swept
        # solids already checked in prepare.py. No contact/force solver runs.
        selected = sorted(set(range(0, len(frames), 12)) | {int(t * FPS) for t in (2.75, 4.25, 6.25, 9.25, 11.5, 13., 15.)})
        for n in selected:
            row = frames[n]
            if not row["object_visible"]:
                continue
            fixture = S.solid(world_mesh(support, np.asarray(row["support_transform"])))
            body = S.solid(world_mesh(obj, np.asarray(row["object_transform"])))
            overlap = material_volume(fixture ^ body)
            checks.append(dict(frame=n, overlap_m3=overlap))
            if overlap > 1e-10:
                raise RuntimeError(f"Displayed object penetrates fixture: {mode}, frame {n}, {overlap}")
        motion = dict(mode=mode, fps=FPS, duration_seconds=DURATION,
                      frame_count=len(frames), travel_m=TRAVEL, frames=frames)
        (IDEA / f"data/{mode}_motion.json").write_text(json.dumps(motion, indent=2) + "\n")
        summaries[mode] = dict(minimum_fixture_z_m=float(min_floor),
                               displayed_path_checks=checks,
                               fixed_support=mode == "fixed_seat_reuse")
    (IDEA / "data/motion_checks.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("MOTION COMPLETE", {k: max(row["overlap_m3"] for row in v["displayed_path_checks"]) for k, v in summaries.items()}, flush=True)


if __name__ == "__main__":
    main()
