"""Stage 0c: sample stable placements T0 by dropping each object on the ground.

Each object is dropped from a uniformly random orientation and simulated until it
comes to rest.  Which pose an object rests in is decided entirely by which face
touches the ground, i.e. by the direction of gravity expressed in the mesh frame;
yaw about the world z axis is free.  Settled trials are therefore clustered on
that direction, and the most frequent distinct clusters are kept, with the
cluster frequency reported as the probability of that placement.

Writes objects/<name>/poses.json and objects/<name>/scene.xml.

    python slides/tools/drop_sample.py [--trials 40] [--keep 5] [--jobs 6]
"""
from __future__ import annotations
import coordinates as COORD

import argparse
import os
import zlib
from multiprocessing import Pool

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from common import (OBJ_DIR, mat_to_quat_wxyz, obj_path, objects_with_meshes,
                    quat_wxyz_to_mat, read_json, se3, write_json)
from scene import build_xml, write_scene

DROP_CLEARANCE = 0.02      # m above the ground the object starts from
MAX_SIM_TIME = 30.0        # s before a trial is abandoned as never settling
# Rest is judged on how much the pose actually moved over a window, not on
# instantaneous velocity: rounded objects creep for a long time at velocities
# that never trip a velocity threshold, while many-contact objects chatter
# numerically at velocities that never fall below one.
REST_WINDOW = 1.0          # s
REST_SAMPLE = 0.05         # s between pose samples
REST_DPOS = 5e-4           # m of travel allowed within the window
REST_DANG = 0.5            # deg of rotation allowed within the window
CLUSTER_DEG = 8.0          # two placements are the same if gravity agrees to within this
CONTACT_EPS = 1.5e-3       # m; vertices this close to the ground are support points


def compile_scene(name: str):
    """Compile the scene from the object directory so mesh paths resolve."""
    d = obj_path(name)
    tmp = d / f".scene_tmp_{os.getpid()}.xml"
    tmp.write_text(build_xml(name, drop_z=0.3))
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def random_rotations(n: int, seed: int) -> np.ndarray:
    """n uniformly random rotation matrices (normalised Gaussian quaternions)."""
    q = np.random.default_rng(seed).normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Rotation.from_quat(q).as_matrix()


def place_above_ground(V: np.ndarray, R: np.ndarray, clearance: float,
                       com_xy: np.ndarray | None = None) -> np.ndarray:
    """Translation putting the rotated mesh `clearance` above z=0, centred in xy."""
    W = V @ R.T
    t = np.zeros(3)
    t[2] = clearance - W[:, 2].min()
    if com_xy is not None:
        t[:2] = -com_xy
    return t


def angle_between(qa: np.ndarray, qb: np.ndarray) -> float:
    """Rotation angle in degrees between two (w,x,y,z) quaternions."""
    dR = quat_wxyz_to_mat(qb) @ quat_wxyz_to_mat(qa).T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def run(model, data, pos, quat, seconds: float):
    """Simulate from a given pose for a fixed time; return the final pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = pos
    data.qpos[3:7] = quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data.qpos[:3].copy(), data.qpos[3:7].copy()


def simulate_until_rest(model, data, pos, quat) -> tuple[np.ndarray, np.ndarray, bool]:
    """Drop from the given pose and return the pose it comes to rest in."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = pos
    data.qpos[3:7] = quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    every = max(1, int(REST_SAMPLE / dt))
    win = int(REST_WINDOW / REST_SAMPLE)
    hist: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(int(MAX_SIM_TIME / dt)):
        mujoco.mj_step(model, data)
        if i % every:
            continue
        hist.append((data.qpos[:3].copy(), data.qpos[3:7].copy()))
        if len(hist) > win:
            (p0, q0), (p1, q1) = hist[-win - 1], hist[-1]
            if np.linalg.norm(p1 - p0) < REST_DPOS and angle_between(q0, q1) < REST_DANG:
                return p1, q1, True
    return data.qpos[:3].copy(), data.qpos[3:7].copy(), False


def gravity_in_mesh_frame(quat_wxyz: np.ndarray) -> np.ndarray:
    """Unit vector pointing along gravity, expressed in the object's own frame.

    Identifies the resting face and is invariant to the free yaw, so it is the
    right key to cluster placements on.
    """
    R = quat_wxyz_to_mat(quat_wxyz)
    return R.T @ np.array([0.0, 0.0, -1.0])


def cluster_directions(dirs: np.ndarray, deg: float):
    """Greedy clustering of unit vectors by angle, largest cluster first."""
    cos_tol = np.cos(np.deg2rad(deg))
    clusters: list[list[int]] = []
    centers: list[np.ndarray] = []
    for i, v in enumerate(dirs):
        for k, c in enumerate(centers):
            if float(v @ c) >= cos_tol:
                clusters[k].append(i)
                m = dirs[clusters[k]].mean(axis=0)
                centers[k] = m / np.linalg.norm(m)
                break
        else:
            clusters.append([i])
            centers.append(v.copy())
    order = np.argsort([-len(c) for c in clusters])
    return [clusters[i] for i in order], [centers[i] for i in order]


def canonical_pose(V: np.ndarray, com: np.ndarray, g_mesh: np.ndarray):
    """Yaw-free pose that lays the object on the face `g_mesh` points at."""
    R = Rotation.align_vectors([[0.0, 0.0, -1.0]], [g_mesh])[0].as_matrix()
    com_w_xy = COORD.floor(R @ com)
    t = place_above_ground(V, R, clearance=0.0, com_xy=com_w_xy)
    return R, t


def support_geometry(V: np.ndarray, com: np.ndarray, R: np.ndarray, t: np.ndarray) -> dict:
    """Support polygon, CoM height, and the tipping distances d for each hull edge.

    d/h over the hull edges is the tipping ratio of section 5 (C1): the pivot edge
    of a tip is one of these edges, and mg*d/h is the force needed to start it.
    """
    W = V @ R.T + t
    com_w = R @ com + t
    z0 = W[:, 2].min()
    sup = COORD.floor(W[W[:, 2] <= z0 + CONTACT_EPS])

    out = {
        "com_world": [float(v) for v in com_w],
        "h_com_m": float(com_w[2] - z0),
        "n_support_points": int(len(sup)),
    }
    try:
        hull = ConvexHull(sup)
    except (QhullError, ValueError):
        out.update({"support_polygon": [[float(a) for a in p] for p in sup],
                    "support_area_m2": 0.0, "degenerate_support": True,
                    "d_min_m": 0.0, "d_max_m": 0.0, "d_over_h_min": 0.0,
                    "com_inside_support": False})
        return out

    poly = sup[hull.vertices]
    p = COORD.floor(com_w)
    d = []
    inside = True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        n = np.array([e[1], -e[0]])           # outward normal (hull is CCW)
        n /= np.linalg.norm(n) + 1e-15
        signed = float(n @ (p - a))
        d.append(abs(signed))
        inside &= signed <= 1e-6
    d = np.asarray(d)
    out.update({
        "support_polygon": [[float(a) for a in q] for q in poly],
        "support_area_m2": float(hull.volume),
        "degenerate_support": False,
        "com_inside_support": bool(inside),
        # per-edge tipping distance; the edge order matches support_polygon
        "d_edges_m": [float(v) for v in d],
        "d_min_m": float(d.min()),
        "d_max_m": float(d.max()),
        "d_over_h_min": float(d.min() / max(com_w[2] - z0, 1e-9)),
    })
    return out


def sample_object(args) -> dict:
    name, n_trials, keep, seed = args
    d = obj_path(name)
    meta = read_json(d / "meta.json")
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    V = np.asarray(mesh.vertices)
    com = np.asarray(meta["com_mesh_frame"])

    model = compile_scene(name)
    data = mujoco.MjData(model)
    rots = random_rotations(n_trials, seed)

    dirs, n_settled = [], 0
    for k in range(n_trials):
        R0 = rots[k]
        t0 = place_above_ground(V, R0, DROP_CLEARANCE, com_xy=COORD.floor(R0 @ com))
        pos, quat, ok = simulate_until_rest(model, data, t0, mat_to_quat_wxyz(R0))
        if ok:
            n_settled += 1
            dirs.append(gravity_in_mesh_frame(quat))
    if not dirs:
        return {"object": name, "error": "no trial settled"}

    dirs = np.asarray(dirs)
    clusters, centers = cluster_directions(dirs, CLUSTER_DEG)

    poses, qposes = [], []
    for c, g in zip(clusters[:keep], centers[:keep]):
        # The cluster mean is a few tenths of a degree off the true equilibrium
        # for curved contacts, so set the object down once and let it relax onto
        # the equilibrium before canonicalising.
        R, t = canonical_pose(V, com, g)
        _, quat_relaxed = run(model, data, t + [0, 0, 1e-4], mat_to_quat_wxyz(R), seconds=2.0)
        g = gravity_in_mesh_frame(quat_relaxed)
        R, t = canonical_pose(V, com, g)
        quat = mat_to_quat_wxyz(R)

        # A placement we keep must hold still when simply set down at it.
        pos2, quat2 = run(model, data, t + [0, 0, 1e-4], quat, seconds=2.0)
        drift_deg = angle_between(quat, quat2)
        drift_xy = float(np.linalg.norm(COORD.floor(pos2 - t)))

        entry = {
            "index": len(poses),
            "count": len(c),
            "probability": len(c) / len(dirs),
            "quat_wxyz": [float(v) for v in quat],
            "pos": [float(v) for v in t],
            "T_world_mesh": [[float(v) for v in row] for row in se3(R, t)],
            "gravity_in_mesh_frame": [float(v) for v in g],
            "verified_stable": bool(drift_deg < 3.0 and drift_xy < 3e-3),
            "reverify_drift_deg": drift_deg,
            "reverify_drift_xy_m": drift_xy,
        }
        entry.update(support_geometry(V, com, R, t))
        poses.append(entry)
        qposes.append(np.concatenate([t, quat]))

    write_scene(name)

    out = {
        "object": name,
        "mass_kg": meta["mass_kg"],
        "n_trials": n_trials,
        "n_settled": n_settled,
        "n_distinct_placements": len(clusters),
        "n_kept": len(poses),
        "coverage": float(sum(len(c) for c in clusters[:keep]) / len(dirs)),
        "cluster_tolerance_deg": CLUSTER_DEG,
        "seed": int(seed),
        "poses": poses,
    }
    write_json(d / "poses.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or objects_with_meshes()
    # Seed off the name, not the list position, so adding or removing an object
    # does not resample every other object.
    jobs = [(n, args.trials, args.keep, args.seed ^ zlib.crc32(n.encode())) for n in names]
    results = []
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(sample_object, jobs):
            results.append(r)
            if "error" in r:
                print(f"{r['object']:6s} FAILED: {r['error']}", flush=True)
                continue
            probs = ", ".join(f"{p['probability']:.2f}" for p in r["poses"])
            bad = sum(not p["verified_stable"] for p in r["poses"])
            print(f"{r['object']:6s} settled {r['n_settled']:3d}/{r['n_trials']:3d}  "
                  f"{r['n_distinct_placements']:2d} distinct, kept {r['n_kept']} "
                  f"(cover {r['coverage']:.2f})  p=[{probs}]"
                  + (f"  !! {bad} unverified" if bad else ""), flush=True)

    # Fold the placement summary back into the library index.
    index = read_json(OBJ_DIR / "index.json")
    by_name = {r["object"]: r for r in results if "error" not in r}
    for entry in index["objects"]:
        r = by_name.get(entry["name"])
        if r is None:
            continue
        entry.update({
            "n_settled": r["n_settled"],
            "n_trials": r["n_trials"],
            "n_distinct_placements": r["n_distinct_placements"],
            "n_kept": r["n_kept"],
            "coverage": r["coverage"],
            "d_over_h_min": min(p["d_over_h_min"] for p in r["poses"]),
            "d_over_h_max": max(p["d_over_h_min"] for p in r["poses"]),
        })
    write_json(OBJ_DIR / "index.json", index)


if __name__ == "__main__":
    main()
