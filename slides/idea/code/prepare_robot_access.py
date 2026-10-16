"""Prepare an illustration of two opposed saved working areas.

Only saved meshes, saved placement transforms and native Panda FK are used.
The spindle is a schematic processing tool. This is a static illustration of
one-sided straight tool access, not a robot workspace or machining certificate.
"""
from pathlib import Path
import hashlib
import json
import sys

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA.parents[1]
ASSETS = ROOT / "slides/Co-optimize/idea/assets"
sys.path.insert(0, str(ASSETS))
from franka import Franka

DEST = IDEA / "data/robot_access"
LABEL = "pose1+6+11+13+14+17"
EXPERIMENT = "wrap_surface_reinforced_all52_20261015_03"
FRONT, BACK, DISPLAY_POSE = 17, 13, 11
TOOL_LENGTH = .155


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def transform(mesh, T):
    result = mesh.copy()
    result.apply_transform(T)
    return result


def save_mesh(mesh, name):
    mesh.export(DEST / (name + ".obj"))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "slides/Co-optimize/data/results/B" / LABEL / "step5" / EXPERIMENT
    report_path = folder / "data/report.json"
    report = json.loads(report_path.read_text())
    assert report["complete"] and report["passed"]
    source = ROOT / report["selected_step4_source"]
    source_report = json.loads((source / "data/report.json").read_text())
    layout = np.load(source / "layout.npz")
    i = source_report["poses"].index("pose_" + str(DISPLAY_POSE))
    fixture_transform = layout["native_world"][int(layout["hosts"][i])]
    object_transform = fixture_transform @ layout["placements"][i]
    mesh_path = ROOT / "objects/B/mesh.stl"
    body = trimesh.load(mesh_path, force="mesh", process=False)
    fixture_path = folder / "support.obj"
    assert sha(fixture_path) == report["support_sha256"]
    support = trimesh.load(fixture_path, force="mesh", process=False)
    setup_paths = {p: ROOT / f"objects/B/poses/pose_{p}/setup.npz" for p in (FRONT, BACK)}
    masks = {p: np.load(path)["work_faces"] for p, path in setup_paths.items()}

    world = transform(body, object_transform)
    ids = np.flatnonzero(masks[FRONT])
    normal = np.average(world.face_normals[ids], axis=0, weights=world.area_faces[ids])
    yaw = np.pi - np.arctan2(normal[1], normal[0])
    station = np.eye(4)
    station[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    # Raise the entire saved installation onto a simple work pedestal.
    center = (world.bounds[0] + world.bounds[1]) / 2
    station[:3, 3] = np.array([.72, .0, 0.]) - station[:3, :3] @ np.r_[center[:2], 0.]
    station[2, 3] = .60
    U, T = station @ object_transform, station @ fixture_transform
    world = transform(body, U)
    fixture = transform(support, T)
    save_mesh(world, "object")
    save_mesh(fixture, "fixture")

    patches = {}
    for p, mask in masks.items():
        ids = np.flatnonzero(mask)
        triangles = world.triangles[ids].copy()
        normals = world.face_normals[ids]
        offset_triangles = triangles + normals[:, None, :] * .00035
        patch = trimesh.Trimesh(offset_triangles.reshape(-1, 3),
                                np.arange(len(ids) * 3).reshape(-1, 3), process=False)
        save_mesh(patch, "front_patch" if p == FRONT else "back_patch")
        area = world.area_faces[ids]
        normal = np.average(normals, weights=area, axis=0)
        normal /= np.linalg.norm(normal)
        center = np.average(world.triangles_center[ids], weights=area, axis=0)
        patches[p] = dict(center=center, normal=normal, face_ids=ids,
                          area_mm2=float(area.sum() * 1e6))

    # Select a genuine front triangle near the area's center with a clear
    # normal approach, including the saved blue support as an obstruction.
    first = patches[FRONT]
    combined = trimesh.util.concatenate([world, fixture])
    candidates = sorted(first["face_ids"], key=lambda j: np.linalg.norm(world.triangles_center[j] - first["center"]))
    contact = normal = None
    for j in candidates:
        c, n = world.triangles_center[j], world.face_normals[j]
        if n @ first["normal"] < .95:
            continue
        origins = np.array([c + n * .2])
        points, _, _ = combined.ray.intersects_location(origins, np.array([-n]), multiple_hits=True)
        distance = np.linalg.norm(points - origins[0], axis=1)
        if np.all(distance >= .2 - 1e-6):
            contact, normal = c, n
            break
    if contact is None:
        raise RuntimeError("No clear front surface approach")

    robot = Franka()
    robot.model.body_pos[robot.model.body("link0").id] = [0., 0., 0.]
    # Tool +Z points inward into the target surface; the hand frame lies
    # TOOL_LENGTH outside the workpiece. Roll is selected by native IK.
    axis = -normal
    x = np.cross(np.array([0., 0., 1.]), axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    H = np.eye(4)
    H[:3, :3] = np.column_stack([x, y, axis])
    H[:3, 3] = contact - axis * TOOL_LENGTH
    solutions = []
    for roll in np.linspace(0, 2*np.pi, 16, endpoint=False):
        target = H.copy()
        target[:3, :3] = H[:3, :3] @ Rotation.from_euler("z", roll).as_matrix()
        q = robot.ik(target)
        if q is not None:
            solutions.append((np.linalg.norm(q - robot.home), q, target))
    if not solutions:
        raise RuntimeError("Native Panda cannot reach the front tool pose")
    _, joints, target = min(solutions, key=lambda row: row[0])
    actual = robot.fk(joints)
    np.testing.assert_allclose(actual, target, atol=1e-4)
    tip = actual[:3, 3] + actual[:3, :3] @ [0., 0., TOOL_LENGTH]
    assert np.linalg.norm(tip - contact) < 2e-5

    # Export actual MuJoCo visual meshes after FK, without the gripper.
    visual_model = mujoco.MjModel.from_xml_path(str(ASSETS / "franka_emika_panda/panda.xml"))
    visual_data = mujoco.MjData(visual_model)
    visual_data.qpos[:7] = joints
    mujoco.mj_forward(visual_model, visual_data)
    groups = {"white": [], "black": []}
    for g in range(visual_model.ngeom):
        body_name = visual_model.body(visual_model.geom_bodyid[g]).name
        if not body_name.startswith("link") or visual_model.geom_group[g] != 2:
            continue
        if visual_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = visual_model.geom_dataid[g]
        va, fa = visual_model.mesh_vertadr[mid], visual_model.mesh_faceadr[mid]
        v = visual_model.mesh_vert[va:va+visual_model.mesh_vertnum[mid]].copy()
        f = visual_model.mesh_face[fa:fa+visual_model.mesh_facenum[mid]].copy()
        P = np.eye(4)
        P[:3, :3] = visual_data.geom_xmat[g].reshape(3, 3)
        P[:3, 3] = visual_data.geom_xpos[g]
        m = transform(trimesh.Trimesh(v, f, process=False), P)
        rgba = visual_model.mat_rgba[visual_model.geom_matid[g]]
        groups["white" if np.mean(rgba[:3]) > .6 else "black"].append(m)
    for color, parts in groups.items():
        save_mesh(trimesh.util.concatenate(parts), "robot_" + color)

    # Geometric evidence for the depicted blocked *straight* approach.
    rear = patches[BACK]
    rear_ids = sorted(rear["face_ids"], key=lambda j: np.linalg.norm(world.triangles_center[j] - rear["center"]))
    rear_contact = world.triangles_center[rear_ids[0]]
    start = actual[:3, 3]
    direction = rear_contact - start
    length = np.linalg.norm(direction)
    direction /= length
    hits, _, _ = world.ray.intersects_location([start], [direction], multiple_hits=True)
    distances = np.linalg.norm(hits - start, axis=1)
    before = np.flatnonzero((distances > .001) & (distances < length - .003))
    if not len(before):
        raise RuntimeError("The chosen straight rear approach is not blocked")
    blocker = hits[before[np.argmin(distances[before])]]

    sources = [mesh_path, fixture_path, report_path, source / "layout.npz", source / "data/report.json",
               *setup_paths.values(), ASSETS / "franka.py", ASSETS / "franka_kinematics/panda.xml",
               ASSETS / "franka_emika_panda/panda.xml"]
    manifest = dict(
        purpose="Fixed robot at the front working area; the opposite working area is occluded by the workpiece",
        source_output=f"slides/Co-optimize/output/B/{LABEL}/step5.3/pose_{DISPLAY_POSE}.png",
        source_set=LABEL, source_experiment=EXPERIMENT, displayed_saved_pose=DISPLAY_POSE,
        front_work_area_pose=FRONT, rear_work_area_pose=BACK,
        patch_center_distance_mm=np.linalg.norm(first["center"] - rear["center"]) * 1e3,
        patch_normal_angle_deg=np.degrees(np.arccos(np.clip(first["normal"] @ rear["normal"], -1, 1))),
        patches=patches, object_world_transform=U, fixture_world_transform=T,
        station_transform=station, native_robot_scale=1., robot_joints=joints,
        tool_hand_transform=actual, tool_length_m=TOOL_LENGTH, tool_tip=tip,
        front_contact=contact, front_outward_normal=normal,
        rear_contact=rear_contact, blocked_segment_start=start, blocked_segment_hit=blocker,
        tool_tip_error_m=np.linalg.norm(tip - contact),
        straight_rear_approach_blocked=True,
        rear_target_distance_m=length, first_blocker_distance_m=np.linalg.norm(blocker - start),
        original_object_fixture_relationship_preserved=bool(np.allclose(np.linalg.inv(T) @ U, layout["placements"][i])),
        robot_global_unreachability_tested=False,
        annotation="Red dashed ray and cross denote the tested straight segment, not all Panda configurations",
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
        rendering_input_sha256={p.name: sha(p) for p in DEST.glob("*.obj")},
    )
    (DEST / "scene.json").write_text(json.dumps(serial(manifest), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: serial(manifest[k]) for k in ["source_set", "front_work_area_pose", "rear_work_area_pose",
        "patch_center_distance_mm", "patch_normal_angle_deg", "tool_tip_error_m", "straight_rear_approach_blocked"]}))


if __name__ == "__main__":
    main()
