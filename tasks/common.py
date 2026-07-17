"""Shared harness for the grasp-modality tasks (insertion, tongs, wrench, spanning).

Every task is: an SO-101 arm (optionally carrying a fixed tool) acquires an object
that has the affordance for that grasp modality, then lifts it. The motion is a
scripted kinematic demonstration (IK to waypoints; the grasped object is carried
kinematically) — the same idealization as the insertion demo. It shows the grasp
strategy working on the synthesized affordance; contact dynamics are a later step.

This module provides the pieces each task shares:
  - build_scene(): SO-101 + floor + your object bodies + an optional fixed tool
    welded onto the gripper, returned as a ready MuJoCo model + handy indices.
  - ik_solve(): position-only damped-least-squares IK on the gripper site.
  - grasp helpers: READY pose, kinematic attach, a renderer + default camera.

The scene is written INTO the SO-101 model dir so its relative mesh paths resolve.
"""

import math
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OBJECTS_DIR = os.path.join(REPO, "objects")
MODEL_DIR = os.path.join(REPO, "demo", "menagerie", "robotstudio_so101")
SO101 = os.path.join(MODEL_DIR, "so101.xml")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
READY = np.array([0.0, -0.6, 1.2, 0.8, 0.0])   # good downward-reach IK seed
GRIPPER_SITE = "gripperframe"

# Reachable placement envelope for the small SO-101 (r in [0.16,0.20], +/-40deg).
REACH_MIN, REACH_MAX = 0.16, 0.20


def load_object_bodies(rel_path):
    """Return the list of <body> elements from an objects/<cat>/<file>.xml (the
    manipulated object), so a task can place them into its scene."""
    tree = ET.parse(os.path.join(OBJECTS_DIR, rel_path))
    return tree.getroot().findall(".//worldbody/body")


def _std_visual_and_ground(root):
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(vis, "global", {"azimuth": "160", "elevation": "-20"})
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
                                     "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
                                     "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane",
                                      "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
    wb = root.find("worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "size": "0 0 0.05", "pos": "0 0 0",
                               "type": "plane", "material": "groundplane"})


def build_scene(object_bodies=(), tool_geoms=(), extra_assets=()):
    """Compose SO-101 + floor + objects (+ an optional fixed tool on the gripper).

    - object_bodies: iterable of <body> ET elements added to the worldbody
      (the manipulated object; give a free joint if you'll move it kinematically).
    - tool_geoms: iterable of <geom>/<body> ET elements appended to the SO-101
      'gripper' body, i.e. a FIXED tool that rides rigidly with the wrist.
    - extra_assets: iterable of <asset> child elements (e.g. meshes) to register.

    Returns (model, data, info) where info has the gripper site id, arm joint
    index arrays (dof/qpos), gripper joint qpos addr, and joint limits.
    """
    root = ET.parse(SO101).getroot()
    _std_visual_and_ground(root)
    asset = root.find("asset")
    for a in extra_assets:
        asset.append(a)
    wb = root.find("worldbody")
    for b in object_bodies:
        wb.append(b)
    gripper = root.find(".//body[@name='gripper']")
    for g in tool_geoms:
        gripper.append(g)

    # Unique temp file in the model dir (so relative mesh paths resolve) — unique
    # per process so concurrent tasks don't clobber each other. Removed after load.
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="_task_scene_", dir=MODEL_DIR)
    os.write(fd, ET.tostring(root, encoding="unicode").encode())
    os.close(fd)
    try:
        model = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    data = mujoco.MjData(model)

    jid = lambda j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    info = {
        "site_ee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE),
        "dof": np.array([model.jnt_dofadr[jid(j)] for j in ARM_JOINTS]),
        "qpos": np.array([model.jnt_qposadr[jid(j)] for j in ARM_JOINTS]),
        "grip_q": model.jnt_qposadr[jid("gripper")],
        "lo": np.array([model.jnt_range[jid(j), 0] for j in ARM_JOINTS]),
        "hi": np.array([model.jnt_range[jid(j), 1] for j in ARM_JOINTS]),
    }
    return model, data, info


def ik_solve(model, data, info, target_pos, iters=500, damping=8e-4, step=0.4):
    """Position-only damped-least-squares IK for the gripper site, warm-started
    from data.qpos. Returns the final position error (metres)."""
    sid, dof, qpos_i, lo, hi = info["site_ee"], info["dof"], info["qpos"], info["lo"], info["hi"]
    target = np.asarray(target_pos, dtype=float)
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < 3e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = jacp[:, dof]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        data.qpos[qpos_i] = np.clip(data.qpos[qpos_i] + step * dq, lo, hi)
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - data.site_xpos[sid]))


def set_free_body(data, qpos_adr, pos, yaw=0.0):
    """Place a free-joint body (kinematic attach): position + yaw about z, upright."""
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def free_qpos_adr(model, joint_name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]


def default_camera(lookat=(0.15, 0.0, 0.05), distance=0.55, azimuth=150.0, elevation=-20.0):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def overlay(frame, title, sub, sub_rgb=(160, 235, 170)):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255))
    d.text((8, 19), sub, fill=sub_rgb)
    return np.asarray(img)


def interpolate_and_render(model, data, info, renderer, cam, waypoints, on_frame,
                           title, seg_frames=26, hold=30, fps=30):
    """Run the arm through IK waypoints, calling on_frame(data, label) each step to
    pose any carried object, and return the list of rendered frames.

    waypoints: list of (target_xyz, label, grip_value). IK solves each endpoint on
    a scratch copy (warm-started), then interpolates joint-space for smooth video.
    on_frame(data, label) must set any attached free-body qpos, then the frame is
    rendered. The gripper joint is animated between waypoints' grip_value.
    """
    qpos_i, grip_q = info["qpos"], info["grip_q"]
    frames = []
    q_prev = data.qpos[qpos_i].copy()
    grip_prev = data.qpos[grip_q]
    errors = []
    for target, label, grip_val in waypoints:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.qpos[qpos_i] = q_prev
        err = ik_solve(model, scratch, info, target)
        q_goal = scratch.qpos[qpos_i].copy()
        errors.append((label, err))
        for f in range(1, seg_frames + 1):
            t = f / seg_frames
            data.qpos[qpos_i] = q_prev + (q_goal - q_prev) * t
            data.qpos[grip_q] = grip_prev + (grip_val - grip_prev) * t
            mujoco.mj_forward(model, data)
            on_frame(data, label)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            frames.append(overlay(renderer.render(), title, label))
        q_prev, grip_prev = q_goal, grip_val
    frames.extend([frames[-1]] * hold)
    return frames, errors


if __name__ == "__main__":
    # smoke test: load the tongs object, IK the gripper to just above its tab, render.
    import imageio.v2 as imageio
    bodies = load_object_bodies("tongs/tab_handle.xml")
    for b in bodies:
        b.set("pos", "0.17 0.0 0.0")
    model, data, info = build_scene(object_bodies=bodies)
    data.qpos[info["qpos"]] = READY
    data.qpos[info["grip_q"]] = 1.2
    err = ik_solve(model, data, info, [0.17, 0.0, 0.10])
    print("build_scene OK | ngeom:", model.ngeom, "| IK err: %.1f mm" % (err * 1000))
    r = mujoco.Renderer(model, height=360, width=480)
    r.update_scene(data, default_camera(lookat=(0.15, 0, 0.04), distance=0.5))
    out = os.path.join(REPO, "output", "_common_smoketest.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.imwrite(out, r.render())
    r.close()
    print("wrote", out)
