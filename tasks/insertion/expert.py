"""Scripted expert that ACTUALLY performs the peg-in-hole insertion.

Why this exists: smolvla_base cannot do insertion (it's not fine-tuned). To get
real insertion videos -- and, later, the demonstrations you'd fine-tune a VLA on
-- we drive the SO-101 with inverse kinematics through a fixed waypoint script:

    above peg -> descend/grasp -> lift -> above hole -> descend/insert

IK is damped-least-squares on the `gripperframe` site (position + a downward
tool-orientation term). The grasped peg is carried kinematically (held vertically
below the gripper), so this is a clean *kinematic demonstration* of the motion --
no contact dynamics. Swap this trajectory source for a trained policy to evaluate
a VLA in the same scene.

Run:
    conda run -n smolvla python smolvla_examples/insert_expert.py --sample 0
"""

import argparse
import math
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import scene as S
import shape_gen as G

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
READY = np.array([0.0, -0.6, 1.2, 0.8, 0.0])   # good downward-reach IK seed


def joint_addrs(model):
    dof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    grip_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
    peg_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_free")]
    return np.array(dof), np.array(qpos), grip_q, peg_q


def joint_limits(model):
    lo = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j), 0] for j in ARM_JOINTS])
    hi = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j), 1] for j in ARM_JOINTS])
    return lo, hi


def ik_solve(model, data, sid, target_pos, dof_idx, qpos_idx, lo, hi,
             iters=500, damping=8e-4, step=0.4):
    """Position-only damped-least-squares IK, warm-started from data.qpos. Returns pos error.

    Orientation is intentionally left free: a 5-DOF arm can't hit arbitrary
    orientations, and we carry the peg vertically below the tool anyway, so
    constraining orientation only hurt convergence.
    """
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[sid]
        if np.linalg.norm(err) < 3e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = jacp[:, dof_idx]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        data.qpos[qpos_idx] = np.clip(data.qpos[qpos_idx] + step * dq, lo, hi)
    mujoco.mj_forward(model, data)
    return np.linalg.norm(target_pos - data.site_xpos[sid])


def set_peg(data, peg_q, xy, center_z, grasped, site_xy=None, yaw=0.0):
    """Place the peg: standing at xy on the floor, or held vertically below the gripper.

    yaw (about z) is kept fixed at the socket's yaw so the plug's polygon lines up
    with the hole's polygon for an exact fit.
    """
    if grasped:
        data.qpos[peg_q:peg_q + 3] = [site_xy[0], site_xy[1], center_z]
    else:
        data.qpos[peg_q:peg_q + 3] = [xy[0], xy[1], center_z]
    data.qpos[peg_q + 3:peg_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]  # upright, yaw


def overlay(frame, title, sub):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255))
    d.text((8, 19), sub, fill=(160, 235, 170))
    return np.asarray(img)


def run_sample(spec, out_path, width=640, height=480, fps=30, seg_frames=26):
    model, data, info = S.build_model_spec(spec, peg_free=True)
    dof, qpos_i, grip_q, peg_q = joint_addrs(model)
    lo, hi = joint_limits(model)
    sid = info["site_ee"]
    stand_center = info["stand_center"]
    peg_below_site = info["peg_below_site"]
    grasp_site_z = info["grasp_z"]

    # ready pose: arm folded down, gripper open
    data.qpos[qpos_i] = READY
    data.qpos[grip_q] = 1.2  # open
    px, py = spec["peg_xy"]
    hx, hy = info["hole_top"][0], info["hole_top"][1]
    yaw = spec["hole_yaw"]
    set_peg(data, peg_q, (px, py), stand_center, grasped=False, yaw=yaw)
    mujoco.mj_forward(model, data)

    # (target_pos, label, grasped_after, grip_value)
    waypoints = [
        ([px, py, info["approach_z"]], "approach part", False, 1.2),
        ([px, py, grasp_site_z], "descend to part", False, 1.2),
        ([px, py, grasp_site_z], "grasp", True, 0.1),
        ([px, py, info["transit_z"]], "lift", True, 0.1),
        ([hx, hy, info["transit_z"]], "move over socket", True, 0.1),
        ([hx, hy, info["align_z"]], "align above socket", True, 0.1),
        ([hx, hy, info["insert_site_z"]], "INSERT boss into socket", True, 0.1),
    ]

    frames = []
    title = f"scripted expert | {spec['shape']} boss->socket | hole=({hx:.2f},{hy:.2f})"
    q_prev = data.qpos[qpos_i].copy()
    grip_prev = data.qpos[grip_q]
    grasped = False
    errors = []

    for target, label, grasp_after, grip_val in waypoints:
        # solve IK at the segment endpoint on a scratch copy (warm-started from q_prev)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        scratch.qpos[qpos_i] = q_prev
        err = ik_solve(model, scratch, sid, np.array(target, dtype=float), dof, qpos_i, lo, hi)
        q_goal = scratch.qpos[qpos_i].copy()
        errors.append((label, err))

        for f in range(1, seg_frames + 1):
            t = f / seg_frames
            data.qpos[qpos_i] = q_prev + (q_goal - q_prev) * t
            data.qpos[grip_q] = grip_prev + (grip_val - grip_prev) * t
            mujoco.mj_forward(model, data)
            site_xy = data.site_xpos[sid][:2]
            if grasped:
                center_z = data.site_xpos[sid][2] - peg_below_site
                set_peg(data, peg_q, None, center_z, grasped=True, site_xy=site_xy, yaw=yaw)
            else:
                set_peg(data, peg_q, (px, py), stand_center, grasped=False, yaw=yaw)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam)
            frames.append(overlay(renderer.render(), title, label))
        q_prev = q_goal
        grip_prev = grip_val
        if grasp_after:
            grasped = True

    # hold final frame
    for _ in range(fps):
        frames.append(frames[-1])

    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    return errors, info


# module-level renderer/cam set in main (reused across samples)
renderer = None
cam = None


def main():
    global renderer, cam
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    out_dir = os.path.join(S.REPO, "output", "insertion")
    os.makedirs(out_dir, exist_ok=True)
    specs = G.sample_specs(max(args.n, args.sample + 1), args.seed, write=True)

    # build one model first just to init the renderer
    m0, _, _ = S.build_model_spec(specs[0])
    renderer = mujoco.Renderer(m0, height=args.height, width=args.width)
    cam = S.preview_camera()

    idxs = range(args.n) if args.n > 1 else [args.sample]
    for i in idxs:
        spec = specs[i]
        out = os.path.join(out_dir, f"sample_{i:02d}.mp4")
        errs, info = run_sample(spec, out, args.width, args.height)
        maxerr = max(e for _, e in errs)
        print(f"sample {i:02d}  {spec['shape']:9s} hole=({info['hole_top'][0]:.3f},{info['hole_top'][1]:.3f})  "
              f"max_ik_pos_err={maxerr*1000:.1f}mm  -> {out}")
    renderer.close()


if __name__ == "__main__":
    main()
