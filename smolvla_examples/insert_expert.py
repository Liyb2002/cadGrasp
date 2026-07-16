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
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import insertion_scene as S

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PEG_HALF = S.PEG_HALF_LEN
GRASP_SITE_Z = 0.052           # gripper-site height when grasping the standing peg
PEG_BELOW_SITE = GRASP_SITE_Z - PEG_HALF   # vertical offset: peg center below the site


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


def set_peg(data, peg_q, xy, center_z, grasped, site_xy=None):
    """Place the peg: standing at xy on the floor, or held vertically below the gripper."""
    if grasped:
        data.qpos[peg_q:peg_q + 3] = [site_xy[0], site_xy[1], center_z]
    else:
        data.qpos[peg_q:peg_q + 3] = [xy[0], xy[1], center_z]
    data.qpos[peg_q + 3:peg_q + 7] = [1, 0, 0, 0]  # upright


def overlay(frame, title, sub):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255))
    d.text((8, 19), sub, fill=(160, 235, 170))
    return np.asarray(img)


READY = np.array([0.0, -0.6, 1.2, 0.8, 0.0])   # good downward-reach seed
INSERT_SITE_Z = S.HOLE_TOP_Z + 0.018            # tool height that seats the peg on the hole floor


def run_sample(hole_xy, peg_xy, out_path, width=640, height=480, fps=30, seg_frames=26):
    peg_start = (peg_xy[0], peg_xy[1], S.TABLE_Z + PEG_HALF)
    model, data, info = S.build_model(hole_xy, peg_start, peg_free=True)
    dof, qpos_i, grip_q, peg_q = joint_addrs(model)
    lo, hi = joint_limits(model)
    sid = info["site_ee"]

    # ready pose: arm folded down, gripper open
    data.qpos[qpos_i] = READY
    data.qpos[grip_q] = 1.2  # open
    set_peg(data, peg_q, peg_start[:2], PEG_HALF, grasped=False)
    mujoco.mj_forward(model, data)

    px, py = peg_start[0], peg_start[1]
    hx, hy = info["hole_top"][0], info["hole_top"][1]
    # (target_pos, label, grasped_after, grip_value)
    waypoints = [
        ([px, py, 0.14], "approach peg", False, 1.2),
        ([px, py, GRASP_SITE_Z], "descend to peg", False, 1.2),
        ([px, py, GRASP_SITE_Z], "grasp", True, 0.1),
        ([px, py, 0.16], "lift", True, 0.1),
        ([hx, hy, 0.16], "move over hole", True, 0.1),
        ([hx, hy, 0.10], "align above hole", True, 0.1),
        ([hx, hy, INSERT_SITE_Z], "INSERT", True, 0.1),
    ]

    frames = []
    title = f"SO-101 scripted expert  |  insert peg into hole  |  hole=({hx:.2f}, {hy:.2f})"
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
                center_z = data.site_xpos[sid][2] - PEG_BELOW_SITE
                set_peg(data, peg_q, None, center_z, grasped=True, site_xy=site_xy)
            else:
                set_peg(data, peg_q, peg_start[:2], PEG_HALF, grasped=False)
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
    layouts = S.sample_layouts(max(args.n, args.sample + 1), args.seed)

    # build one model first just to init the renderer
    hole0, peg0 = layouts[0]
    m0, _, _ = S.build_model(hole0, (peg0[0], peg0[1], S.TABLE_Z + PEG_HALF))
    renderer = mujoco.Renderer(m0, height=args.height, width=args.width)
    cam = S.preview_camera()

    idxs = range(args.n) if args.n > 1 else [args.sample]
    for i in idxs:
        hole_xy, peg_xy = layouts[i]
        out = os.path.join(out_dir, f"sample_{i:02d}.mp4")
        errs, info = run_sample(hole_xy, peg_xy, out, args.width, args.height)
        maxerr = max(e for _, e in errs)
        print(f"sample {i:02d}  hole=({info['hole_top'][0]:.3f},{info['hole_top'][1]:.3f})  "
              f"max_ik_pos_err={maxerr*1000:.1f}mm  -> {out}")
    renderer.close()


if __name__ == "__main__":
    main()
