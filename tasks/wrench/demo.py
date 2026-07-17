"""WRENCH modality demo: a fixed HOOK catches under a synthesized lip and lifts.

Story (cadGrasp affordance = the UNDERCUT LIP): the object is a thin stem capped by
an overhanging disk (a mushroom / T). A parallel jaw slides off the smooth round cap,
so instead the SO-101 carries a FIXED L-HOOK on its wrist. The hook descends beside
the cap, slides horizontally into the annular undercut (the open space beneath the cap
rim, radius 0.008 -> 0.020, below z=0.036), and catches under the lip. Lifting then
raises the object by a moment -- it hangs from the hook.

Like the insertion demo, this is a KINEMATIC scripted demonstration: IK the gripper
site through waypoints; once the hook is engaged under the lip, the object's free joint
is carried RIGIDLY IN THE GRIPPER FRAME (object origin = gripper_pos + R @ local_offset),
so the hook stays exactly under the lip as the arm lifts -- no contact dynamics. All IK
converges < 1 mm.

Run from repo root:
    python tasks/wrench/demo.py
"""

import os
import sys

sys.path.insert(0, "tasks")

import imageio.v2 as imageio
import mujoco
import numpy as np
import xml.etree.ElementTree as ET

import common

# ---- object placement ------------------------------------------------------
OX, OY = 0.17, 0.0          # object stands here (within the SO-101 reach envelope)
HOOK_COL = "0.72 0.72 0.78 1"   # metallic hook color, distinct from the object


def hook_geoms():
    """A FIXED L-hook appended to the gripper body (visual only, no collision).

    Geometry is in the gripper-body local frame, where the 'gripperframe' site (the
    jaw tip / IK target) sits at ~(0.012, 0, -0.098) and the jaws point down (-z).
    The hook drops a short shaft past the jaw tip, then reaches in -x (toward the
    object center at engage, approaching from the far +x side) so its upturned tip
    slips into the undercut beneath the cap lip.
    """
    def geom(name, size, pos):
        return ET.Element("geom", {
            "name": name, "type": "box", "size": size, "pos": pos,
            "rgba": HOOK_COL, "contype": "0", "conaffinity": "0"})
    return [
        # vertical shaft: descends alongside the cap, past the jaw tip
        geom("hook_shaft", "0.005 0.005 0.034", "0.030 0 -0.083"),
        # horizontal arm: reaches in under the cap rim, below the lip
        geom("hook_arm",   "0.021 0.005 0.004", "0.010 0 -0.113"),
        # upturned tip: tucks up into the undercut to catch the lip
        geom("hook_tip",   "0.005 0.005 0.006", "-0.009 0 -0.109"),
    ]


def build():
    bodies = common.load_object_bodies("wrench/lipped_post.xml")
    obj = bodies[0]
    obj.insert(0, ET.Element("freejoint", {"name": "obj_free"}))
    obj.set("pos", f"{OX} {OY} 0")
    model, data, info = common.build_scene(object_bodies=[obj], tool_geoms=hook_geoms())
    return model, data, info


def run(out_path, width=640, height=480, fps=30):
    model, data, info = build()
    oq = common.free_qpos_adr(model, "obj_free")
    sid = info["site_ee"]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    # ready pose: arm folded down, hook riding on the wrist, object standing upright
    data.qpos[info["qpos"]] = common.READY
    data.qpos[info["grip_q"]] = 0.3
    common.set_free_body(data, oq, [OX, OY, 0.0])
    mujoco.mj_forward(model, data)

    # ---- waypoints: approach beside cap -> descend -> slide hook under lip -> lift ----
    # Approach from the far (+x) side, fully within reach, so descent/lift stay vertical.
    ENGAGE = [0.198, 0.0, 0.048]     # site here => hook tip in the undercut, under the lip
    waypoints = [
        ([0.215, 0.0, 0.150], "approach beside the capped post",  0.3),
        ([0.215, 0.0, 0.050], "descend beside the lip",           0.3),
        (ENGAGE,              "slide hook UNDER the lip",          0.3),
        ([0.198, 0.0, 0.130], "LIFT: object hangs from the hook",  0.3),
        ([0.194, 0.0, 0.160], "raise clear",                      0.3),
    ]
    LIFT_LABELS = {"LIFT: object hangs from the hook", "raise clear"}

    # Pre-solve the engage pose and record the object origin IN THE GRIPPER LOCAL FRAME.
    # Carrying by this rigid offset keeps the hook exactly under the lip as the arm moves.
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[info["qpos"]] = common.READY
    common.ik_solve(model, scratch, info, ENGAGE)
    R_e = scratch.xmat[gid].reshape(3, 3)
    obj_local = R_e.T @ (np.array([OX, OY, 0.0]) - scratch.xpos[gid])

    def on_frame(d, label):
        if label in LIFT_LABELS:
            obj_world = d.xpos[gid] + d.xmat[gid].reshape(3, 3) @ obj_local
            common.set_free_body(d, oq, obj_world)      # carried rigidly by the hook
        else:
            common.set_free_body(d, oq, [OX, OY, 0.0])  # standing on the floor

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = common.default_camera(lookat=(0.17, 0.0, 0.085), distance=0.42,
                                azimuth=250.0, elevation=-10.0)
    title = "cadGrasp | WRENCH: hook catches under the synthesized lip"
    frames, errors = common.interpolate_and_render(
        model, data, info, renderer, cam, waypoints, on_frame, title,
        seg_frames=26, hold=40, fps=fps)
    renderer.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    return errors


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(repo, "output", "wrench", "demo.mp4")
    errors = run(out)
    maxerr = max(e for _, e in errors)
    for label, e in errors:
        print(f"  {label:38s} IK {e*1000:6.2f} mm")
    print(f"max IK error = {maxerr*1000:.2f} mm  ->  {out}")


if __name__ == "__main__":
    main()
