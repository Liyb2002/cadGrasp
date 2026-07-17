"""SPANNING (cage-around-a-neck) grasp-modality demo for cadGrasp.

Story: the object is a spool -- two wide 20 mm disks flanking a narrow 8 mm neck.
It's too wide to pinch and heavy at the rims, so the SO-101's parallel jaw fails.
The cadGrasp affordance is the WAIST: we ride a FIXED C-cage on the wrist, lower it
beside the neck, slide the C's opening around the 8 mm neck, and lift. The two disks
block axial escape -- a geometric (caging) grasp, not a friction grasp.

Like the insertion demo (smolvla_examples/insert_expert.py) this is a KINEMATIC
scripted demonstration: IK the gripper through waypoints (all IK < 1 mm), and once
the cage has captured the neck the spool is carried by making its free-joint pose
follow the gripper site. The C-cage is a visual-only tool (contype/conaffinity 0);
the kinematic attach does the lifting, but it looks like the cage wraps the neck.

Run from repo root:
    python tasks/spanning/demo.py
Output: output/spanning/demo.mp4
"""

import os
import sys

sys.path.insert(0, "tasks")

import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np

import common

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "output", "spanning", "demo.mp4")

# Object placement (spool stands on the floor within the SO-101 reach envelope).
OX, OY = 0.17, 0.0
NECK_Z = 0.028          # neck centre height (object-frame z of the 8 mm waist)

# --- gripper-frame geometry (see so101.xml) ---
# The 'gripperframe' site (the IK target) sits at this gripper-body-local pose,
# ~0.098 m below the body origin at the jaw tip. We build the cage centred on it,
# so when IK puts the site at the neck centre, the cage encircles the neck.
SITE_LOCAL = (0.012, -0.000218, -0.098)


def build_cage_tool():
    """A FIXED C-cage: a thin horizontal ring of capsule chords with a ~110 deg
    opening, inner radius just larger than the 8 mm neck. The opening faces the
    gripper's local -y so the neck can slide in laterally. Plus two short vertical
    stems tying the ring up to the jaw tip so it reads as one attached tool."""
    cx, cy, cz = SITE_LOCAL
    R = 0.013            # ring mean radius (inner ~0.0105 > neck r 0.008)
    wall = 0.0025        # capsule radius -> thin wall
    gap_deg = 110.0      # opening angle
    gap_center = 270.0   # opening faces local -y (the lateral approach side)
    n_seg = 12
    blue = "0.55 0.75 0.95 1"
    geoms = []
    a_lo = np.radians(gap_center + gap_deg / 2.0)
    a_hi = np.radians(gap_center + 360.0 - gap_deg / 2.0)
    angs = np.linspace(a_lo, a_hi, n_seg + 1)
    for i in range(n_seg):
        p0 = (cx + R * np.cos(angs[i]),     cy + R * np.sin(angs[i]),     cz)
        p1 = (cx + R * np.cos(angs[i + 1]), cy + R * np.sin(angs[i + 1]), cz)
        geoms.append(ET.Element("geom", {
            "type": "capsule",
            "fromto": f"{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f} {p1[0]:.5f} {p1[1]:.5f} {p1[2]:.5f}",
            "size": f"{wall}", "rgba": blue,
            "contype": "0", "conaffinity": "0",
        }))
    # two stems from the ring (back, near the closed side) up toward the jaw tip
    for ang in (np.radians(gap_center + 180 - 35), np.radians(gap_center + 180 + 35)):
        sx, sy = cx + R * np.cos(ang), cy + R * np.sin(ang)
        geoms.append(ET.Element("geom", {
            "type": "capsule",
            "fromto": f"{sx:.5f} {sy:.5f} {cz:.5f} {sx:.5f} {sy:.5f} {cz + 0.028:.5f}",
            "size": "0.0022", "rgba": blue,
            "contype": "0", "conaffinity": "0",
        }))
    return geoms


def main():
    # 1-2. object + free joint, placed standing on the floor
    bodies = common.load_object_bodies("spanning/spool.xml")
    obj = bodies[0]
    obj.insert(0, ET.Element("freejoint", {"name": "obj_free"}))
    obj.set("pos", f"{OX} {OY} 0")

    # 3. scene with the fixed C-cage tool on the gripper
    model, data, info = common.build_scene(object_bodies=[obj], tool_geoms=build_cage_tool())
    oq = common.free_qpos_adr(model, "obj_free")

    # 5. ready pose; spool standing
    data.qpos[info["qpos"]] = common.READY
    data.qpos[info["grip_q"]] = 0.3
    common.set_free_body(data, oq, [OX, OY, 0])
    mujoco.mj_forward(model, data)

    # 6. waypoints: approach above+beside -> lower beside neck -> slide C around
    #    neck -> lift. Grip stays open (0.3): the jaw can't pinch it; the cage does.
    LIFT = "lift: caged by the neck"
    waypoints = [
        ([OX, OY + 0.05,  0.14],    "approach: cage above the neck", 0.3),
        ([OX, OY + 0.045, NECK_Z],  "lower cage beside the neck",    0.3),
        ([OX, OY,         NECK_Z],  "slide C-cage around the neck",  0.3),
        ([OX, OY,         0.115],   LIFT,                            0.3),
    ]

    # 7. carry offset: once caged, object origin = site + offset keeps the neck at
    #    the site. Engage centres the site at [OX,OY,NECK_Z]; object origin is
    #    NECK_Z below that.
    carry_offset = np.array([OX, OY, 0.0]) - np.array([OX, OY, NECK_Z])
    sid = info["site_ee"]

    def on_frame(data, label):
        if label == LIFT:
            pos = data.site_xpos[sid] + carry_offset
            common.set_free_body(data, oq, pos)
        else:
            common.set_free_body(data, oq, [OX, OY, 0])

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = common.default_camera(lookat=(OX, OY, 0.065), distance=0.42,
                                azimuth=125, elevation=-14)
    title = "cadGrasp | SPANNING: C-cage around the neck (SO-101)"
    frames, errors = common.interpolate_and_render(
        model, data, info, renderer, cam, waypoints, on_frame, title,
        seg_frames=28, hold=30, fps=30)
    renderer.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=30, codec="libx264", quality=8)

    maxerr = max(e for _, e in errors)
    for label, e in errors:
        print(f"  IK {e*1000:6.2f} mm  {label}")
    print(f"max IK pos err = {maxerr*1000:.2f} mm  ->  {OUT}")


if __name__ == "__main__":
    main()
