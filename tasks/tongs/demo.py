"""TONGS modality demo: an SO-101 parallel jaw pinch-grasps a synthesized tab.

The manipulated object (objects/tongs/tab_handle.xml) is a round puck with a thin
PINCH TAB on top. The puck is deliberately un-pinchable by a parallel jaw (round,
too wide); the tab is the cadGrasp affordance -- two parallel faces 8 mm apart that
the native two-jaw gripper can bracket and lift. Story: the naive parallel jaw would
skid off the round puck; it succeeds by pinching the synthesized tab.

Like the insertion reference, this is a KINEMATIC scripted demonstration (no contact
dynamics): the arm is driven through IK waypoints on the gripper site, and the grasped
object is carried by making its free-joint pose follow the gripper. Motion is:

    approach above tab -> descend so the jaws straddle the tab -> pinch (close jaw) -> lift

Key geometry (measured from the native SO-101 jaw): the two jaws close along world X,
and the point where they meet sits ~13 mm behind the gripper site, so the site is IK'd
to tab_x + XOFF to land the pinch on the tab. The closed gripper value (~-0.06) brings
the two finger tips just outside the tab's two 8 mm-apart faces.

Run from the repo root:
    python tasks/tongs/demo.py
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "tasks")  # so `import common` works when run from the repo root

import imageio.v2 as imageio
import mujoco
import numpy as np

import common

# --- object placement -------------------------------------------------------
OX, OY = 0.17, 0.0          # tab (object body origin) on the floor, within reach
# The jaws meet ~13 mm behind the gripper site, so target the site XOFF ahead of the
# tab in x to land the pinch exactly on the tab's two faces.
XOFF = 0.013
Z_GRASP = 0.045             # site height so the finger tips bracket the tab (~mid-tab)
Z_APPROACH = 0.13           # above the tab, clear of it
Z_LIFT = 0.14               # lift height (object rises ~95 mm off the floor)
GRIP_OPEN = 1.2             # jaw wide open
GRIP_CLOSED = -0.06         # jaw closed to ~11 mm -> tips just outside the 8 mm tab

CARRY_LABEL = "lift"        # phases whose label contains this string carry the object


def main():
    # 1-2. load the object, give it a free joint, stand it on the floor.
    bodies = common.load_object_bodies("tongs/tab_handle.xml")
    obj = bodies[0]
    obj.insert(0, ET.Element("freejoint", {"name": "obj_free"}))
    obj.set("pos", f"{OX} {OY} 0")

    # 3. TONGS uses the NATIVE jaw -- no tool_geoms.
    model, data, info = common.build_scene(object_bodies=[obj])
    oq = common.free_qpos_adr(model, "obj_free")

    # 5. ready pose: arm folded down, jaw open, tab standing on the floor.
    data.qpos[info["qpos"]] = common.READY
    data.qpos[info["grip_q"]] = GRIP_OPEN
    common.set_free_body(data, oq, [OX, OY, 0.0])
    mujoco.mj_forward(model, data)

    # 6. waypoints: (target_xyz, label, grip_value). The site is targeted XOFF ahead
    #    of the tab in x so the pinch lands on it.
    tx = OX + XOFF
    waypoints = [
        ([tx, OY, Z_APPROACH], "approach above tab", GRIP_OPEN),
        ([tx, OY, Z_GRASP], "descend: jaws straddle tab", GRIP_OPEN),
        ([tx, OY, Z_GRASP], "pinch tab (close jaw)", GRIP_CLOSED),
        ([tx, OY, Z_LIFT], "lift", GRIP_CLOSED),
    ]

    # 7. carry offset = object_origin - grasp_site. At grasp the site is at the grasp
    #    target, so during carry: object_origin = site + (object_origin - grasp_target).
    grasp_target = np.array([tx, OY, Z_GRASP])
    offset = np.array([OX, OY, 0.0]) - grasp_target   # = [-XOFF, 0, -Z_GRASP]

    site_ee = info["site_ee"]

    def on_frame(data, label):
        # 8. carry the tab during lift; otherwise keep it standing on the floor.
        if CARRY_LABEL in label:
            site = data.site_xpos[site_ee]
            common.set_free_body(data, oq, site + offset, yaw=0.0)
        else:
            common.set_free_body(data, oq, [OX, OY, 0.0], yaw=0.0)

    # 9. render.
    renderer = mujoco.Renderer(model, 480, 640)
    # Look along +Y (az 270), perpendicular to the world-X jaw-closing axis, so the
    # two jaws are seen bracketing the tab and the lift-off gap under the puck is clear.
    cam = common.default_camera(lookat=(0.14, 0.0, 0.085), distance=0.40,
                                azimuth=270.0, elevation=-15.0)
    title = "cadGrasp TONGS | parallel-jaw pinch of a synthesized tab"
    frames, errors = common.interpolate_and_render(
        model, data, info, renderer, cam, waypoints, on_frame, title,
        seg_frames=28, hold=34, fps=30)
    renderer.close()

    out_dir = os.path.join("output", "tongs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo.mp4")
    imageio.mimsave(out_path, frames, fps=30, codec="libx264", quality=8)

    max_err = max(e for _, e in errors)
    print("TONGS demo waypoint IK errors:")
    for label, e in errors:
        print(f"  {label:32s} {e * 1000:6.2f} mm")
    print(f"max IK error: {max_err * 1000:.2f} mm")
    print(f"wrote {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
