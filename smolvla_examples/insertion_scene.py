"""Compose an insertion scene: SO-101 arm + peg (object A) + box-with-hole (object B).

The object geometry lives in ../objects/*.xml (human-editable MJCF). This module
reads those bodies, places them at chosen positions, and stitches them together
with the SO-101 model (via <include>) into one MuJoCo model.

It also provides:
  - sample_layouts(n, seed): random reachable hole positions on the table
  - build_model(...): returns (model, data, info) with the gripper site + hole geometry
  - a --preview CLI that renders one static frame so you can eyeball the scene.

The composed XML is written INTO the SO-101 model dir so its relative mesh paths
(meshdir="assets") resolve.
"""

import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OBJECTS_DIR = os.path.join(REPO, "objects")
MODEL_DIR = os.path.join(REPO, "demo", "menagerie", "robotstudio_so101")
SCENE_TMP = os.path.join(MODEL_DIR, "_insertion_scene_tmp.xml")

TABLE_Z = 0.0                 # objects rest on the floor
HOLE_TOP_Z = TABLE_Z + 0.040  # box is 40 mm tall; the opening is at the top
PEG_HALF_LEN = 0.03

# Reachable annulus for downward reach (tuned to the small SO-101 workspace).
REACH_MIN, REACH_MAX = 0.15, 0.19
ANGLE_MIN, ANGLE_MAX = math.radians(-32), math.radians(32)
MIN_SEP = 0.07  # keep peg and hole from overlapping


def _object_body(filename, pos, add_free_joint=False):
    """Load a <body> from objects/<filename>, clone it, set pos, optional free joint."""
    tree = ET.parse(os.path.join(OBJECTS_DIR, filename))
    body = copy.deepcopy(tree.getroot().find(".//body"))
    body.set("pos", f"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}")
    if add_free_joint:
        fj = ET.Element("freejoint", {"name": body.get("name") + "_free"})
        body.insert(0, fj)
    return body


def build_scene_xml(box_xy, peg_pos, peg_free=True) -> str:
    """Return composed MJCF as a string."""
    m = ET.Element("mujoco", {"model": "insertion"})
    ET.SubElement(m, "include", {"file": "so101.xml"})
    ET.SubElement(m, "option", {"gravity": "0 0 -9.81"})

    visual = ET.SubElement(m, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "global", {"azimuth": "160", "elevation": "-20"})

    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
                                     "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
                                     "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane",
                                      "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})

    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "size": "0 0 0.05", "pos": "0 0 0",
                               "type": "plane", "material": "groundplane"})
    wb.append(_object_body("box_with_hole.xml", (box_xy[0], box_xy[1], TABLE_Z), add_free_joint=False))
    wb.append(_object_body("peg.xml", peg_pos, add_free_joint=peg_free))

    if peg_free:
        eq = ET.SubElement(m, "equality")
        # weld created inactive; the controller activates it to "grasp" the peg.
        ET.SubElement(eq, "weld", {"name": "grasp", "body1": "peg", "body2": "gripper", "active": "false"})

    return ET.tostring(m, encoding="unicode")


def build_model(box_xy, peg_pos, peg_free=True):
    xml = build_scene_xml(box_xy, peg_pos, peg_free)
    with open(SCENE_TMP, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(SCENE_TMP)
    data = mujoco.MjData(model)
    info = {
        "site_ee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
        "hole_top": np.array([box_xy[0], box_xy[1], HOLE_TOP_Z]),
        "box_xy": np.array(box_xy),
    }
    return model, data, info


def _scene_skeleton():
    """Shared scene head: include arm, gravity, visuals, ground. Returns (root, asset, worldbody)."""
    m = ET.Element("mujoco", {"model": "insertion"})
    ET.SubElement(m, "include", {"file": "so101.xml"})
    ET.SubElement(m, "option", {"gravity": "0 0 -9.81"})
    visual = ET.SubElement(m, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "global", {"azimuth": "160", "elevation": "-20"})
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", {"type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
                                     "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
                                     "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "groundplane", "texture": "groundplane",
                                      "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"})
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "size": "0 0 0.05", "pos": "0 0 0",
                               "type": "plane", "material": "groundplane"})
    return m, asset, wb


def build_model_spec(spec, peg_free=True):
    """Build a scene from a shape_gen spec: SO-101 + generated socket + generated plug."""
    import shape_gen as G

    m, asset, wb = _scene_skeleton()
    hx, hy = spec["hole_xy"]
    px, py = spec["peg_xy"]
    peg_half = spec["length"] / 2.0

    p_assets, p_geoms = G.peg_elements(spec)
    s_assets, s_geoms = G.socket_elements(spec)
    for a in p_assets + s_assets:
        asset.append(a)

    socket = ET.SubElement(wb, "body", {"name": "box", "pos": f"{hx:.5f} {hy:.5f} 0",
                                        "quat": G._yaw_quat(spec["hole_yaw"])})
    for g in s_geoms:
        socket.append(g)

    # Plug is oriented to the socket's yaw so their polygons line up (exact fit).
    peg = ET.SubElement(wb, "body", {"name": "peg", "pos": f"{px:.5f} {py:.5f} {peg_half:.5f}",
                                     "quat": G._yaw_quat(spec["hole_yaw"])})
    if peg_free:
        ET.SubElement(peg, "freejoint", {"name": "peg_free"})
    for g in p_geoms:
        peg.append(g)

    with open(SCENE_TMP, "w") as f:
        f.write(ET.tostring(m, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(SCENE_TMP)
    data = mujoco.MjData(model)
    info = {
        "site_ee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
        "hole_top": np.array([hx, hy, G.hole_top_z(spec)]),
        "box_xy": np.array([hx, hy]),
        "peg_half": peg_half,
        "floor": spec["floor"],
    }
    return model, data, info


def _rand_xy(rng):
    r = rng.uniform(REACH_MIN, REACH_MAX)
    a = rng.uniform(ANGLE_MIN, ANGLE_MAX)
    return (float(r * math.cos(a)), float(r * math.sin(a)))


def sample_layouts(n, seed=0):
    """Return n (hole_xy, peg_xy) pairs, both in the reachable band and separated."""
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        hole = _rand_xy(rng)
        peg = _rand_xy(rng)
        if math.hypot(hole[0] - peg[0], hole[1] - peg[1]) >= MIN_SEP:
            out.append((hole, peg))
    return out


def preview_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.13, 0.0, 0.06]
    cam.distance = 0.72
    cam.azimuth = 150.0
    cam.elevation = -23.0
    return cam


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "output", "insertion", "_preview.png"))
    args = ap.parse_args()

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    layouts = sample_layouts(args.n, args.seed)
    box_xy, peg_xy = layouts[0]
    peg_pos = (peg_xy[0], peg_xy[1], TABLE_Z + PEG_HALF_LEN)  # peg standing at sampled spot
    model, data, info = build_model(box_xy, peg_pos, peg_free=False)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=480, width=640)
    r.update_scene(data, preview_camera())
    imageio.imwrite(args.out, r.render())
    r.close()
    print(f"box_xy={box_xy}, hole_top={info['hole_top']}")
    print(f"gripper site id={info['site_ee']}, site world pos={data.site_xpos[info['site_ee']].round(3)}")
    print(f"preview -> {args.out}")


if __name__ == "__main__":
    _main()
