"""Matched boss/socket insertion parts, with body and joint shapes DECOUPLED.

Cut a solid in the middle -> two halves with flat mating faces. On one face put a
BOSS (male stub); into the other cut a matching SOCKET (female recess). Because
boss and socket share shape/size (+ tiny clearance), the halves fit by
construction. Here the BODY (the big part) and the JOINT (the stub/hole) are
independent:

  body  : cylinder | cuboid | triangle/pentagon/hexagon/octagon prism   (varied size)
  joint : cylinder (round peg) | cuboid (square peg)                    (varied size)

So a sample might be a cuboid body with a round joint, or a hexagonal body with a
square joint, etc. Each part is a real solid built with CSG (trimesh + manifold):
  boss part   = body  UNION  boss-stub
  socket part = body  MINUS  hole
emitted as an inline MuJoCo vertex+face mesh. Deterministic per (seed, i).
"""

import colorsys
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SAMPLES_DIR = os.path.join(REPO, "objects", "samples")

BODY_SHAPES = ["cylinder", "cuboid", "triangle", "pentagon", "hexagon", "octagon"]
JOINT_SHAPES = ["cylinder", "cuboid"]
SECTIONS = {"cylinder": 48, "triangle": 3, "pentagon": 5, "hexagon": 6, "octagon": 8}

FIT_CLEARANCE = 0.00015   # socket is this much larger than the boss (snug exact fit)
GRASP_INSET = 0.006       # tool grips this far below the body's top

# Reachable placement envelope for the small SO-101 (r in [0.16,0.20], +/-40deg).
REACH_MIN, REACH_MAX = 0.16, 0.20
ANGLE = math.radians(40)
MIN_SEP = 0.085


def _rand_xy(rng):
    r = rng.uniform(REACH_MIN, REACH_MAX)
    a = rng.uniform(-ANGLE, ANGLE)
    return (float(r * math.cos(a)), float(r * math.sin(a)))


def _hsv(h, sat, val):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, sat, val)
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


# ---- trimesh builders -------------------------------------------------------

def _body_tm(spec, H):
    """Body solid of height H, centered at the origin."""
    if spec["body"] == "cuboid":
        return trimesh.creation.box(extents=(2 * spec["hx"], 2 * spec["hy"], H))
    return trimesh.creation.cylinder(radius=spec["R"], height=H, sections=SECTIONS[spec["body"]])


def _joint_tm(spec, height, hole):
    """Joint solid (boss or hole) of given height, centered at the origin."""
    cl = spec["clearance"] if hole else 0.0
    if spec["joint"] == "cylinder":
        return trimesh.creation.cylinder(radius=spec["jr"] + cl, height=height, sections=40)
    a = spec["jr"] + cl
    return trimesh.creation.box(extents=(2 * a, 2 * a, height))


def _mesh_el(name, m):
    vs = "  ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in m.vertices)
    fs = "  ".join(f"{int(a)} {int(b)} {int(c)}" for a, b, c in m.faces)
    return ET.Element("mesh", {"name": name, "vertex": vs, "face": fs})


def peg_elements(spec, mesh_name="peg_mesh"):
    """Boss part: body (height Hb) UNION a boss stub (length d) protruding DOWN.
    Mesh origin at the body center."""
    Hb, d = spec["Hb"], spec["d"]
    body = _body_tm(spec, Hb)
    boss = _joint_tm(spec, d + 0.002, hole=False)
    boss.apply_translation([0, 0, -Hb / 2 - d / 2 + 0.001])  # protrude d, overlap 0.002 into body
    part = trimesh.boolean.union([body, boss])
    geom = ET.Element("geom", {"name": "peg", "type": "mesh", "mesh": mesh_name,
                               "rgba": spec["peg_rgba"], "mass": "0.05",
                               "contype": "0", "conaffinity": "0"})
    return [_mesh_el(mesh_name, part)], [geom]


def socket_elements(spec, mesh_name="socket_mesh"):
    """Socket part: body (height Hs) MINUS a hole (depth d) in the top face.
    Mesh origin at the bottom center."""
    Hs, d = spec["Hs"], spec["d"]
    body = _body_tm(spec, Hs)
    body.apply_translation([0, 0, Hs / 2])          # base at z=0, top at Hs
    hole = _joint_tm(spec, 2 * d, hole=True)
    hole.apply_translation([0, 0, Hs])              # blind hole depth d from the top
    part = trimesh.boolean.difference([body, hole])
    geom = ET.Element("geom", {"name": "socket", "type": "mesh", "mesh": mesh_name,
                               "rgba": spec["socket_rgba"], "contype": "0", "conaffinity": "0"})
    return [_mesh_el(mesh_name, part)], [geom]


# ---- sampling ---------------------------------------------------------------

def _body_inradius(spec):
    if spec["body"] == "cuboid":
        return min(spec["hx"], spec["hy"])
    return spec["R"] * math.cos(math.pi / SECTIONS[spec["body"]])


def sample_specs(n, seed=0, write=True):
    """Return n specs; spec[i] depends only on (seed, i), not on n."""
    broster = np.random.RandomState(seed)
    roster = []
    for _ in range(4):
        block = list(BODY_SHAPES)
        broster.shuffle(block)
        roster += block

    specs = []
    for i in range(n):
        rng = np.random.RandomState(seed * 997 + i * 7919 + 1)
        body = roster[i % len(roster)]
        joint = JOINT_SHAPES[rng.randint(len(JOINT_SHAPES))]
        spec = {"i": i, "body": body, "joint": joint,
                "shape": f"{body}/{joint}", "clearance": FIT_CLEARANCE}

        # body size (varies a lot)
        if body == "cuboid":
            spec["hx"] = float(rng.uniform(0.013, 0.030))
            spec["hy"] = float(rng.uniform(0.013, 0.030))
        else:
            rmin = 0.013 / math.cos(math.pi / SECTIONS[body])   # keep inradius reasonable for low-n
            spec["R"] = float(rng.uniform(rmin, max(rmin + 0.002, 0.028)))
        spec["Hb"] = float(rng.uniform(0.022, 0.055))
        spec["Hs"] = float(rng.uniform(0.022, 0.055))

        # joint size, bounded to fit inside the body with a rim
        inr = _body_inradius(spec)
        if joint == "cylinder":
            jmax = min(0.013, inr - 0.004)
        else:  # square: corners reach jr*sqrt2
            jmax = min(0.010, inr / 1.42 - 0.003)
        spec["jr"] = float(rng.uniform(0.004, max(0.0045, jmax)))
        spec["d"] = float(min(rng.uniform(0.008, 0.020),
                              spec["Hb"] / 2 - 0.003, spec["Hs"] - 0.006))

        hole_xy = _rand_xy(rng)
        peg_xy = _rand_xy(rng)
        while math.hypot(hole_xy[0] - peg_xy[0], hole_xy[1] - peg_xy[1]) < MIN_SEP:
            peg_xy = _rand_xy(rng)
        hue = float(rng.uniform(0, 1))
        spec.update({
            "hole_xy": hole_xy, "peg_xy": peg_xy,
            "hole_yaw": float(rng.uniform(0, 2 * math.pi)),
            "peg_rgba": _hsv(hue, 0.78, 0.92),                 # boss part
            "socket_rgba": _hsv(hue + 0.45, 0.5, 0.62),        # socket part (contrasting hue)
        })
        specs.append(spec)

    if write:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        for s in specs:
            write_standalone(s)
    return specs


def _yaw_quat(yaw):
    return f"{math.cos(yaw / 2):.5f} 0 0 {math.sin(yaw / 2):.5f}"


def write_standalone(spec):
    """Write a viewable MJCF (socket part + boss part beside it) to objects/samples/."""
    m = ET.Element("mujoco", {"model": f"pair_{spec['i']:02d}"})
    ET.SubElement(m, "option", {"gravity": "0 0 -9.81"})
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "256", "height": "1536"})
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 0.4", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "0.3 0.3 0.05", "rgba": "0.3 0.3 0.35 1"})
    pa, pg = peg_elements(spec)
    sa, sg = socket_elements(spec)
    for a in pa + sa:
        asset.append(a)
    sock = ET.SubElement(wb, "body", {"name": "socket", "pos": "0 0 0", "quat": _yaw_quat(spec["hole_yaw"])})
    for g in sg:
        sock.append(g)
    peg = ET.SubElement(wb, "body", {"name": "peg", "pos": f"0.07 0 {spec['d'] + spec['Hb'] / 2:.5f}",
                                     "quat": _yaw_quat(spec["hole_yaw"])})
    for g in pg:
        peg.append(g)
    path = os.path.join(SAMPLES_DIR, f"sample_{spec['i']:02d}.xml")
    ET.ElementTree(m).write(path)
    return path


if __name__ == "__main__":
    for s in sample_specs(10, seed=0, write=True):
        sz = (f"hx={s['hx']*1000:.0f} hy={s['hy']*1000:.0f}" if s["body"] == "cuboid"
              else f"R={s['R']*1000:.0f}")
        print(f"sample {s['i']:02d}  body={s['body']:9s} joint={s['joint']:8s} {sz} "
              f"Hb={s['Hb']*1000:.0f} Hs={s['Hs']*1000:.0f} jr={s['jr']*1000:.1f} d={s['d']*1000:.1f}mm")
    print(f"\nwrote standalone previews to {SAMPLES_DIR}/")
