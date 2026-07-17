"""Matched boss/socket insertion parts, built the "cut cylinder" way.

Take a cylinder and cut it across the middle: you get two halves with flat mating
faces. Put a BOSS (a male stub) on one half's face and a matching SOCKET (a female
recess) on the other half's face. Because the boss and socket share the same
radius/shape, the two halves fit together by construction — that's the joint.

Each part is a surface of revolution swept from a 2D (radius, z) profile, so the
socket is a real recessed hole in a solid part (not a ring of walls). Revolving
with `n` angular segments gives the cross-section: round = a 32-gon (reads as
round), or a true N-gon for triangle/square/... variety across samples.

  boss part (robot carries it, boss points DOWN):     socket part (sits on table):
      ┌───────────┐  body top                             ┌───────────┐  top face
      │           │  height Hb                            │   ╷   ╷   │
      └────┐ ┌────┘  body bottom (mates here)             │   │hole│  depth d
           │ │  boss  length d                            │   └───┘   │
           └─┘                                            └───────────┘  height Hs

A "spec" is a dict for one sample; peg_elements/socket_elements emit the MuJoCo
<mesh>+<geom> nodes (inline vertex+face). Deterministic per (seed, i).
"""

import colorsys
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAMPLES_DIR = os.path.join(REPO, "objects", "insertion", "samples")

# name -> cross-section sides (0 => round, swept as a 32-gon)
SHAPES = {"round": 0, "triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
REVOLVE_ROUND = 32

# The socket hole is the SAME polygon as the boss, this much larger in radius:
# a snug exact-shape fit (also avoids coplanar z-fighting). The boss part is
# inserted at the socket's yaw so their cross-sections line up.
FIT_CLEARANCE = 0.00015
GRASP_INSET = 0.008  # how far below the body top the gripper grips

# Reachable placement envelope for the small SO-101 (r in [0.16,0.20], +/-40deg).
REACH_MIN, REACH_MAX = 0.16, 0.20
ANGLE = math.radians(40)
MIN_SEP = 0.075


def _rand_xy(rng):
    r = rng.uniform(REACH_MIN, REACH_MAX)
    a = rng.uniform(-ANGLE, ANGLE)
    return (float(r * math.cos(a)), float(r * math.sin(a)))


def _rgba(rng, sat, val):
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), sat, val)
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def _nsides(spec):
    return REVOLVE_ROUND if spec["n"] == 0 else spec["n"]


def revolve(profile, n):
    """Sweep a 2D (radius, z) profile around the z-axis into a closed mesh.

    The profile must start and end on the axis (radius 0) so the poles close the
    solid. Winding is set so face normals point outward. Returns (verts, faces).
    """
    verts, faces = [], []
    P = len(profile)
    for (r, z) in profile:
        for k in range(n):
            a = 2 * math.pi * k / n + math.pi / n
            verts.append((r * math.cos(a), r * math.sin(a), z))
    idx = lambda i, k: i * n + (k % n)
    for i in range(P - 1):
        for k in range(n):
            a, b, c, d = idx(i, k), idx(i + 1, k), idx(i + 1, k + 1), idx(i, k + 1)
            faces += [(a, c, b), (a, d, c)]  # outward-facing winding
    return verts, faces


def _mesh_el(name, verts, faces):
    vs = "  ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    fs = "  ".join(f"{a} {b} {c}" for a, b, c in faces)
    return ET.Element("mesh", {"name": name, "vertex": vs, "face": fs})


def peg_elements(spec, mesh_name="peg_mesh"):
    """Boss part (the moved half): body of height Hb with a boss of length d
    protruding DOWN. Mesh origin at the BODY CENTER."""
    n, R, Hb, rb, d = _nsides(spec), spec["R"], spec["Hb"], spec["rb"], spec["d"]
    profile = [(0, -Hb / 2 - d), (rb, -Hb / 2 - d), (rb, -Hb / 2),
               (R, -Hb / 2), (R, Hb / 2), (0, Hb / 2)]
    verts, faces = revolve(profile, n)
    assets = [_mesh_el(mesh_name, verts, faces)]
    geoms = [ET.Element("geom", {"name": "peg", "type": "mesh", "mesh": mesh_name,
                                 "rgba": spec["peg_rgba"], "mass": "0.05",
                                 "contype": "0", "conaffinity": "0"})]
    return assets, geoms


def socket_elements(spec, mesh_name="socket_mesh"):
    """Socket part (stays on table): body of height Hs with a blind hole (radius
    rb + clearance, depth d) in the TOP face. Mesh origin at the BOTTOM CENTER."""
    n, R, Hs, rb, d = _nsides(spec), spec["R"], spec["Hs"], spec["rb"], spec["d"]
    rh = rb + spec["clearance"]
    profile = [(0, 0), (R, 0), (R, Hs), (rh, Hs), (rh, Hs - d), (0, Hs - d)]
    verts, faces = revolve(profile, n)
    assets = [_mesh_el(mesh_name, verts, faces)]
    geoms = [ET.Element("geom", {"name": "socket", "type": "mesh", "mesh": mesh_name,
                                 "rgba": spec["socket_rgba"], "contype": "0", "conaffinity": "0"})]
    return assets, geoms


def sample_specs(n, seed=0, write=True):
    """Return n specs; spec[i] depends only on (seed, i), not on n."""
    broster = np.random.RandomState(seed)
    roster = []
    for _ in range(4):
        block = list(SHAPES)
        broster.shuffle(block)
        roster += block

    specs = []
    for i in range(n):
        rng = np.random.RandomState(seed * 997 + i * 7919 + 1)
        shape = roster[i % len(roster)]
        R = float(rng.uniform(0.013, 0.018))
        rb = float(R * rng.uniform(0.40, 0.52))          # boss radius (< R)
        Hb = float(rng.uniform(0.030, 0.040))            # boss-part body height
        Hs = float(rng.uniform(0.030, 0.040))            # socket-part height
        d = float(min(rng.uniform(0.011, 0.016), Hb / 2 - 0.003, Hs - 0.010))  # mating depth

        hole_xy = _rand_xy(rng)
        peg_xy = _rand_xy(rng)
        while math.hypot(hole_xy[0] - peg_xy[0], hole_xy[1] - peg_xy[1]) < MIN_SEP:
            peg_xy = _rand_xy(rng)

        specs.append({
            "i": i, "shape": shape, "n": SHAPES[shape],
            "R": R, "rb": rb, "Hb": Hb, "Hs": Hs, "d": d, "clearance": FIT_CLEARANCE,
            "hole_xy": hole_xy, "peg_xy": peg_xy,
            "hole_yaw": float(rng.uniform(0, 2 * math.pi)),
            "peg_rgba": _rgba(rng, 0.75, 0.9),
            "socket_rgba": _rgba(rng, 0.30, 0.65),
        })
    if write:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        for s in specs:
            write_standalone(s)
    return specs


def _yaw_quat(yaw):
    return f"{math.cos(yaw / 2):.5f} 0 0 {math.sin(yaw / 2):.5f}"


def write_standalone(spec):
    """Write a viewable MJCF (socket part + boss part beside it) to objects/samples/."""
    m = ET.Element("mujoco", {"model": f"pair_{spec['i']:02d}_{spec['shape']}"})
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
    # boss part standing (boss down): body center at z = d + Hb/2
    peg = ET.SubElement(wb, "body", {"name": "peg", "pos": f"0.05 0 {spec['d'] + spec['Hb'] / 2:.5f}",
                                     "quat": _yaw_quat(spec["hole_yaw"])})
    for g in pg:
        peg.append(g)

    path = os.path.join(SAMPLES_DIR, f"sample_{spec['i']:02d}.xml")
    ET.ElementTree(m).write(path)
    return path


if __name__ == "__main__":
    specs = sample_specs(10, seed=0, write=True)
    for s in specs:
        print(f"sample {s['i']:02d}  {s['shape']:9s} R={s['R']*1000:.1f} rb={s['rb']*1000:.1f} "
              f"Hb={s['Hb']*1000:.0f} Hs={s['Hs']*1000:.0f} d={s['d']*1000:.1f}mm  "
              f"hole={tuple(round(v,3) for v in s['hole_xy'])} peg={tuple(round(v,3) for v in s['peg_xy'])}")
    print(f"\nwrote standalone previews to {SAMPLES_DIR}/")
