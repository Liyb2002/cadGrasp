"""Parametric plug + socket generator for the insertion task.

Each sample gets its OWN matched pair: a convex-prism (or round) plug and a
socket whose hole is the same N-gon, sized with clearance so the plug drops in.
Shape, size, colour, position and yaw are all randomized per sample.

A "spec" is a plain dict describing one sample. The element emitters
(peg_elements / socket_elements) return MuJoCo <asset> and <geom> ElementTree
nodes that insertion_scene stitches onto the arm scene; write_standalone dumps a
viewable peg+socket MJCF into objects/samples/ so you can inspect the shapes.
"""

import colorsys
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SAMPLES_DIR = os.path.join(REPO, "objects", "samples")

# name -> number of sides (0 => round, approximated by a many-sided ring)
SHAPES = {"round": 0, "triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
ROUND_SEGMENTS = 24

# Reachable placement envelope for the small SO-101. The arm can't reach both
# close AND high, and every object is approached from above, so the inner radius
# is bounded (empirical reach map: high-z waypoints need r >= 0.16). Yaw is free.
REACH_MIN, REACH_MAX = 0.16, 0.20
ANGLE = math.radians(40)
MIN_SEP = 0.075


def _rand_xy(rng):
    r = rng.uniform(REACH_MIN, REACH_MAX)
    a = rng.uniform(-ANGLE, ANGLE)
    return (float(r * math.cos(a)), float(r * math.sin(a)))


def _rgba(rng, sat=0.65, val=0.9):
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), sat, val)
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def sample_specs(n, seed=0, write=True):
    """Return n sample specs. spec[i] depends only on (seed, i), NOT on n, so
    rendering --sample 3 and --n 10 give the identical sample 3."""
    # Roster of shuffled full-permutation blocks (independent of n) => stable
    # shape per index, and all 6 shapes appear before any repeats.
    broster = np.random.RandomState(seed)
    roster = []
    for _ in range(4):
        block = list(SHAPES)
        broster.shuffle(block)
        roster += block

    specs = []
    for i in range(n):
        rng = np.random.RandomState(seed * 997 + i * 7919 + 1)  # per-sample, n-independent
        shape = roster[i % len(roster)]
        r_peg = float(rng.uniform(0.007, 0.011))
        length = float(rng.uniform(0.052, 0.072))
        clearance = float(rng.uniform(0.003, 0.006))
        depth = float(rng.uniform(0.028, 0.044))
        wall_t = float(rng.uniform(0.006, 0.010))
        floor = 0.006

        hole_xy = _rand_xy(rng)
        peg_xy = _rand_xy(rng)
        while math.hypot(hole_xy[0] - peg_xy[0], hole_xy[1] - peg_xy[1]) < MIN_SEP:
            peg_xy = _rand_xy(rng)

        specs.append({
            "i": i, "shape": shape, "n": SHAPES[shape],
            "r_peg": r_peg, "length": length, "clearance": clearance,
            "depth": depth, "wall_t": wall_t, "floor": floor,
            "hole_xy": hole_xy, "peg_xy": peg_xy,
            "hole_yaw": float(rng.uniform(0, 2 * math.pi)),
            "peg_yaw": float(rng.uniform(0, 2 * math.pi)),
            "peg_rgba": _rgba(rng, sat=0.75, val=0.9),
            "socket_rgba": _rgba(rng, sat=0.35, val=0.7),
        })
    if write:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        for s in specs:
            write_standalone(s)
    return specs


def _yaw_quat(yaw):
    return f"{math.cos(yaw / 2):.5f} 0 0 {math.sin(yaw / 2):.5f}"


def _nsides(spec):
    return ROUND_SEGMENTS if spec["n"] == 0 else spec["n"]


def peg_elements(spec, mesh_name="peg_mesh"):
    """Return (list of <asset> nodes, list of <geom> nodes) for the plug, centered at body origin."""
    h = spec["length"] / 2.0
    r = spec["r_peg"]
    assets, geoms = [], []
    if spec["n"] == 0:
        geoms.append(ET.Element("geom", {
            "name": "peg", "type": "cylinder", "size": f"{r:.5f} {h:.5f}",
            "rgba": spec["peg_rgba"], "mass": "0.02", "friction": "1 0.02 0.001", "condim": "4"}))
    else:
        N = spec["n"]
        verts = []
        for j in range(N):
            a = 2 * math.pi * j / N + math.pi / N
            x, y = r * math.cos(a), r * math.sin(a)
            verts += [f"{x:.5f} {y:.5f} {-h:.5f}", f"{x:.5f} {y:.5f} {h:.5f}"]
        assets.append(ET.Element("mesh", {"name": mesh_name, "vertex": "  ".join(verts)}))
        geoms.append(ET.Element("geom", {
            "name": "peg", "type": "mesh", "mesh": mesh_name,
            "rgba": spec["peg_rgba"], "mass": "0.02", "friction": "1 0.02 0.001", "condim": "4"}))
    return assets, geoms


def socket_elements(spec):
    """Return (assets=[], geoms=[...]) for the socket: a floor plate + a ring of walls forming the hole."""
    N = _nsides(spec)
    R_in = spec["r_peg"] + spec["clearance"]          # inradius of the hole (wall-to-center)
    wall_t, depth, floor = spec["wall_t"], spec["depth"], spec["floor"]
    edge_half = R_in * math.tan(math.pi / N)
    R_out = R_in / math.cos(math.pi / N) + wall_t     # footprint reach

    geoms = [ET.Element("geom", {
        "name": "socket_floor", "type": "box",
        "size": f"{R_out:.5f} {R_out:.5f} {floor / 2:.5f}", "pos": f"0 0 {floor / 2:.5f}",
        "rgba": spec["socket_rgba"]})]
    dark = spec["socket_rgba"]
    for k in range(N):
        theta = 2 * math.pi * k / N
        nx, ny = math.cos(theta), math.sin(theta)
        cx = (R_in + wall_t / 2) * nx
        cy = (R_in + wall_t / 2) * ny
        geoms.append(ET.Element("geom", {
            "name": f"socket_wall_{k}", "type": "box",
            "size": f"{wall_t / 2:.5f} {edge_half + wall_t:.5f} {depth / 2:.5f}",
            "pos": f"{cx:.5f} {cy:.5f} {floor + depth / 2:.5f}",
            "quat": _yaw_quat(theta), "rgba": dark}))
    return [], geoms


def hole_top_z(spec):
    return spec["floor"] + spec["depth"]


def write_standalone(spec):
    """Write a viewable peg+socket MJCF (no arm) to objects/samples/sample_XX.xml."""
    m = ET.Element("mujoco", {"model": f"pair_{spec['i']:02d}_{spec['shape']}"})
    ET.SubElement(m, "option", {"gravity": "0 0 -9.81"})
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "256", "height": "1536"})
    wb = ET.SubElement(m, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 1", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "1 1 0.05", "rgba": "0.3 0.3 0.35 1"})

    p_assets, p_geoms = peg_elements(spec)
    s_assets, s_geoms = socket_elements(spec)
    for a in p_assets + s_assets:
        asset.append(a)

    socket = ET.SubElement(wb, "body", {"name": "socket", "pos": "0 0 0", "quat": _yaw_quat(spec["hole_yaw"])})
    for g in s_geoms:
        socket.append(g)
    peg = ET.SubElement(wb, "body", {"name": "peg", "pos": f"0.05 0 {spec['length'] / 2:.5f}"})
    for g in p_geoms:
        peg.append(g)

    path = os.path.join(SAMPLES_DIR, f"sample_{spec['i']:02d}.xml")
    ET.ElementTree(m).write(path)
    return path


if __name__ == "__main__":
    specs = sample_specs(10, seed=0, write=True)
    for s in specs:
        print(f"sample {s['i']:02d}  {s['shape']:9s} r_peg={s['r_peg']*1000:.1f}mm len={s['length']*1000:.0f}mm "
              f"depth={s['depth']*1000:.0f}mm  hole={tuple(round(v,3) for v in s['hole_xy'])} "
              f"peg={tuple(round(v,3) for v in s['peg_xy'])}")
    print(f"\nwrote standalone previews to {SAMPLES_DIR}/")
