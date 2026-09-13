"""Stage 1a: example target poses -- the object tipped up off a stable placement.

For each stable placement, two examples are produced: the object tipped about a
straight edge of its footprint, and tipped about a single corner of it.

The rule the target pose has to obey is that the object only rotates -- it never
translates, and the ground contact it keeps is part of the contact it already
had. Rotating about a line in the ground plane sends everything on one side of
the line upwards and everything on the other side below the floor, so the line
has to touch the footprint from outside: it is a supporting line. What stays on
the ground is then whatever part of the footprint lies on that line -- a straight
segment if the footprint has a straight side there, otherwise a single point.

Nothing here is physics: these are static poses, held by a robot that is not
modelled yet.

    python slides/tools/tip_examples.py A1-f
"""
from __future__ import annotations
import coordinates as COORD

import argparse

import mujoco
from mujoco import Renderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from common import (mat_to_quat_wxyz, obj_path, objects_with_meshes, read_json,
                    se3, write_json)

COLLINEAR_DEG = 5.0    # footprint sides closer than this in direction are one side
MIN_USEFUL_DEG = 2.0   # below this the "pivot" is a curved contact, not a real one
TIP_FRACTION = 0.6     # how far towards the tipping limit the examples are posed
PIVOT_RADIUS = 8e-4    # m, the red marker drawn on the pivot

SCENE = """<mujoco model="tip">
  <compiler meshdir="." angle="radian"/>
  <visual>
    <global offwidth="1200" offheight="1200"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.005" zfar="30"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <material name="floor" rgba="0.97 0.97 0.97 1" reflectance="0"/>
    <material name="part" rgba="0.87 0.58 0.28 1" specular="0.3" shininess="0.4"/>
    <mesh name="visual" file="mesh.stl"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.5 1.4 -0.6" dir="-0.35 -1 0.42" directional="true" castshadow="true"
           diffuse="0.7 0.7 0.7"/>
    <geom name="ground" type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="2 2 0.05" material="floor"/>
    <body name="obj" pos="{pos}" quat="{quat}">
      <geom type="mesh" mesh="visual" material="part"/>
    </body>
{marker}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------- footprint ---

def merge_sides(poly: np.ndarray) -> list[tuple[int, int]]:
    """Group the footprint outline's tiny hull edges into straight sides.

    Returns (start, end) index pairs into `poly` (counter-clockwise, wrapping).
    """
    n = len(poly)
    d = poly[(np.arange(n) + 1) % n] - poly            # edge vectors
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.arctan2(np.sin(ang - np.roll(ang, 1)),
                                 np.cos(ang - np.roll(ang, 1))))

    # Break where the direction has turned by more than the tolerance *since the
    # side began*. Comparing only against the previous facet would merge a whole
    # circular arc into one "side", because each of its facets turns by very little.
    start = int(np.argmax(np.abs(turn)))               # begin at the sharpest corner
    breaks, acc = [start], 0.0
    for k in range(1, n):
        i = (start + k) % n
        acc += turn[i]
        if abs(acc) > COLLINEAR_DEG:
            breaks.append(i)
            acc = 0.0
    if len(breaks) < 2:      # a closed curve with no corners: no straight side at all
        return [(i, (i + 1) % n) for i in range(n)]
    return [(breaks[k], breaks[(k + 1) % len(breaks)]) for k in range(len(breaks))]


def outward_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outward normal of the edge a->b of a counter-clockwise polygon."""
    e = b - a
    n = np.array([e[1], -e[0]])
    return n / (np.linalg.norm(n) + 1e-15)


def edge_pivot(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longest straight side of the footprint: two endpoints in the ground plane."""
    sides = merge_sides(poly)
    lengths = [np.linalg.norm(poly[e] - poly[s]) for s, e in sides]
    s, e = sides[int(np.argmax(lengths))]
    if np.linalg.norm(poly[e] - poly[s]) < 1e-9:
        raise ValueError("footprint has no side of non-zero length")
    return poly[s], poly[e]


def point_pivot(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sharpest corner of the footprint, and a supporting line touching only it.

    A line through the corner supports the footprint as long as its normal lies
    in the wedge between the normals of the two sides meeting there. The middle
    of that wedge is taken, which keeps every other point of the footprint
    strictly off the line. The middle is found by rotating one normal by half
    the turn rather than by averaging the two, which stays well defined at a
    sliver, where the two normals are opposite and the average vanishes.
    """
    n_pts = len(poly)
    corners = [s for s, _ in merge_sides(poly)]
    m = len(corners)

    def bisector(i_prev: int, i: int, i_next: int) -> tuple[float, np.ndarray]:
        n1 = outward_normal(poly[i_prev], poly[i])
        n2 = outward_normal(poly[i], poly[i_next])
        t = float(np.arctan2(n1[0] * n2[1] - n1[1] * n2[0], n1 @ n2)) % (2 * np.pi)
        half = t / 2
        return t, np.array([n1[0] * np.cos(half) - n1[1] * np.sin(half),
                            n1[0] * np.sin(half) + n1[1] * np.cos(half)])

    # Rank corners by the turn between the merged sides, so a rounded footprint
    # does not win on the tiny turn between two of its hull facets. Take the
    # supporting normal from the raw neighbouring facets, which are exact.
    turns = [bisector(corners[i - 1], c, corners[(i + 1) % m])[0]
             for i, c in enumerate(corners)]
    k = corners[int(np.argmax(turns))]
    n = bisector((k - 1) % n_pts, k, (k + 1) % n_pts)[1]
    c = poly[k]

    reach = (poly - c) @ n
    if reach.max() > 1e-7:                             # must touch at this point only
        raise ValueError("supporting line at the chosen corner cuts the footprint")
    return c, np.array([-n[1], n[0]])                  # point, line direction


# --------------------------------------------------------------------- tip ---

def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def tip(V: np.ndarray, com: np.ndarray, T0: np.ndarray,
        q: np.ndarray, axis: np.ndarray, angle: float):
    """Rotate the placed object by `angle` about the ground line (q, axis)."""
    R = rotation_about(axis, angle)
    R_new = R @ T0[:3, :3]
    t_new = R @ (T0[:3, 3] - q) + q
    return R_new, t_new


def tip_limits(V: np.ndarray, com_w: np.ndarray, T0: np.ndarray,
               q: np.ndarray, axis: np.ndarray) -> dict:
    """How far the object can be tipped about this line, and why it stops.

    Two things end the tip: the centre of gravity arriving over the pivot line,
    after which gravity drives the fall instead of resisting it; and some other
    part of the object reaching the floor, after which the pivot has moved.
    """
    a = axis / np.linalg.norm(axis)
    r = com_w - q
    r_perp = r - (r @ a) * a                            # offset from the line
    h = float(r_perp[2])
    d = float(np.linalg.norm(COORD.floor(r_perp) * [1, 1]) if abs(r_perp[2]) < 1e-12
              else np.linalg.norm(r_perp - r_perp[2] * np.array([0, 0, 1.0])))
    balance = float(np.arctan2(d, h)) if h > 1e-9 else np.pi / 2

    # Sign: tip the way that lifts the centre of gravity.
    sign = 1.0
    probe = rotation_about(a, 1e-3) @ (com_w - q) + q
    if probe[2] < com_w[2]:
        sign = -1.0

    # March out until some other part of the object reaches the floor, then
    # bisect so the reported angle is the one where it just touches.
    def lowest(ang):
        R, t = tip(V, None, T0, q, sign * a, ang)
        return (V @ R.T + t)[:, 2].min()

    hit = balance
    lo_ang = 0.0
    for ang in np.linspace(0, balance, 120)[1:]:
        if lowest(ang) < -1e-5:
            for _ in range(40):
                mid = (lo_ang + ang) / 2
                if lowest(mid) < -1e-5:
                    ang = mid
                else:
                    lo_ang = mid
            hit = float(lo_ang)
            break
        lo_ang = ang
    limit = float(np.degrees(min(balance, hit)))
    return {"sign": sign, "axis": [float(v) for v in a], "point": [float(v) for v in q],
            "balance_deg": float(np.degrees(balance)),
            "ground_hit_deg": float(np.degrees(hit)),
            "limit_deg": limit,
            # A tip that dies immediately means the line is not really a
            # supporting line of the contact: the footprint curves away from it,
            # so there is no pivot of this kind at this placement at all.
            "pivot_exists": bool(limit > MIN_USEFUL_DEG),
            "limited_by": "balance" if balance <= hit else "another part hits the floor"}


def make_example(V: np.ndarray, com: np.ndarray, T0: np.ndarray,
                 q: np.ndarray, axis: np.ndarray, kind: str,
                 fraction: float = TIP_FRACTION) -> dict:
    lim = tip_limits(V, T0[:3, :3] @ com + T0[:3, 3], T0, q, axis)
    angle = np.radians(lim["limit_deg"]) * fraction
    R, t = tip(V, com, T0, q, lim["sign"] * np.asarray(lim["axis"]), angle)

    W = V @ R.T + t
    lifted = W[(V @ T0[:3, :3].T + T0[:3, 3])[:, 2] < 1.5e-3]   # was touching before
    return {
        "pivot": kind,
        "tip_deg": float(np.degrees(angle)),
        "T_world_mesh": [[float(v) for v in row] for row in se3(R, t)],
        "gap_max_m": float(lifted[:, 2].max()) if len(lifted) else 0.0,
        "ground_clearance_min_m": float(W[:, 2].min()),
        **lim,
    }


# ------------------------------------------------------------------ render ---

def render(name: str, entry: dict, px: int, view: str) -> Image.Image:
    d = obj_path(name)
    T = np.asarray(entry["T_world_mesh"])
    quat_wxyz = mat_to_quat_wxyz(T[:3, :3])

    a = np.asarray(entry["axis"])
    p = np.asarray(entry["point"])
    if entry["pivot"] == "none":
        marker = ""
    elif entry["pivot"] == "edge":
        half = entry["half_length"] * a
        marker = (f'    <geom type="capsule" size="{PIVOT_RADIUS}" rgba="0.85 0.15 0.15 1"'
                  f' fromto="{" ".join(f"{v:.6g}" for v in np.r_[p - half, p + half])}"/>')
    else:
        marker = (f'    <geom type="sphere" size="{PIVOT_RADIUS * 2.2}" rgba="0.85 0.15 0.15 1"'
                  f' pos="{" ".join(f"{v:.6g}" for v in p)}"/>')

    xml = SCENE.format(pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
                       quat=" ".join(f"{v:.9g}" for v in quat_wxyz),
                       marker=marker)
    tmp = d / ".tip_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    V = np.asarray(trimesh.load(d / "mesh.stl", force="mesh").vertices)
    W = V @ T[:3, :3].T + T[:3, 3]
    lo, hi = W.min(axis=0), W.max(axis=0)

    cam = mujoco.MjvCamera()
    margin = 1.7
    if view == "side":   # look along the pivot line, from just above the floor,
        cam.azimuth = float(np.degrees(np.arctan2(a[1], a[0])))   # so the wedge of
        cam.elevation = -2.0                                      # clearance under
        cam.lookat[:] = [(lo + hi)[0] / 2, (lo + hi)[1] / 2, hi[2] * 0.3]  # it shows
        margin = 1.35
    else:
        cam.azimuth, cam.elevation = 135.0, -22.0
        cam.lookat[:] = (lo + hi) / 2
    cam.distance = margin * float(np.linalg.norm(hi - lo)) / 2 / np.tan(
        np.deg2rad(model.vis.global_.fovy / 2))

    with Renderer(model, px, px) as r:
        r.update_scene(data, camera=cam)
        return Image.fromarray(r.render())


def footprint_figure(poly: np.ndarray, com_xy: np.ndarray, edge: tuple, corner: np.ndarray,
                     px: int) -> Image.Image:
    """Top view of the footprint with the two chosen pivots marked."""
    img = Image.new("RGB", (px, px), "white")
    dr = ImageDraw.Draw(img)
    pts = np.vstack([poly, com_xy, edge[0], edge[1], corner])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max((hi - lo).max(), 1e-6) * 1.25
    mid = (lo + hi) / 2

    def to_px(p):
        u = (p - mid) / span + 0.5
        return (float(u[0] * px), float((1 - u[1]) * px))

    dr.polygon([to_px(p) for p in poly], fill=(235, 238, 243), outline=(150, 155, 165))
    dr.line([to_px(edge[0]), to_px(edge[1])], fill=(200, 40, 40), width=max(3, px // 90))
    cx, cy = to_px(corner)
    r = max(4, px // 55)
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 90, 200))
    gx, gy = to_px(com_xy)
    r = max(3, px // 80)
    dr.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(20, 20, 20), width=2)
    dr.line([(gx - 2 * r, gy), (gx + 2 * r, gy)], fill=(20, 20, 20), width=1)
    dr.line([(gx, gy - 2 * r), (gx, gy + 2 * r)], fill=(20, 20, 20), width=1)
    return img


def _font(size: int):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--blocked-top", type=int, default=12,
                    help="how many self-blocked tips to put in the gallery")
    ap.add_argument("--blocked-only", action="store_true",
                    help="rebuild only the gallery, from the tips.json already written")
    args = ap.parse_args()

    names = objects_with_meshes() if args.all else args.objects
    blocked = []
    if args.blocked_only:                              # reuse what is already on disk
        for name in names:
            f = obj_path(name) / "tips" / "tips.json"
            if f.exists():
                blocked += [r for r in read_json(f)["examples"]
                            if r["limited_by"] != "balance"]
    else:
        for name in names:
            blocked += [r for r in run_object(name, args.size)
                        if r["limited_by"] != "balance"]
    if len(names) > 1:
        blocked_sheet(blocked, args.size, args.blocked_top)


def run_object(name: str, size: int) -> list[dict]:
    print(f"--- {name} ---", flush=True)
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    V = np.asarray(mesh.vertices)
    com = np.asarray(read_json(d / "meta.json")["com_mesh_frame"])
    placements = read_json(d / "poses.json")["poses"]

    out_dir = d / "tips"
    (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    rows, records = [], []

    for pl in placements:
        T0 = np.asarray(pl["T_world_mesh"])
        poly = np.asarray(pl["support_polygon"])
        com_xy = COORD.floor(pl["com_world"])

        e0, e1 = edge_pivot(poly)
        c, c_dir = point_pivot(poly)

        edge_len = float(np.linalg.norm(e1 - e0))
        ex_edge = make_example(V, com, T0, COORD.lift_floor((e0+e1)/2),
                               COORD.lift_floor(e1-e0), "edge")
        ex_edge["half_length"] = edge_len / 2
        ex_edge["edge_length_m"] = edge_len
        ex_point = make_example(V, com, T0, COORD.lift_floor(c), COORD.lift_floor(c_dir), "point")
        ex_point["half_length"] = 0.0
        for ex in (ex_edge, ex_point):
            ex["placement"] = pl["index"]
            ex["object"] = name
            ex["probability"] = pl["probability"]
        records += [ex_edge, ex_point]

        # Where the tip is cut short by the object itself, also record the pose
        # at the limit -- the moment a second part of it reaches the floor.
        for ex, q_, ax_, kind in ((ex_edge, COORD.lift_floor((e0+e1)/2), COORD.lift_floor(e1-e0), "edge"),
                                  (ex_point, COORD.lift_floor(c), COORD.lift_floor(c_dir), "point")):
            if ex["limited_by"] == "balance":
                continue
            at_limit = make_example(V, com, T0, q_, ax_, kind, fraction=1.0)
            at_limit.update({"placement": pl["index"], "object": name,
                             "half_length": ex["half_length"]})
            ex["at_limit"] = at_limit

        lying = {"pivot": "none", "axis": [1.0, 0, 0], "point": [0.0, 0, 0],
                 "T_world_mesh": pl["T_world_mesh"]}
        imgs = [render(name, lying, size, "iso"),
                footprint_figure(poly, com_xy, (e0, e1), c, size),
                render(name, ex_edge, size, "iso"),
                render(name, ex_edge, size, "side"),
                render(name, ex_point, size, "iso"),
                render(name, ex_point, size, "side")]
        for i, im in enumerate(imgs):
            im.save(out_dir / "renders" / f"p{pl['index']}_{i}.png")
        rows.append((pl, ex_edge, ex_point, imgs))
        print(f"placement {pl['index']}: edge tip {ex_edge['tip_deg']:5.1f} deg "
              f"(limit {ex_edge['limit_deg']:5.1f}, {ex_edge['limited_by']}), "
              f"gap {ex_edge['gap_max_m'] * 1000:5.1f} mm | "
              f"point tip {ex_point['tip_deg']:5.1f} deg "
              f"(limit {ex_point['limit_deg']:5.1f}, {ex_point['limited_by']}), "
              f"gap {ex_point['gap_max_m'] * 1000:5.1f} mm", flush=True)

    write_json(out_dir / "tips.json", {"object": name, "tip_fraction": TIP_FRACTION,
                                       "examples": records})

    px = size
    label_w, head_h = int(px * 0.5), int(px * 0.22)
    sheet = Image.new("RGB", (label_w + 6 * px, head_h + px * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.07)), _font(int(px * 0.058))
    for i, h in enumerate(["lying on the ground", "footprint,\nred = edge, blue = corner",
                           "tipped about the edge", "the same, seen along the edge",
                           "tipped about the corner", "the same, seen along the line"]):
        dr.text((label_w + i * px + 8, int(head_h * 0.18)), h, fill=(20, 20, 20), font=fs)
    for r, (pl, ex_e, ex_p, imgs) in enumerate(rows):
        y = head_h + r * px
        dr.text((10, y + px // 2 - int(px * 0.1)),
                f"placement {pl['index']}\np = {pl['probability']:.2f}",
                fill=(20, 20, 20), font=f)
        for c_i, im in enumerate(imgs):
            sheet.paste(im, (label_w + c_i * px, y))
        for c_i, ex in ((2, ex_e), (4, ex_p)):
            dr.text((label_w + c_i * px + 6, y + px - int(px * 0.075)),
                    f"{ex['tip_deg']:.0f} deg of {ex['limit_deg']:.0f} max"
                    f"   gap {ex['gap_max_m'] * 1000:.0f} mm",
                    fill=(60, 60, 60), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(220, 220, 220))
    sheet.save(out_dir / "sheet.png")
    print(f"{out_dir / 'sheet.png'}  {sheet.size[0]}x{sheet.size[1]}\n")
    return records


def blocked_sheet(blocked: list[dict], px: int, top: int = 12) -> None:
    """Gallery of the tips that are cut short by the object, not by balance.

    Left: the object posed part-way. Right: the same tip taken to its limit,
    where a second part of the object has come back down onto the floor and the
    pivot would move if it went any further.
    """
    from common import OBJ_DIR

    total = len(blocked)
    # worst first: how little of the tip the balance limit would have allowed
    blocked = sorted(blocked, key=lambda r: r["limit_deg"] / max(r["balance_deg"], 1e-9))[:top]
    label_w, head_h = int(px * 0.62), int(px * 0.2)
    sheet = Image.new("RGB", (label_w + 3 * px, head_h + px * len(blocked)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.07)), _font(int(px * 0.058))
    for i, h in enumerate(["tipped part-way", "taken to the limit",
                           "the limit, seen along the pivot"]):
        dr.text((label_w + i * px + 8, int(head_h * 0.25)), h, fill=(20, 20, 20), font=fs)

    for r, ex in enumerate(blocked):
        y = head_h + r * px
        lim = ex["at_limit"]
        for c_i, im in enumerate([render(ex["object"], ex, px, "iso"),
                                  render(ex["object"], lim, px, "iso"),
                                  render(ex["object"], lim, px, "side")]):
            sheet.paste(im, (label_w + c_i * px, y))
        dr.text((10, y + px // 2 - int(px * 0.17)),
                f"{ex['object']}  placement {ex['placement']}\n"
                f"{ex['pivot']} pivot\n\n"
                f"stops at {ex['limit_deg']:.0f} deg\n"
                f"balance alone would\nallow {ex['balance_deg']:.0f} deg",
                fill=(20, 20, 20), font=f)
        dr.line([(0, y), (sheet.size[0], y)], fill=(220, 220, 220))
    out = OBJ_DIR / "tips_self_blocked.png"
    sheet.save(out)
    print(f"{total} self-blocked tips, worst {len(blocked)} shown -> {out}  "
          f"{sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
