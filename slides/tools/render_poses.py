"""Stage 0d: render every sampled placement and tile them into a contact sheet.

Writes objects/<name>/renders/pose_XX.png, objects/<name>/renders/strip.png and
objects/contact_sheet.png.

    python slides/tools/render_poses.py [--size 260]
"""
from __future__ import annotations

import argparse

import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from common import OBJ_DIR, obj_path, objects_with_meshes, read_json
from scene import write_scene

AZIMUTH, ELEVATION = 135.0, -22.0
MARGIN = 1.6  # padding around the object's bounding sphere


def _font(size: int):
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def render_object(name: str, px: int) -> list[Image.Image]:
    d = obj_path(name)
    model = mujoco.MjModel.from_xml_path(str(write_scene(name)))
    data = mujoco.MjData(model)
    poses = read_json(d / "poses.json")["poses"]
    V = np.asarray(trimesh.load(d / "mesh.stl", force="mesh").vertices)

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # ground
    opt.geomgroup[2] = 1  # visual mesh; the convex collision pieces (group 3) stay hidden

    half_fov = np.deg2rad(model.vis.global_.fovy / 2)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = AZIMUTH, ELEVATION

    out_dir = d / "renders"
    out_dir.mkdir(exist_ok=True)
    imgs = []
    with YUpRenderer(model, px, px) as renderer:
        for p in poses:
            mujoco.mj_resetData(model, data)
            data.qpos[:3] = p["pos"]
            data.qpos[3:7] = p["quat_wxyz"]
            mujoco.mj_forward(model, data)

            # Frame each placement on its own world-space bounding box.
            W = V @ np.asarray(p["T_world_mesh"])[:3, :3].T + np.asarray(p["pos"])
            lo, hi = W.min(axis=0), W.max(axis=0)
            cam.lookat[:] = (lo + hi) / 2
            cam.distance = MARGIN * float(np.linalg.norm(hi - lo)) / 2 / np.tan(half_fov)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = Image.fromarray(renderer.render())
            img.save(out_dir / f"pose_{p['index']:02d}.png")
            imgs.append(img)

    strip = Image.new("RGB", (px * len(imgs), px), "white")
    for i, im in enumerate(imgs):
        strip.paste(im, (i * px, 0))
    strip.save(out_dir / "strip.png")
    return imgs


def contact_sheet(rows: list[tuple[str, list[Image.Image]]], px: int, keep: int) -> Image.Image:
    label_w = int(px * 0.42)
    W = label_w + keep * px
    H = px * len(rows)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    f_name, f_small = _font(int(px * 0.11)), _font(int(px * 0.075))

    for r, (name, imgs) in enumerate(rows):
        y = r * px
        poses = read_json(obj_path(name) / "poses.json")
        draw.text((10, y + px // 2 - int(px * 0.14)), name, fill="black", font=f_name)
        draw.text((10, y + px // 2 + int(px * 0.02)),
                  f"{poses['mass_kg'] * 1000:.0f} g\n{poses['n_distinct_placements']} distinct",
                  fill=(110, 110, 110), font=f_small)
        for c, im in enumerate(imgs):
            x = label_w + c * px
            sheet.paste(im, (x, y))
            p = poses["poses"][c]
            tag = f"p={p['probability']:.2f}  d/h={p['d_over_h_min']:.2f}"
            draw.text((x + 6, y + px - int(px * 0.1)), tag, fill=(60, 60, 60), font=f_small)
            if not p["verified_stable"]:
                draw.text((x + 6, y + 6), "unverified", fill=(200, 40, 40), font=f_small)
        draw.line([(0, y), (W, y)], fill=(220, 220, 220), width=1)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=260)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = [n for n in (args.only or objects_with_meshes())
             if (obj_path(n) / "poses.json").exists()]
    rows = []
    for n in names:
        imgs = render_object(n, args.size)
        rows.append((n, imgs))
        print(f"{n:6s} {len(imgs)} renders", flush=True)

    keep = max(len(i) for _, i in rows)
    sheet = contact_sheet(rows, args.size, keep)
    out = OBJ_DIR / "contact_sheet.png"
    sheet.save(out)
    print(f"\ncontact sheet {sheet.size[0]}x{sheet.size[1]} -> {out}")


if __name__ == "__main__":
    main()
