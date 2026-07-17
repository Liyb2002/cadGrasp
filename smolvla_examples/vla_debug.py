"""Show SmolVLA's exact INPUT and OUTPUT for one insertion observation.

Renders the 3 camera views the model consumes, feeds them (+ the 6-D joint state
+ the instruction) to smolvla_base, and dumps everything to
output/insertion/debug/:

    camera1_front.png / camera2_top.png / camera3_wrist.png  -- the RGB inputs (256x256)
    action_chunk.png   -- the 6 joint-target trajectories the model predicts
    panel.png          -- one image: inputs on top, output plot below
    io.json            -- instruction, state, action chunk (50x6) + per-joint stats

NOTE: smolvla_base is NOT fine-tuned, so the action chunk is a correctly-shaped
but untrained prediction. The point here is to SEE the I/O format on real data.

Run:
    conda run -n smolvla python smolvla_examples/vla_debug.py --sample 0
    conda run -n smolvla python smolvla_examples/vla_debug.py --sample 3 --phase carry
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

# the insertion task modules now live under tasks/insertion/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "insertion"))
import scene as S
import shape_gen as G
from expert import READY, joint_addrs
from smolvla_runner import build_frame, load_smolvla, predict_chunk, select_device

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PLOT_COLORS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]
IMG = 256  # camera resolution the model receives


def _free_cam(lookat, dist, az, el):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def render_views(model, data):
    """Return {camera_key: uint8 HxWx3} for the 3 views fed to the model."""
    r = mujoco.Renderer(model, height=IMG, width=IMG)
    views = {}
    r.update_scene(data, _free_cam([0.15, 0.0, 0.05], 0.42, 90, -18))   # front
    views["observation.images.camera1"] = r.render().copy()
    r.update_scene(data, _free_cam([0.15, 0.0, 0.03], 0.38, 90, -75))   # top-down
    views["observation.images.camera2"] = r.render().copy()
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if wid >= 0:
        r.update_scene(data, "wrist_cam")                               # in-hand
    else:
        r.update_scene(data, _free_cam([0.15, 0.0, 0.05], 0.25, 150, -35))
    views["observation.images.camera3"] = r.render().copy()
    r.close()
    return views


def to_chw(img_uint8):
    return torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def plot_chunk(chunk, w=780, h=300):
    img = Image.new("RGB", (w, h), (252, 252, 252))
    d = ImageDraw.Draw(img)
    ml, mr, mt, mb = 46, 150, 26, 30
    pw, ph = w - ml - mr, h - mt - mb
    n, dim = chunk.shape
    ymin, ymax = float(chunk.min()), float(chunk.max())
    pad = 0.1 * (ymax - ymin + 1e-6)
    ymin, ymax = ymin - pad, ymax + pad
    X = lambda i: ml + pw * i / (n - 1)
    Y = lambda v: mt + ph * (1 - (v - ymin) / (ymax - ymin))
    d.rectangle([ml, mt, ml + pw, mt + ph], outline=(180, 180, 180))
    if ymin < 0 < ymax:
        d.line([(ml, Y(0)), (ml + pw, Y(0))], fill=(210, 210, 210))
    for v in (ymin, 0 if ymin < 0 < ymax else (ymin + ymax) / 2, ymax):
        d.text((4, Y(v) - 6), f"{v:+.2f}", fill=(110, 110, 110))
    for j in range(dim):
        pts = [(X(i), Y(chunk[i, j])) for i in range(n)]
        d.line(pts, fill=PLOT_COLORS[j], width=2)
        ly = mt + 6 + j * 18
        d.rectangle([ml + pw + 12, ly, ml + pw + 24, ly + 10], fill=PLOT_COLORS[j])
        d.text((ml + pw + 30, ly - 1), JOINT_NAMES[j], fill=(60, 60, 60))
    d.text((ml, h - 16), f"step 0 -> {n - 1}   (output: {n}x{dim} joint-position targets)", fill=(90, 90, 90))
    return img


def make_panel(views, instruction, state, chunk):
    thumb = 224
    imgs = [Image.fromarray(views[k]).resize((thumb, thumb)) for k in
            ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]]
    labels = ["camera1  (front)", "camera2  (top-down)", "camera3  (wrist)"]
    plot = plot_chunk(chunk)
    W = max(3 * thumb + 4 * 12, plot.width + 24)
    top_h, txt_h = thumb + 46, 60
    H = top_h + txt_h + plot.height + 30
    panel = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    d.text((12, 8), "SmolVLA I/O  -  base model (untrained output), shown to see the interface",
           fill=(20, 20, 20))
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        x = 12 + i * (thumb + 12)
        panel.paste(im, (x, 30))
        d.text((x, 30 + thumb + 4), lab, fill=(40, 40, 40))
    ty = top_h + 6
    d.text((12, ty), "INPUT  instruction: " + repr(instruction), fill=(20, 20, 20))
    d.text((12, ty + 18), "INPUT  state (6-D joints): [" + ", ".join(f"{v:+.3f}" for v in state) + "]",
           fill=(20, 20, 20))
    d.text((12, ty + 36), "OUTPUT  action chunk (plotted below):", fill=(20, 20, 20))
    panel.paste(plot, (12, top_h + txt_h))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default="insert the plug into the socket")
    ap.add_argument("--phase", choices=["start", "carry"], default="start")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    spec = G.sample_specs(args.sample + 1, args.seed, write=False)[args.sample]
    model, data, info = S.build_model_spec(spec)
    dof, qpos_i, grip_q, peg_q = joint_addrs(model)

    data.qpos[qpos_i] = READY
    data.qpos[grip_q] = 1.2
    if args.phase == "carry":  # pose arm over the socket, plug held
        data.qpos[grip_q] = 0.1
    mujoco.mj_forward(model, data)

    state = np.concatenate([data.qpos[qpos_i], [data.qpos[grip_q]]]).astype(np.float32)  # 6-D
    views = render_views(model, data)

    device = select_device(args.device)
    print(f"device: {device}\nloading lerobot/smolvla_base ...")
    policy, preprocess, postprocess = load_smolvla(device)
    images = {k: to_chw(v) for k, v in views.items()}
    frame = build_frame(policy, task=args.instruction, images=images, state=state)
    chunk = predict_chunk(policy, preprocess, postprocess, frame)  # (50, 6)

    out_dir = os.path.join(S.REPO, "output", "insertion", "debug")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(views["observation.images.camera1"]).save(os.path.join(out_dir, "camera1_front.png"))
    Image.fromarray(views["observation.images.camera2"]).save(os.path.join(out_dir, "camera2_top.png"))
    Image.fromarray(views["observation.images.camera3"]).save(os.path.join(out_dir, "camera3_wrist.png"))
    plot_chunk(chunk).save(os.path.join(out_dir, "action_chunk.png"))
    make_panel(views, args.instruction, state, chunk).save(os.path.join(out_dir, "panel.png"))

    record = {
        "note": "smolvla_base is NOT fine-tuned; action chunk is correctly-shaped but untrained.",
        "model": "lerobot/smolvla_base", "device": str(device),
        "sample": args.sample, "shape": spec["shape"], "phase": args.phase,
        "input": {
            "instruction": args.instruction,
            "state_6d": {"joints": JOINT_NAMES, "values": [round(float(v), 5) for v in state]},
            "images": {k: {"shape_CHW": [3, IMG, IMG], "dtype": "float32", "range": [0, 1]} for k in views},
        },
        "output": {
            "action_chunk_shape": list(chunk.shape),
            "meaning": "50 future timesteps x 6 joint-position targets",
            "per_joint": {JOINT_NAMES[j]: {
                "first": round(float(chunk[0, j]), 5), "last": round(float(chunk[-1, j]), 5),
                "min": round(float(chunk[:, j].min()), 5), "max": round(float(chunk[:, j].max()), 5),
            } for j in range(chunk.shape[1])},
            "action_chunk": [[round(float(v), 5) for v in row] for row in chunk],
        },
    }
    with open(os.path.join(out_dir, "io.json"), "w") as f:
        json.dump(record, f, indent=2)

    print(f"wrote debug I/O to {out_dir}/")
    print("  panel.png  (inputs + output at a glance)")
    print(f"  io.json    (instruction, state, {chunk.shape[0]}x{chunk.shape[1]} action chunk + stats)")


if __name__ == "__main__":
    main()
