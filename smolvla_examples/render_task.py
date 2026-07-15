"""Render SmolVLA predictions as MP4 videos of an SO-101 arm (MuJoCo).

Pipeline: SmolVLA predicts a 50-step x 6-D joint-target chunk -> we drive a
MuJoCo SO-101 arm (MuJoCo Menagerie `robotstudio_so101`, whose 6 joints map 1:1
to SmolVLA's 6 action dims) through that trajectory -> render offscreen -> encode
to output/<task>.mp4.

Run:
    conda run -n smolvla python smolvla_examples/render_task.py --task all
    conda run -n smolvla python smolvla_examples/render_task.py --task grasp
    conda run -n smolvla python smolvla_examples/render_task.py --instruction "stack the two cubes" --name stack

HONESTY: smolvla_base is NOT fine-tuned and the arm is not the exact robot the
data came from, so this is an *illustrative* playback of the predicted joint
trajectory, not a competent skill. The action values are interpreted directly
as radian joint targets, clamped to each joint's limit. It faithfully shows the
SHAPE of what the VLA predicted; it does not show a task being solved.
"""

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from run_task import TASKS
from smolvla_runner import build_frame, load_smolvla, predict_chunk, select_device

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "demo", "menagerie", "robotstudio_so101", "scene.xml")
OUT_DIR = os.path.join(REPO, "output")


def chunk_to_qpos(model: mujoco.MjModel, chunk: np.ndarray, substeps: int) -> np.ndarray:
    """Map an (n_steps, 6) action chunk to a smooth (T, nq) joint trajectory.

    Actions are treated as absolute radian joint targets (SmolVLA uses absolute,
    not delta, joint actions) and clamped to each joint's range. Consecutive
    chunk points are linearly interpolated by `substeps` for a smooth video.
    """
    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    targets = np.clip(chunk[:, : model.nq], lo, hi)  # (n_steps, nq)

    frames = [targets[0]]  # brief settle handled by caller via hold
    for a, b in zip(targets[:-1], targets[1:]):
        for s in range(1, substeps + 1):
            frames.append(a + (b - a) * (s / substeps))
    return np.asarray(frames)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.12]
    cam.distance = 0.75
    cam.azimuth = 140.0
    cam.elevation = -18.0
    return cam


def overlay(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255))
    d.text((8, 19), subtitle, fill=(180, 220, 255))
    return np.asarray(img)


def render_trajectory(model, data, renderer, cam, qpos_traj, title, hold=15, height=None):
    frames = []
    total = len(qpos_traj)
    for i in range(hold):  # hold on the first pose
        qpos_traj_i = qpos_traj[0]
        data.qpos[:] = qpos_traj_i
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(overlay(renderer.render(), title, "step 0/%d" % (total - 1)))
    for i, q in enumerate(qpos_traj):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(overlay(renderer.render(), title, "frame %d/%d" % (i, total - 1)))
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", help="grasp|poke|insert|force|all")
    ap.add_argument("--instruction", help="free-form instruction (with --name)")
    ap.add_argument("--name", help="output filename stem for --instruction")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--substeps", type=int, default=3, help="interpolation between chunk steps")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.exists(SCENE):
        raise SystemExit(f"SO-101 scene not found at {SCENE}\n"
                         f"Fetch it with:\n"
                         f"  git clone --depth 1 --filter=blob:none --sparse "
                         f"https://github.com/google-deepmind/mujoco_menagerie demo/menagerie\n"
                         f"  (cd demo/menagerie && git sparse-checkout set robotstudio_so101)")

    if args.instruction:
        jobs = [(args.name or "custom", args.instruction)]
    elif args.task == "all":
        jobs = [(k, v["instruction"]) for k, v in TASKS.items()]
    else:
        jobs = [(args.task, TASKS[args.task]["instruction"])]

    os.makedirs(OUT_DIR, exist_ok=True)
    device = select_device(args.device)
    print(f"device: {device}")
    print("loading lerobot/smolvla_base ...")
    policy, preprocess, postprocess = load_smolvla(device)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    for name, instruction in jobs:
        print(f"\n[{name}] predicting chunk for: {instruction!r}")
        frame = build_frame(policy, task=instruction)
        chunk = predict_chunk(policy, preprocess, postprocess, frame)
        qpos_traj = chunk_to_qpos(model, chunk, args.substeps)
        title = f"SmolVLA (base) | {name}: {instruction}"
        frames = render_trajectory(model, data, renderer, cam, qpos_traj, title)
        out = os.path.join(OUT_DIR, f"{name}.mp4")
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"[{name}] wrote {out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")

    renderer.close()
    print(f"\nDone. Videos in {OUT_DIR}/")


if __name__ == "__main__":
    main()
