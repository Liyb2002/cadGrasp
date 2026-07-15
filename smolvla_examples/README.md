# SmolVLA examples (cadGrasp)

Exploring **SmolVLA** — Hugging Face / LeRobot's compact (450M) **Vision-Language-Action**
model — on this Mac. A VLA maps *camera view(s) + robot joint state + a language
instruction* → *a chunk of future robot actions*. You change behaviour with words,
not with a new network.

These scripts run the real model end-to-end on Apple Silicon (MPS) and print the
predicted action chunk, so you can see exactly what a VLA consumes and produces.

## ⚠️ Read this first — what these examples are (and aren't)

`lerobot/smolvla_base` is a **base** model. It is **not fine-tuned** on grasping,
poking, insertion, or anything else, and there is **no robot or simulator attached
here**. So the action numbers are a *correctly-shaped, real model prediction* that
demonstrates the interface — **not a competent skill**. To get motions that actually
solve a task, you fine-tune on ~50 teleop episodes of that task (see "Next steps").

This is the honest state of open VLAs today: the *plumbing* is easy and runs on a
laptop; the *competence* comes from task data.

## What's installed

A dedicated conda env **`smolvla`** (your `cadgrasp` env was left untouched):

- `lerobot 0.4.4` (`pip install "lerobot[smolvla]"`), `torch 2.10` (MPS), `transformers 4.57`
- Backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, downloaded on first load (~1.8 GB, cached)
- For video rendering: `mujoco`, `imageio-ffmpeg`, and the SO-101 arm model from
  MuJoCo Menagerie, sparse-checked-out into `demo/menagerie/robotstudio_so101/`
  (both `demo/menagerie/` and `output/` are gitignored)

## The checkpoint's interface (`smolvla_base`)

- **inputs:** 3 camera views (`camera1/2/3`, each `3×256×256`) + a **6-D** joint state
  (SO-100/SO-101 arm: 5 joints + 1 gripper)
- **output:** a **50-step** chunk of **6-D** actions = *joint-position targets*
- The scripts read these keys from `policy.config.input_features`, so they adapt
  automatically if you swap in a differently-configured checkpoint.

## Run it

```bash
# from the repo root
conda run -n smolvla python smolvla_examples/check_setup.py            # sanity check + one forward pass
conda run -n smolvla python smolvla_examples/run_task.py --task grasp
conda run -n smolvla python smolvla_examples/run_task.py --task poke
conda run -n smolvla python smolvla_examples/run_task.py --task insert
conda run -n smolvla python smolvla_examples/run_task.py --task force
conda run -n smolvla python smolvla_examples/run_task.py --instruction "stack the two cubes"
conda run -n smolvla python smolvla_examples/run_task.py --task grasp --image /path/to/photo.jpg
```

(Or `conda activate smolvla` first, then `cd smolvla_examples && python run_task.py ...`.)

## The four tasks

| Task | Instruction | What it shows |
|------|-------------|---------------|
| **grasp** | "Grasp the red cube and lift it off the table." | Canonical VLA skill; the gripper dim (last action dim) is the most active. |
| **poke** | "Poke the blue block with the gripper tip." | Reach-and-touch; arm-approach joints dominate, gripper barely moves. |
| **insert** | "Insert the peg into the hole." | Contact-rich alignment; tighter motion. Position-only VLAs are weak here without force feedback. |
| **force** | "Push down… until contact." | **Caveat task** — see below. |

The *only* thing that differs between tasks is the instruction string. Same model,
same observation. You can confirm the instruction matters: the chunks differ per task.

## Why "force" is a caveat, not a skill

SmolVLA outputs **position** targets and takes **no** force/torque input — it cannot
do force control. To get real force behaviour you need one of:

- **(a) A controller in a torque-capable sim.** Feed SmolVLA's position targets into
  an impedance/admittance controller (e.g. in MuJoCo) that turns position error into
  joint torque with a chosen stiffness. Force lives in the *controller*, not the VLA:

  ```python
  # pseudo-code: impedance wrapper around a position-based policy
  q_des = smolvla_action          # 6-D position target from the VLA
  tau = Kp * (q_des - q) - Kd * qd   # low Kp = compliant/"soft" contact
  sim.data.ctrl[:] = tau
  ```

- **(b) Force in the observation.** Fine-tune a checkpoint whose state vector includes
  force/torque sensor readings, so the policy can react to contact.

The `--task force` preset still runs and prints the position chunk you'd feed into (a).

## Videos → `output/`

`render_task.py` turns a prediction into an MP4 of an **SO-101 arm** (MuJoCo
Menagerie `robotstudio_so101`, whose 6 joints map 1:1 to SmolVLA's 6 action dims)
playing back the predicted joint trajectory:

```bash
conda run -n smolvla python smolvla_examples/render_task.py --task all      # -> output/{grasp,poke,insert,force}.mp4
conda run -n smolvla python smolvla_examples/render_task.py --task grasp
conda run -n smolvla python smolvla_examples/render_task.py --instruction "stack the two cubes" --name stack
```

Videos land in `output/` (gitignored) as ~5 s 640×480 h264 clips with the
instruction overlaid. The action values are read straight into the joints as
radian targets, clamped to each joint's limit.

**What the video is:** an honest playback of the *shape* of the trajectory the VLA
predicted. **What it is not:** a task being solved — `smolvla_base` isn't fine-tuned
and this arm isn't the exact robot the (nonexistent) training data came from, so the
motion is illustrative. Fine-tune to make it real; this is the visualization harness.

First run needs the arm model (one-time, ~gitignored):
```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie demo/menagerie
cd demo/menagerie && git sparse-checkout set robotstudio_so101
```

## Next steps (if you want this to actually *do* something)

1. **Close the loop in MuJoCo (Tier B).** You already have MuJoCo + Menagerie in the
   `cadgrasp` env. Put an SO-101-like arm in a scene, step the sim, and feed live
   renders/state into this same inference loop. This is where grasp/poke/insert
   become *scenes* and where the impedance wrapper above gives you the "force" example
   for real. Still un-fine-tuned, so motions stay incompetent — but it's the harness.
2. **Fine-tune for a task.** Record ~50 teleop episodes (or use a LeRobot dataset),
   then `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...`.
   ~4 h on an A100; on this Mac use the LeRobot Colab notebook instead of local.
3. **Tie into cadGrasp.** A natural fit: use your synthesized CAD shapes as the
   objects in the MuJoCo scene from step 1, and use a VLA (fine-tuned per gripper) as
   one of the "genuinely-failing gripper" policies you evaluate graspability against.

## Files

- `smolvla_runner.py` — shared helpers: load policy + processors, build an observation
  frame from the checkpoint's own feature spec, run one forward pass.
- `check_setup.py` — env sanity check + one forward pass.
- `run_task.py` — the grasp / poke / insert / force examples (and free-form instructions).
- `render_task.py` — renders a prediction as an SO-101 arm MP4 into `output/`.

## References

- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- Checkpoint: https://huggingface.co/lerobot/smolvla_base
- Paper: https://huggingface.co/papers/2506.01844
