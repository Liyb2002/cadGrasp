# Peg-in-hole insertion (cadGrasp)

A real, working insertion task in MuJoCo. Each of **10 samples has its own
matched plug + socket shape** — round, triangle, square, pentagon, hexagon,
octagon — at randomized size, colour, position and yaw. The SO-101 arm grasps
the plug and inserts it into the matching socket.
Videos → `output/insertion/sample_00.mp4 … sample_09.mp4`.

The shapes are generated parametrically (`shape_gen.py`): the plug is a convex
prism (or cylinder) and the socket is a ring of convex walls forming the same
N-gon hole, sized with clearance so the plug drops in. Each sample's spec is a
deterministic function of `(seed, i)`, so `--sample 3` and `--n 10` agree on
sample 3. Viewable per-sample MJCF is written to `objects/samples/` (gitignored,
regenerate any time).

## ⚠️ Who is driving — read this

**The base VLA is *not* driving these videos.** `smolvla_base` is not fine-tuned
and cannot do insertion; pointing it at this scene produces random joint targets.
So the arm here is driven by a **scripted inverse-kinematics expert**, not SmolVLA.

Why bother, then? Because this is the missing prerequisite for a VLA that *can*:

```
scripted expert  ──record──▶  ~50 LeRobot demo episodes  ──fine-tune──▶  SmolVLA
      │                                                                     │
      └──────────────── same scene + same closed loop ─────────────────────┘
                         (swap the trajectory source to evaluate the policy)
```

You need a working policy to generate demonstrations before a VLA can learn the
task. The expert is that policy. Swapping it for a trained SmolVLA (to *evaluate*
the VLA on the 10 random holes) is a change of trajectory source, nothing else.

## What's real vs. idealized

- **Real:** the objects, their geometry and collision walls, the SO-101 kinematics,
  10 genuinely random reachable hole/peg layouts, and IK that hits every waypoint
  to < 1 mm.
- **Idealized (kinematic playback):** the grasp is a kinematic attach (peg carried
  vertically below the tool) rather than friction-based, and poses are set directly
  rather than stepped through contact dynamics. So it's a clean *demonstration* of
  the motion, not a contact-physics rollout. Adding actuator control + a weld-on-
  contact grasp + `mj_step` is the next fidelity bump (and what you'd want before
  recording demos for real).

## Files

- `../objects/peg.xml`, `../objects/box_with_hole.xml` — the original fixed
  cylinder + square-socket pair (still used by `insertion_scene.build_model`).
- `../objects/samples/sample_XX.xml` — the 10 generated plug+socket pairs
  (viewable standalone; gitignored, written by `shape_gen.py`).
- `shape_gen.py` — parametric plug/socket generator (shapes, sizes, placement).
- `insertion_scene.py` — composes SO-101 + a sample's plug + socket into one
  model (`build_model_spec`); `--preview` renders a static frame.
- `insert_expert.py` — the scripted IK expert + video rendering (per-sample
  grasp/insert heights derived from each shape's dimensions).

Every object stays within the arm's reachable band (r ∈ [0.16, 0.20] m, ±40°) —
the SO-101 can't reach both close and high, and each object is approached from
above.

## Run

```bash
# all 10 random layouts -> output/insertion/sample_00..09.mp4
conda run -n smolvla python smolvla_examples/insert_expert.py --n 10 --seed 0

# one layout, or a static scene preview
conda run -n smolvla python smolvla_examples/insert_expert.py --sample 3
conda run -n smolvla python smolvla_examples/insertion_scene.py --seed 0   # -> output/insertion/_preview.png
```

Each video: ~7 s, 640×480, showing approach → grasp → lift → align → **insert**,
with the hole position (random per sample) in the overlay.

## Next step toward a real VLA insertion

1. Add contact dynamics + a real grasp to the expert (fidelity).
2. Record its rollouts as a LeRobot dataset (the 3 camera views SmolVLA expects +
   the 6-D joint state/action) across many random layouts.
3. `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...` (Colab/GPU).
4. Drop the fine-tuned policy into this scene's loop and measure insertion success
   over fresh random holes. **That** is "the VLA guiding the robot."
