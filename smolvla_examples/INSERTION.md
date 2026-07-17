# Peg-in-hole insertion (cadGrasp)

A real, working insertion task in MuJoCo. Each of **10 samples is a matched
boss/socket pair** built the "cut cylinder" way, at randomized shape, size,
colour, position and yaw. The SO-101 arm grasps the boss part and inserts its
boss into the socket part. Videos → `output/insertion/sample_00.mp4 … 09.mp4`.

**How the shapes are made (`shape_gen.py`).** Cut a solid across the middle → two
halves with flat mating faces. Put a **boss** (male stub) on one half's face and a
matching **socket** (female recess) on the other; sharing shape/size, they fit by
construction — that's the joint. **Body and joint shapes are independent:**

- **body**: cylinder | cuboid | triangle/pentagon/hexagon/octagon prism — size varies a lot
- **joint**: cylinder (round peg) | cuboid (square peg) — size varies a lot

So a sample can be a cuboid body with a round joint, a hexagonal body with a
square joint, etc. Each part is a real solid built by **CSG** (trimesh +
manifold), so the socket is a genuine recessed hole:

```
  boss part  = body  UNION  boss-stub   (arm carries it, boss points DOWN)
  socket part = body  MINUS  hole        (sits on the table)
```

emitted as an inline MuJoCo `vertex+face` mesh. Boss and socket share joint
shape/size (+0.15 mm clearance) and yaw, so the fit is exact; when seated the
boss part's flat face meets the socket's top face. Each spec is a deterministic
function of `(seed, i)`; viewable per-sample MJCF is written to `objects/samples/`
(gitignored).

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
- `../objects/samples/sample_XX.xml` — the 10 generated boss+socket pairs
  (viewable standalone; gitignored, written by `shape_gen.py`).
- `shape_gen.py` — parametric boss/socket generator (surface-of-revolution
  meshes; shapes, sizes, placement).
- `insertion_scene.py` — composes SO-101 + a sample's boss + socket parts into one
  model (`build_model_spec`); `--preview` renders a static frame.
- `insert_expert.py` — the scripted IK expert + video rendering (per-sample
  grasp/insert heights derived from each shape's dimensions).
- `vla_debug.py` — dump SmolVLA's exact input/output for one observation
  (`output/insertion/debug/panel.png` + `io.json`).

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

## See the VLA's input & output

`vla_debug.py` renders the 3 camera views SmolVLA consumes for one scene
observation, feeds them (+ the 6-D joint state + instruction) to `smolvla_base`,
and writes `output/insertion/debug/`:

- `panel.png` — inputs (3 views + instruction + state) on top, the predicted
  50×6 action chunk plotted below, all in one image.
- `camera{1,2,3}_*.png` — the raw 256×256 RGB inputs.
- `action_chunk.png` — the 6 joint-target trajectories.
- `io.json` — instruction, state, the full 50×6 chunk, per-joint stats.

```bash
conda run -n smolvla python smolvla_examples/vla_debug.py --sample 0
```

(The action chunk is base-model, untrained — this is to *see the I/O format*, not
a competent insertion.)

Each video: ~7 s, 640×480, showing approach → grasp → lift → align → **insert**,
with the hole position (random per sample) in the overlay.

## Next step toward a real VLA insertion

1. Add contact dynamics + a real grasp to the expert (fidelity).
2. Record its rollouts as a LeRobot dataset (the 3 camera views SmolVLA expects +
   the 6-D joint state/action) across many random layouts.
3. `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...` (Colab/GPU).
4. Drop the fine-tuned policy into this scene's loop and measure insertion success
   over fresh random holes. **That** is "the VLA guiding the robot."
