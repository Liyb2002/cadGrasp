# Grasp-modality tasks (cadGrasp)

One folder per **grasp modality**. Each is a fixed gripper type that *fails* on a
naive shape but succeeds once the object carries the right **synthesized
affordance** — the cadGrasp thesis. The four modalities are the passive-grasp
strategies from the Amazon "Passive Grippers" work (insertion, tongs, wrench,
spanning).

Each task is a scripted **kinematic** demonstration on an SO-101 arm (MuJoCo): IK
the gripper through waypoints, carry the grasped object kinematically, render an
MP4. Contact dynamics are a later fidelity step.

| Folder | Gripper | Object affordance | Demo |
|--------|---------|-------------------|------|
| `insertion/` | SO-101 jaw holds a boss part | boss → socket | `output/insertion/…` |
| `tongs/` | parallel jaw (native) | a thin **pinch tab** on a round puck | `output/tongs/demo.mp4` |
| `wrench/` | fixed **hook** | an overhanging **lip** (mushroom) | `output/wrench/demo.mp4` |
| `spanning/` | fixed **C-cage** | a narrow **neck** between two disks | `output/spanning/demo.mp4` |

The objects live in `../objects/<category>/` (one demo object per category).
Rendered videos go to `../output/<category>/` (gitignored).

## Shared harness — `common.py`

- `build_scene(object_bodies, tool_geoms, extra_assets)` — SO-101 + floor + your
  object(s) + an optional fixed tool welded onto the gripper. Returns the model
  and handy joint/site indices.
- `ik_solve(model, data, info, target)` — position-only damped-least-squares IK on
  the gripper site.
- `interpolate_and_render(...)` — run the arm through IK waypoints, pose the
  carried object each frame, render.
- `set_free_body`, `free_qpos_adr`, `default_camera`, `overlay`, `READY`.

`insertion/` predates `common.py` and uses its own `scene.py`/`shape_gen.py`
(procedural boss/socket generation); `tongs`/`wrench`/`spanning` each have a
single `demo.py` built on `common.py`.

## Run

```bash
conda run -n smolvla python tasks/tongs/demo.py
conda run -n smolvla python tasks/wrench/demo.py
conda run -n smolvla python tasks/spanning/demo.py
conda run -n smolvla python tasks/insertion/expert.py --sample 0
```

## Not here

SmolVLA (vision-language-action) exploration tools — the model runner, task
runner, video renderer, setup check — live in `../smolvla_examples/`. They're
orthogonal to the grasp-task suite (they explore *learned* policies; these tasks
are scripted experts / affordance demos).

## Next

The demos are geometry + kinematics. The natural upgrades: a quasi-static
grasp-stability check (contact points + force/torque balance under gravity, per
the Amazon "Grasp Configuration"), then the real thesis — **one synthesized shape
graspable by all four fixed grippers at once**.
