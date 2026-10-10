# Historical once-loaded pilot

This auxiliary pilot is outside the current repeated-loading research line.
The fixed-grasp Step4 directory was removed at the user's request; its completed
validation records remain in `../data/experiments/step4_grasp_research_B_20261007`.
Current initialization and optimization are [Step4.1](../step4.1/README.md) and
[Step4.2](../step4.2/README.md). The Step4/Step5 labels below describe this
historical pilot, not the active pipeline.

Step4 searches non-working antipodal object contacts and Franka hand rolls,
checks the exact object against hand collision meshes, and solves endpoint IK.
The complete fingertip footprint is clipped against work triangles; checking
only the two selected contact points is insufficient.
It proposes separate loading grasps and common grasps on a 5mm support proxy.
Step5 initializes one shared fixture from the registered Step3 shell/rings,
clips all native floors, and searches one loading direction plus shared relative
translations. Required finite-depth grip patches survive both the nominal
loading sweep and its 1% clearance cut. Historical per-pose exits are not reused.

Actual candidates must pass object+closed-hand insertion, actual two-jaw fixture
contact, approach clearance, all 32,768 original loads per pose, work/floor
clearance, connectivity, coupled quasistatic retention, and sampled Franka
transport IK/collision. Search is a bounded candidate grid, not a convergence or
infeasibility proof. Station offset is `[.45,0,.15]` metres; no table/obstacle
geometry has been supplied. Each listed task pose is considered as the loading
pose after geometry screening; loading direction is not forced to world +z.

Retention assumes equal uniform density for object and fixture, friction .8 at
the jaws, frictionless object/fixture contact and quasistatic gravity. It uses
two separate force balances, not a welded assembly. Dynamic holding, measured
mass/friction, actuator-force limits, continuous swept robot collisions,
trajectory timing, and fixture strength are not certified. `accepted` remains
false even if all pilot checks pass.

```sh
.venv/bin/python slides/Co-optimize/single_load/run.py
.venv/bin/python slides/Co-optimize/single_load/test_single_load.py
```

Default B group is pose1+2+4+6. Outputs are isolated under
`data/experiments/single_load_B_20261007/pose1+2+4+6`. Older Co-optimize results
are preserved. `step4/candidates.json` records grasp transforms; `step5/trace.json`
records rejected/evaluated candidates; `report.json` is the result authority.
The earlier B result is recorded in
`data/experiments/single_load_B_20261007/pose1+2+4+6/pad_checked/report.json`:
no grasp survives the final whole-pad gate; Step5 is not entered in this run.
That failure was finite and has been superseded: expanded non-working regions
and calibrated, centered jaw contact provide a fixed stock-Franka example
passing object-grasp checks at all four poses. See
[current research record](../data/experiments/step4_grasp_research_B_20261007/README.md).

Robot assets are from Google DeepMind MuJoCo Menagerie, directory
`franka_emika_panda`, commit `0059d4335f8156206f63a35662313385f7ad6d74`.
Source: https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/franka_emika_panda
The bundled upstream LICENSE applies. Runtime additionally requires `mujoco==3.15.0`.
The local XML omits visual geoms and unused visual assets; all collision geoms,
kinematics, joint limits, inertias and actuators are retained.
