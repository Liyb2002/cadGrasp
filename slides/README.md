# Current slide figures

The setup and floor pages show five saved object/pose pairs: B/pose_2, B/pose_3,
A1-f/pose_2, A1-f/pose_3 and C5/pose_2. The other current illustrations use B/pose_2.
`tools/slide_scene.py` centralizes the white canvas, faceted grey body, muted green
working surface, Y-up camera `[0.8, 0.12, -1]`, and finite ground plane. The ground
margin is 13% of the object's maximum extent; depicted floor data can enlarge it.

The [working-area schematic](setup/working_area.png) explains the green patch:
12 red arrows show possible process forces at different positions and directions.
Their tips lie on the actual working surface, using saved reachable load samples.
They represent separate possible loads; arrow lengths are illustrative.

Use the existing cadgrasp environment and local object, setup and baseline data:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/render.py
# Redraw setup and floor, and remove the known retired slide exports:
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/render.py --only setup floor --clean-old
```

| Case | Target pose | Required floor loads |
|---|---|---|
| B / pose 2 | [Setup](setup/poses/B/pose_2/pose.png) | [Floor](sys_floor/on_the_floor_B_pose_2.png) |
| B / pose 3 | [Setup](setup/poses/B/pose_3/pose.png) | [Floor](sys_floor/on_the_floor_B_pose_3.png) |
| A1-f / pose 2 | [Setup](setup/poses/A1-f/pose_2/pose.png) | [Floor](sys_floor/on_the_floor_A1-f_pose_2.png) |
| A1-f / pose 3 | [Setup](setup/poses/A1-f/pose_3/pose.png) | [Floor](sys_floor/on_the_floor_A1-f_pose_3.png) |
| C5 / pose 2 | [Setup](setup/poses/C5/pose_2/pose.png) | [Floor](sys_floor/on_the_floor_C5_pose_2.png) |
| All five | [Setup gallery](setup/poses/target_poses.png) | [Floor gallery](sys_floor/on_the_floor.png) |

| Group | Other current figures |
|---|---|
| `setup` | [Working-area forces](setup/working_area.png); B/pose_2 aliases: [target pose](setup/poses/target_pose.png), `setup/poses/tip_B.png` |
| `floor` | [Resultant and floor](sys_floor/row3.png); B/pose_2 alias: [floor](sys_floor/on_the_floor_B.png) |
| `area` | [Contact area and combinations](obj_supp/area/area_B.png) |
| `demand` | [Paired force–moment demand](obj_supp/demand/demand_pairs.png) |
| `heads` | [Head forces](obj_supp/demand/head_total_force.png), [common insertion direction](obj_supp/demand/head_sweep.png) |
| `equations` | [Setup equations](setup/equations/three_equations.png), [workpiece equations](obj_supp/two_equations.png), [Step3 conditions](obj_supp/demand/demand_equation.png), [matrix expansion](obj_supp/solution/solution.png) |
| `trajectory` | [Insertion](trajectory/sweep_demo.png), [sweep equations](trajectory/sweep_eq.png) |

Each floor page independently recomputes its 32,768 paired loads and checks the
results against its own Step4 data. Loads range from zero to 0.5 mg in a reachable
30-degree inward cone. The area percentages are sampled joint contact coverage,
not continuous-domain integrals or final support certificates. The insertion
illustration uses the actual B/pose_2 Step5 support and checks its continuous path.
The opposed-head illustration retains its exact local-cone proof.

Retired images were deleted, including old pose galleries, unselected pose
previews, old A1/C5 exports and the independent insertion-study images. Historical
numerical records and setup data remain. `render.py` verifies that baseline files
and setup inputs have not changed; it never invokes a baseline stage or the
separate `setup/poses/target_poses.py` dataset builder. The explicit `--clean-old`
option only removes known retired exports, excluding all baseline outputs.

Git stores rendering code and Markdown. Generated PNG/JSON/NPZ/media remain local
under the repository's existing ignore rules.
