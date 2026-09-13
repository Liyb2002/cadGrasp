# Current slide figures — B / pose 2

All current workpiece illustrations use the saved B/pose_2 geometry and work
region. `tools/slide_scene.py` centralizes the white canvas, faceted grey body,
muted green work surface, Y-up camera `[0.8, 0.12, -1]`, and finite ground plane.
The ground margin is 13% of the object's maximum extent; only actual depicted
floor data can enlarge its footprint. Main views preserve the same orientation;
contact insets turn the camera toward the local surface without moving the part.

Use the existing `cadgrasp` Python environment and local object/setup/B/pose_2
baseline data:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/render.py
# Or redraw selected groups:
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/render.py --only setup floor heads
```

| Group | Figures |
|---|---|
| `setup` | [Target pose](setup/poses/target_pose.png), also at `setup/poses/tip_B.png` |
| `floor` | [Load locations](sys_floor/on_the_floor_B.png), [resultant and floor](sys_floor/row3.png) |
| `area` | [Contact area and combinations](obj_supp/area/area_B.png) |
| `demand` | [Paired force–moment demand](obj_supp/demand/demand_pairs.png) |
| `heads` | [Head forces](obj_supp/demand/head_total_force.png), [common insertion direction](obj_supp/demand/head_sweep.png) |
| `equations` | [Setup equations](setup/equations/three_equations.png), [workpiece equations](obj_supp/two_equations.png), [Step3 conditions](obj_supp/demand/demand_equation.png), [matrix expansion](obj_supp/solution/solution.png) |
| `trajectory` | [Insertion](trajectory/sweep_demo.png), [sweep equations](trajectory/sweep_eq.png) |

The area and demand pages use the same 32,768 saved paired loads, with magnitude
between zero and 0.5 mg and a reachable 30-degree inward cone. Area percentages
are sampled joint contact-model coverage, not continuous-domain integrals or
final support certificates. Floor points are independently recomputed and
compared to Step4. Insertion geometry uses the actual Step5 support, with its
continuous path replayed before rendering. Head-force arrow lengths remain
illustrative. The opposed-head example retains its exact local-cone proof.

`render.py` checks that baseline files and setup data have not changed. It does
not invoke a baseline stage, perform pose search, or write to `objects/`.
Historical A1/C5 figures, pose-catalogue previews and research tables retain their
real identities. Old experiment entry points require `--legacy`; they are not
part of the current deck renderer. `setup/poses/target_poses.py` is still the
separate dataset builder.

As elsewhere in this repository, Git stores the rendering code and Markdown.
Generated PNG/JSON/NPZ/media files remain local under the existing ignore rules.
