# Slide figures

**2026-09-13 渲染更新：** 当前讲解图统一为 B / pose 2，共用白底、灰色工件、柔和绿色工作面和小地面。运行 `python slides/render.py`；图目与数据来源见 [slides/README.md](slides/README.md)。Baseline 和姿态数据只读，旧多物体实验需显式 `--legacy`。以下早期图目及数值按日期区分。

**2026-09-12 更新：** Baseline 已移至 `slides/baseline_algo/`，世界坐标统一为 Y-up（地面 `y=0`），图像画布为纯白。B/pose_1 与 B/pose_2 已完整重跑并通过验证；运行说明见 [baseline_algo.md](slides/baseline_algo/baseline_algo.md)。Git 保存代码、Markdown、渲染模板和固定测试样例；`objects/`、所有 `output/` 及生成的图片、视频和数据保留在本地。新检出环境需要另行准备物体与姿态数据，才能运行完整案例。

> **当前职责（2026-09-11）：** 加工射线禁区关闭，实际工作面仍须避开。Step3 保留原贪心算法；A1-f/pose_3 上游未完成。Step4 只输出地面需求点和连续外包点；Step5 先找共同退出方向，再造有间隙的框架、需求凸包对应的开口底座和粗连接，最后验证整件轨迹与共享承载，见[当前代码](slides/baseline_algo/step5_connect_support/README.md)。输出统一位于 `output/{object_name}/{pose}/{step_directory}/`，沿用现有阶段目录名。独立汇总和候选目录已删除，不自动恢复；先前接触调整候选也未通过整件承载。下方旧模型及结果须按日期区分。

**2026-09-11 当前更新：** 所有接触头最终连成一个刚体；Step2 取消竖直半球筛选，Step3 贪心保持不变，Step4 输出整体地面需求点云与连续外包，Step5 检查共同轨迹并连接共享底座。入口与运行结果见 [当前 baseline](slides/baseline_algo/baseline_algo.md)。下文独立四脚与逐面向下约束属于本次更新前的历史实现。

Repository layout (2026-09-06): the working directories are `objects/`, `slides/`
and `codes/`. Write code only in `slides/` or `codes/`; `objects/` stores object
data. Shared helpers live in [tools/](slides/tools/README.md), including the direction
cone model and mesh export used by the slides. Optional tool figures are written
under `slides/tools/figures/`.

The old root `pipeline/`, `torque/`, `figures/`, `METHOD.md` and checkpoint cache
have been removed. Mentions of METHOD or pipeline sections in older notes are
historical context; current assumptions are in [problem_statement.md](slides/problem_statement.md#当前决定与讨论记录) and the page notes.

The current baseline is documented in [baseline_algo](slides/baseline_algo/baseline_algo.md).
2026-09-11: Step 4 uses one deterministic, fixed four-pad footprint per selected contact.
Step 5 connects each support independently and cannot move or resize those feet. Failed supports
retain their status and images; successful supports retain solids, STL files and insertion animations.
The overview and each finished support preview are saved during construction, so an interrupted
later search retains the completed outputs. Bars must remain strictly above the floor; only the fixed pads bear on it.
`--connection-workers N` constructs independent supports in isolated processes and publishes each
completion immediately; final records retain the original contact order for assembly checks.
The complete assembly and any compatible subset are checked separately. Partial geometry does not
certify full load coverage. See [Step 4](slides/baseline_algo/step4_floor_contact/README.md),
[Step 5](slides/baseline_algo/step5_connect_support/README.md), and the
历史独立输出（已删除；不自动恢复）.

Step 2 requires every actual contact face to press downward (`n_out,z < 0`). Step 3 selects and
optimizes contacts using joint force–moment coverage, retains each support's own insertion directions,
and checks continuous loads after all samples pass. Step 4 reuses the same reactions in the workpiece
and individual-support equilibrium equations while keeping feet fixed. Step 5 checks complete solids,
working-volume clearance and sequential installation. Structural strength and compliance remain unverified.

Read [当前决定与讨论记录](slides/problem_statement.md#当前决定与讨论记录) before continuing research.
The robot holds `T*` during insertion; process loads range from zero to `0.5mg`, including gravity alone.
Current exports live under `slides/baseline_algo/output/<object>/pose_<number>/`.
A1-f, B and C5 each have four target poses. Older common-ring and whole-hull-only results are historical.

Pages, in the order of the argument (a deleted page keeps its number, so the numbering
has a gap at 2). Each drawing has a reproducible Python entry point,
and every one that measures something has a `.md` beside it carrying its numbers, its
decisions and its known issues. (Page 4 measures nothing; its docstring is its document.)
The three objects are **A1-f**, **B** (the bunny) and **C5** (the chair) — five tips each
on pages 1 and 3, except **B, which has two**: five is a ceiling, a row is a genuinely
different tip or it is not drawn, and the bunny has two within that historical tipping
search (`setup/poses/big_tip.md`). The new named target catalogue also permits held,
ear-tip-down poses outside that search.

| | page | what it says |
|---|---|---|
| 1 | `setup/poses/tip_<name>.png` | a workpiece **at rest**, tipped **60–120° about one visible ground point**, into a pose it cannot self-balance in, with a connected **8–15 %** work region drawn on it. Three columns a row. `setup/poses/big_tip.md` |
| 3 | `sys_floor/on_the_floor_<name>.png` | the same pose and viewing angles, with the camera fitted to the workpiece and landing cloud; **where the load can land** scattered on the floor. `sys_floor/on_the_floor.md` |
| 4 | `sys_floor/row3.png` | a planar illustration of row (3) in three steps, using the full force vector `F_push` with fixed magnitude `0.5mg`. |
| 5 · opening | `obj_supp/two_equations.png` | open `obj_supp` with the first two equations from setup: workpiece force and torque balance, satisfied by the same pressure field. |
| 5 · mapping | `obj_supp/demand/demand_equation.png` | `demand(F_push, pt) = (F_D, tau_D)`: one push produces one ordered force–torque pair, the right-hand sides of the opening equations. |
| 5 · paired views | `obj_supp/demand/demand_pairs.png` | force-direction coverage on the sphere and a moment relief with a height ruler, following `pipeline/step2`'s drawing style on B tip 1. Selected paired samples share colours and numbers; no lines connect the spheres. `obj_supp/demand/demand.md` records the demand field and its aggregation. |
| 5 · solution | `obj_supp/solution/solution.png` | two stacked 3×3 blocks (identity and cross-product matrix) times the expanded Fx, Fy, Fz components; then a sum of these individual matrix-times-force terms, equalling the same six-component demand. `obj_supp/solution/solution.md` |
| 6 | `obj_supp/area/area_<A1-f|B|C5>.png` | three objects, four configurations each: one small support, triple its actual contact area, two supports, and three supports. Numbered close-ups show every region. Each row gives one covered percentage from deterministic integration: polygon area over feasible, visible positions, then adaptive quadrature over force direction. Whole-domain coverage simplifies the ratio to 100%. No confidence intervals, sample fractions, sizes or areas are annotated. `obj_supp/area/supply.md` records the measure, numerical refinement and mesh model. |
| 7 | `trajectory/sweep_eq.png` · `trajectory/sweep_demo.png` | the insertion condition: a support slides in along the floor in one straight line, and **what it sweeps on the way never enters the workpiece** — `Sweep(S, a) ∩ interior(W) = ∅` as one page, then a claw around B's tail with two slide-in directions, a blocked corridor (42.44 cm³) and a clear one (0); its 1.79 mm gap makes it a corridor example, not yet a fitting load-bearing design. `trajectory/sweep_demo.md`. |

**`setup/equations/`** holds the problem statement itself: `setup/equations/equations.md` writes all four
rows out with one notation and is where the coupling between them is stated, and
`setup/equations/three_equations.png` (drawn by `setup/equations/three_equations.py`, self-contained) is
the figure — the three rows as one shape. Its four-row long form, `four_equations.png`, was
judged wrong and deleted on 2026-08-26; the script's docstring records the trades. **`obj_supp/area/supply.md`** describes the A1-f/B/C5 contact-area demos, their input tables,
colours and reproduction commands. `problem_statement.md` is the
prose version. **`obj_supp/equations_to_solve.md`** goes
further on rows (1) and (2) alone: the declarations of 2026-08-28/29 (`K = 0.5`, unbounded
`λ`, no minimality objective) and the characterisation of the **solution set**, which no
other document carries.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/setup/equations/three_equations.py  # 1 s  the problem-statement figure
python -u slides/setup/poses/big_tip.py          # 19 s   pages 1
python -u slides/sys_floor/on_the_floor.py #        page 3   (re-runs big_tip)
python -u slides/sys_floor/row3.py         #  1 s   page 4
python -u slides/obj_supp/two_equations.py         #  1 s   obj_supp opening: rows (1)(2)
python -u slides/obj_supp/demand/demand_equation.py #  1 s   demand mapping
python -u slides/obj_supp/demand/demand.py          #        demand coverage sphere + moment relief
python -u slides/obj_supp/solution/solution.py     #  1 s   geometry matrix × force components = demand
python -u slides/obj_supp/area/area.py             # three objects, four support configurations each
python -u slides/obj_supp/area/verify_area.py      # independently check saved primal/dual evidence
python -u slides/obj_supp/area/continuous.py --check # replay continuous certificates and counterexamples
python -u slides/obj_supp/area/coverage.py --check # check deterministic geometry and integral records
python -u slides/trajectory/sweep_eq.py    #  1 s   page 7, the condition
python -u slides/trajectory/sweep_demo.py  #  2 s   page 7, the demo     (re-runs big_tip)
```

Pages 3 and 7 call `big_tip.sweep` to get the poses, cameras and work regions, so they
rewrite page 1's three PNGs on the way past — **byte-identical**, and that is checked.
Run them one at a time: they share `objects/<name>/_big_tip_tmp`. Area demos prepare
their poses and renders using temporary object copies.

## What was deleted, and why it is recorded here

**`obj_supp/invoices/`** *(2026-09-06)*. Its required B/C5 tip-1 demand tables and
sampling code now live in `obj_supp/area/`, alongside both area demos. The demand
figure checks the relocated B table; trajectory imports its shared colours from
`slides/tools/cover.py`. Both area demos were regenerated and checked before the old
directory was removed.

**`demo_<A4|D1|A2|D2>.png` and their `_p1..p4` panels, and `gallery.png`.** An earlier
demo — a workpiece tipped short of its balance angle and held by one free-standing block,
four panels an object, on a different set of objects. It was superseded by the pages above
when the work moved to the 60–120° single-point tip and to the four-row formulation; A4's,
A2's and `gallery.png`'s files had already gone and this README still described all of
them. Nothing in the repo referenced them.

**`sys_floor/solved.*`, `sys_floor/solution_eq.*` and `sys_floor/three_steps.*`.** Three
earlier cuts of what is now page 4. `row3.py` is the one that is kept; its docstring records
what came off the picture and why.

**`setup/README.md`** *(2026-08-30)*. 371 lines documenting `tip_sequence.py` as the page
that draws the three `tip_<name>.png` files — which it has not been since 2026-08-27, when
`big_tip.py` took those three filenames over. Its own header already said so, and said its
md5 table was "of the page it writes and not of the file now at that name", i.e. of nothing
on disk; it also pointed at `pipeline/region/make_region.py`, which no longer exists.
`setup/poses/big_tip.md` is the note for the page that is actually drawn. **`tip_sequence.py`
itself stays and must not be deleted** — it is no longer an entry point but it is the
scene, camera and page-fitter library that `big_tip`, `on_the_floor`,
`patch` and `sweep_demo` all import.

**`sys_floor/coverage.py` and `coverage.md`, and `sys_floor/support_polygon.md`.** The
region page and the plan view. Both drew the same quantity page 3 draws — the plan view as a
star hull about the plumb point, the region page as `L`, `conv L` and the ring of feet — and
both were correct; neither read beside the setup slide, and the plan view's own three PNGs
had already gone. `sys_floor/support_polygon.py` **stays, trimmed to `cop` and
`exact_R_min`**, which is all page 3 ever imported from it. What the two pages carried and
page 3 does not is the historical *area* of the landing set and a ring construction.
The former `Σ arccos(ρ/Rᵢ) ≥ π` sufficiency claim is superseded: actual bearings matter.
`setup/equations/equations.md` retains the equal-radius, equally spaced special case and
the general convex-hull containment check; neither is drawn any more.

**`obj_supp/cloud.py` and `cloud.md`, and `obj_supp/cloud_<A1-f|B|C5>.png`.** An earlier
cut of page 6's demand set — the rows (1)(2) pushes drawn as a raw 6-D scatter, one point a
push, a force panel beside a torque panel, the cone's extreme rays on top in red. Superseded
on 2026-08-29 by `obj_supp/invoices.py` and `invoices_<name>.png`, which draw the same set
the way `pipeline/step1` draws it — three balls, the moment line owed as a relief — and that
is the drawing the owner preferred. The extreme-ray layer was one number, not a page:
789–1225 extreme rays a tip, 7.5–11.7 % of the pushes, because the demand set is curved and
nearly every boundary push is a hull vertex. Nothing else in the repo referenced them.

**`setup/balance.py`, `balance.md` and `balance_<A1-f|B|C5>.png`** *(2026-09-01)*. The
"stood on one point of itself" demo page. Every number in it was exact and checked, but the
poses are reached by ROLLING — the contact migrates — which the too-heavy-to-lift premise
forbids, its own notes said "it is a demo page; `big_tip.md` is the one to quote", and
nothing downstream used its poses. It also looked more precarious than it was: its rest debt
`w(0)` (15–27 mm) was no larger than the tip pages' own. The deck's setup is page 1 alone.

**`supp_obj/` merged into `obj_supp/`, and most of it deleted** *(2026-09-01)*. What
survived then lived as three page folders inside `obj_supp/` — `demand/` (page 5, with
the joint-equation notes at that time), `invoices/` (page 6) and `area/` (page 6b). Deleted: `demand_space.{py,png}`
(the schematic), `order_book.{py,png}` (the one-push equation), `settle.{py,png}` (the
settlement columns), `skin.{py,md}` and its four PNGs (the per-point supply relief and axis
field), and `patch_<name>.png` with `patch.md` (the patch-verdict pages). `patch.py` itself
stays, in `obj_supp/area/`, for the active contact-mesh and rendering helpers; unused selectors and solvers were removed;
`supply.md` (in `obj_supp/area/`) now documents the three-object area demos. The joint equations
are in `equations_to_solve.md` and the current expanded formula is in `solution/`.

**`obj_supp/demand/` rebuilt, 2026-09-06.** Removed the old `demand_A1-f.png`,
`demand_B.png`, `demand_C5.png` and the separate-ball/coverage drawing code. The current
pages show the demand mapping, force-direction coverage and moment relief; they do not
report support counts or feasibility coverage. Coupled support mathematics moved to [equations_to_solve.md](slides/obj_supp/equations_to_solve.md),
the corrected cone/search helpers and their regressions moved to `slides/tools/contact_cones.py`
and `slides/tools/test_contact_cones.py`, and the area page now owns the small tile-mask helper
it previously imported from `demand.py`. Existing invoice data and area results remain
their original experiments. The intermediate curved-arrow and wire-sphere scatter pages
have also been replaced. The current figure reuses `pipeline/step2`'s surface/relief
drawing components for demand, retaining paired sample provenance. Separate display is
allowed; feasibility still uses the whole pair.
See [demand.md](slides/obj_supp/demand/demand.md).
