# B/pose_2：三个载荷，三个地面点

当前图：[row3.png](row3.png)。三个独立图片：
[1](row3_force_1.png)、[2](row3_force_2.png)、[3](row3_force_3.png)。

```sh
python slides/sys_floor/row3.py
python -m unittest discover -s slides/sys_floor -p 'test*.py'
```

图直接读取 `objects/B/poses.json` 和 `objects/B/tasks/pose_2/setup.npz`，
使用当前 pose_2 和已缩小的工作面（此姿态为总表面积的约 6.74%），
不读取旧 baseline 的同名姿态，不修改姿态、工作面或 baseline 输出。
三幅使用相同相机与比例，每幅只有一个加工力和重力：

- 红色 `F_push` 作用于真实工作面上的 q，方向在保存的 30° 内向法线锥内。
  第 1、2 幅均为 0.5 mg，作用位置和方向明显不同；第 3 幅为 0.1 mg，
  与第 2 幅保持完全相同的作用位置和方向，只改变大小。每幅单独标注力度。
- 蓝色 `mg` 从质心 c 竖直向下；蓝色虚线是它的作用线。
- 黑点 X 是红、蓝两条作用线的真实交点，不是箭头端点，也不是两个作用点的加权平均。
- 橙色 W 是合力；橙色虚线与 `z=0` 的交点 p 是该载荷对应的地面点。

三个力箭头共用同一屏幕长度／力大小比例：在 1100 像素单图中，1 mg 对应 240 像素，
0.5 mg 为 120 像素，0.1 mg 为 24 像素。方向沿真实三维方向的投影；
箭头表示力度而非位移，因此消除方向投影造成的长度缩短，让相同力度显示为相同长度。
q、c、X、p 和辅助线仍使用准确的三维投影。红色实体箭头沿原作用线放在 X 沿受力方向的前方，
箭尾与 X 至少相隔 64 像素，并与表面红点 q 留出间隙；红、蓝箭头向后的虚线延长线相交于 X，红色作用线经过 q。
橙色实体箭头也沿合力作用线移开 X；这些平移不改变力的作用线或力矩。
第三幅保留独立的短实体箭头，尖端不再被 q 的标记遮住。
物体与工作面是原始几何的深度渲染；
受力与辅助线作为自由体图叠加，因此物体内部质心处的重力仍可见。
地面只标该幅的一个 p，没有把它画成物体原有的接地点，也不宣称该姿态已经稳定。

## X 的核对与三维适用范围

旧二维程序中的 X 数值确实同时落在重力和推力作用线上（交线残差小于 1e-12），
问题在于箭头、延长线和标注让构造不够清楚。现在直接画当前物体的三维几何。

三维中的任意两条作用线可能异面，不能总是假设存在 X。为说明“经过 X 的合力
延伸至地面”这个构造，这三个载荷专门从当前工作面和载荷锥内选取**作用线相交**的例子。
程序先在质心竖线上取 X，再检查由 X 指向 q 的推力是否在载荷锥内、是否能从外部到达 q，
先选择两个作用位置与方向明显不同的 0.5 mg 例子，再将第二个的力度缩小为 0.1 mg。
第二、三幅的 X 相同；减小力度后 p 更靠近质心的竖直投影。
它们不代表整个三维载荷集合。

世界坐标 Z 向上，力以 mg 为单位、位置以米为单位。令

\[
G=-mg\hat z,\quad W=G+F_{\rm push},\quad
M=c\times G+q\times F_{\rm push},\quad N=-W_z>0.
\]

交线构造给出 `p = X - (X_z/W_z) W`。程序另用力矩公式
`p = (M_y/N, -M_x/N, 0)` 独立复算，并验证完整三维残差
`M - p × W = 0`，而不只检查两个水平力矩。
这里采用作用点与力组成力矩 `r × F` 的标准 wrench 表示；参见
[Modern Robotics §3.4](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/)。

一般异面载荷仍可用上述力矩公式求地面压力中心，但可能剩下绕竖直轴的力矩，
不能将它画成一个纯合力作用点。回归测试包含这种反例，以及没有唯一 X 的平行力情况。
这里不重新讨论摩擦大小；整套被动支撑是否可承载仍由完整平衡求解判断。

## 其他保存图的来源

`on_the_floor.png` 和 `resultant_B_pose_2.png` 保留的是旧 baseline 输入的图，
不对应这次 `objects/` 中更新的姿态与工作面。`presentation.py` 重画它们时仍读取
baseline 保存数据，并在末尾调用新的 `row3.py`；本次只重画 row3 的三个例子。
下文记录更早的实验，不能作为当前图片的数值说明。

## Historical experiments

# `on_the_floor` — the target pose, with the landings on the ground

**2026-09-11 接入 baseline：** `support_polygon.cop` 的整体压力中心公式被新 Step4 的等价六维需求映射采用；当前 baseline 使用 Step1 的当前姿态及 0–0.5mg 全范围，含零加工力，并另存连续外包。下面的旧图、固定力度和旧姿态样本不作为当前认证。见 [Step4](../baseline_algo/step4_floor_contact/README.md)。

**Current scope (2026-09-06): [README.md](../README.md#当前决定与讨论记录).** Only the full-load set
`L_K={p(q,d,K)}`, `K=0.5`, is required. References below to varying `[0,K]` describe an
optional extension and historical area measurements; they add no zero-load requirement.
The floor hull check treats the workpiece and its supports as one assembly and uses the
convex hull of the assembly's floor contacts. Separate support-by-support floor checks
are outside the current scope; see [equations.md](../setup/equations/equations.md).

`slides/setup/poses/big_tip.py`'s target pose, **unchanged** — same pose, same camera angles,
same grey floor, same green work region, **and no mark on the contact** — with the
places the load can land scattered on the ground. Nothing else on the panel.

`row3.py` / `row3.png` explain the construction in the drawing's plane: intersect the
two coplanar load lines at `X`, then extend their resultant to the floor at `p`.
The construction remains planar. `F_push` is the complete force vector, and both the
resultant equation and arrow label use that notation. The quantified load is fixed at
`|F_push| = 0.5 mg`; no range of force magnitudes is required.

**Two declarations changed on 2026-08-27, and between them they are the difference between
a fixture and a building.** The work region now grows only through faces the process can
reach (`big_tip.md`), and **`K = 0.5`**: the process pushes with half a body weight, not a
whole one. The load's reach from the plumb point went from **0.48–7.78 part widths to
0.18–0.32**, and **3 of the 11 poses that had no bounded region at all now have one** —
every pose does.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/sys_floor/on_the_floor.py
```

| file | md5 | rows |
|---|---|---|
| `on_the_floor_A1-f.png` | `6b86ed85138910d02035afacf21834bf` | 5 |
| `on_the_floor_B.png` | `9236ec1a137156051aa73ec2c4381379` | **2** |
| `on_the_floor_C5.png` | `00e5565465092e4b0440af0203e97826` | 5 |

**Regenerated and checked on 2026-09-06 after the visibility correction.** Across all
12 poses, the visibility mask, workpiece rendering and dot projection use the same
final camera. All 196 394 drawn landing centres pass the workpiece ray test, and every
retained landing fits inside the panel with room for the dot radius. The five floor-model
regressions pass. Setup generation and rendering scratch stayed in temporary copies of
the object directories; the three generated setup PNGs matched the originals byte for
byte, and the original setup PNGs were unchanged.

`row3.png` was regenerated with fixed vector magnitude `|F_push| = 0.5 mg` (md5
`047f295fc4bca5a387224f84f78ab3b1`). Its planar line-intersection result agrees with both
the lever rule and `support_polygon.cop` to `1e-12`; the planar construction is retained.

**B went to two rows on 2026-08-28**, and A1-f's and C5's md5s did not move with it. Twice:
it went from one row to five that morning, when `big_tip` padded a short page with repeats
of the tip it had (`top_up`), and to two that evening, when the padding was replaced by a
wider PLACEMENT SOURCE (`big_tip.md`'s `hull_places`) and every row on every page became a
genuinely different tip. The five were the same corner 0.5–3.5° apart and their landing sets
were near-copies, 41.2–44.1 mm, a spread of 7 %. **The two are two different poses of the
bunny** and their landing sets are not copies: 43.8 mm and 68.2 mm, 0.18 and 0.27 part
widths.

**The previous render was checked for determinism:** two runs gave three identical
md5s before the 2026-09-06 visibility correction. The poses, camera angles and work
regions are `big_tip.sweep`'s own — this page calls it and reads the records it hands
back. The floor page then fits its frame to the workpiece and landings.

## The dots sample the cone rim; the boundary also depends on folds and visibility

Row (3) cuts the free body around the **workpiece and its supports**, so the supports'
forces on the workpiece are internal and cancel, and where the load lands on the floor
depends only on gravity and the process push. The formulas below describe that map at
**`K = 0.5`**. This page evaluates 220 contact points and 96 rim directions per point;
it does not compute the complete continuous boundary.

**1 · A lever rule.** With `g = (c_x, c_y, 0)` the plumb point of the centre of mass, and
`h = q − (q_z/d_z) d` the point where the push's *own* line of action crosses the floor,

> **`p = ( 1·g + (−t d_z)·h ) / ( 1 − t d_z )`**

— the centre of pressure is the average of those two points, weighted by their vertical
force components (1 down for gravity, `−t d_z` for the push). `t = 0` gives `g`; a downward
push puts `p` between the two; an upward one puts it on the far side of `g`, and
`t d_z → 1` sends it to infinity, which is the assembly on the point of floating.

**2 · Optional magnitude extension: the region is a fan.** With `r = q − g`,

> `p − g = t (r_z d_h − d_z r_h) / (1 − t d_z)`

and `t ↦ t/(1 − t d_z)` is monotone on `[0, K]`, so a fixed `(q, d)` sweeps the straight
segment from `g` to its full-magnitude end. The union over those magnitudes is
**star-shaped about `g`**; each ray reaches furthest at **`t = K = 0.5`**, which is the
magnitude drawn on this page. Only that endpoint set is required by the current task;
its interior is not filled by an assumed `[0,K]` load range.

**3 · Stereographic projection gives a rational map.** Substitute
`u = d_h/(1 − d_z)`, the projection of the push direction from the north pole, for which
`d_h = 2u/(1+|u|²)` and `d_z = (|u|²−1)/(1+|u|²)`. At full magnitude:

> `p(u) = g + K[(1−|u|²)r_h + 2r_z u] / [(1+K) + (1−K)|u|²]`
>
> **`p(u) = g + [(1−|u|²)r_h + 2r_z u] / [3+|u|²]` at `K = 0.5`.**

The former quadratic expression is the special case **`K = 1`**. A full, unoccluded
cone that excludes the north pole maps to a disc; one containing the pole requires the
exterior chart or a second projection chart. The original formula in `d` is well defined
even at the pole because `1 − K d_z ≥ 0.5`.

**4 · The boundary can come from the rim, a fold, or an occlusion edge.** For a work-region
point above the floor (`r_z > 0`), the determinant at `K = 0.5` is

> `det DΦ(u) = 4r_z [r_z(3−|u|²) − 4r_h·u] / (3+|u|²)³`.

Thus the critical directions satisfy **`d·r = K r_z = 0.5 r_z`**. For the full cone,

> **`∂Φ(C_full) ⊆ Φ(∂C_full) ∪ Φ(C_full ∩ {d : d·r = 0.5 r_z})`.**

The script counts how many sampled contact points have a full cone intersecting that
critical circle. This is a diagnostic before visibility filtering: it does not establish
which parts of the fold remain reachable. The line-of-sight filter may also create
boundaries inside the original cone. Even a zero fold count would therefore not certify
the rim as the entire visible boundary. The `N_PHI = 96` rim angles themselves are
finite samples.

**Rechecked on 2026-09-05 at `K = 0.5`: 0 of 2 640 sampled full cones meet the critical
circle**, over the current 12 poses (5 A1-f, 2 B, 5 C5). Each pose has 0 of 220. The sampled
landings reach **0.18–0.32 part widths** from `g`; no point is dropped by the frame clip.
At that check, all three floor PNGs reproduced the pre-correction hashes. The setup PNGs
were regenerated in a temporary directory, matched their originals, and the originals
were left untouched.

**5 · The region is bounded at half a body weight.** A singular denominator would require
`1 − K d_z = 0`. At `K = 1` that can happen for a push straight up. At **`K < 1` it cannot
happen at all**, since
`d_z ≤ 1` gives `R ≥ 1 − K`: at `K = 0.5` the assembly's total normal force never falls
below half a body weight and the denominator's amplification is at most 2. The current
`support_polygon.exact_R_min` includes the interior maximum `d_z = 1` when a full cone
contains straight up. Its minimum over full cones is a lower bound after visibility
filtering; all current poses have positive total normal reaction.

**6 · Historical area of the magnitude-extended set.** That set is star-shaped about `g`, so **`A = ½∫₀^{2π} ρ(θ)² dθ`** with `ρ` the radial
support of the union — a shoelace sum once the outline is a polygon. **This page does not
compute it**: the two pages that did, the plan view and the region page, are deleted, and
what survives of them is `support_polygon.py`'s `cop` and `exact_R_min` alone.

> **The assembly normal-reaction equations admit some force distribution when every
> required landing is inside the convex hull of its floor contact.** The floor can only
> push, which permits eliminating its forces into a hull test. This is the current
> assembly-level floor check, alongside the workpiece contact equations (1)(2).
> The picture is the argument: the workpiece touches the ground at ONE point — every pose
> here is a `pivot == "point"` pose — a point's convex hull is itself, and the landings are
> all over the floor round it. **The contact is not marked**, so that one point is where the
> part meets its shadow and the sentence has to carry it; the setup page is where it is
> drawn.

### `K`, and what it is

`K` is METHOD §1's declared disturbance magnitude, **in body weights of the workpiece**:
against a push of `K` body weights the contacts must produce `T = up − K·d`.
**`support_polygon.K = 0.5`** is shared by this page. Both `cop(..., t=K)` and
`exact_R_min(..., t=K)` take `t` as the actual magnitude in body weights, with that
default. The removed METHOD and PIPELINE documents used the historical `K = 1` model.

In newtons, `K = 0.5` is **2.7 N** on A1-f (555 g), **3.6 N** on B (735 g) and **0.09 N**
on C5 (17 g). Since `K` is dimensionless, one `K` for every object declares a different
physical force for each. For an illustrative fixed 1 N load, `K` would instead be
0.18 on A1-f, 0.14 on B and **5.9 on C5**; the last load is outside the current domain.
Workpiece mass is selected to suit the experimental equipment and task. Pickup and
reorientation are allowed during preparation. Changed mass, center of mass or task
loads require checking the design against the resulting load domain.

### What sampling the cone's interior was costing

The version before this drew 48 directions taken at random from *inside* each cone.
For these historical rows, sampling the rim found more distant landings. These are two
finite-sample measurements, not a comparison against an exact continuous boundary:

| | A1-f 1 | A1-f 2 | A1-f 3 | A1-f 4 | A1-f 5 | B 1 | C5 1 | C5 2 | C5 3 | C5 4 | C5 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| interior sample | 0.79 | 1.39 | 2.47 | 3.07 | 3.10 | 0.48 | 5.41 | 5.52 | 5.60 | 2.20 | 1.97 |
| **the rim** | **0.85** | **1.52** | **3.21** | **3.39** | **3.34** | **0.48** | **5.78** | **7.78** | **6.99** | **2.36** | **2.10** |

(Both rows are at `K = 1` and the old work region.) These historical reaches do not
validate the boundary at `K = 0.5`; the current point cloud samples both contact points
and rim angles, and does not draw the fold or occlusion-edge images.

## Drawing decisions

- **The frame is pulled back until the landings are in it.** The setup page frames on the
  workpiece and the load lands further out than that on every row, so at its own framing the
  dots run off the panel. The camera keeps the row's elevation and azimuth; its look-at
  point and distance are fitted to the workpiece and all retained landings, including
  hidden ones. Visibility is then evaluated using this final camera, also used for
  rendering and projection. This avoids a framing/visibility loop and keeps newly
  revealed dots inside the panel. Landings beyond `CLIP` = 3 box diagonals
  are dropped instead and the count prints: zero on every one of the twelve rows now (it
  was zero on eight of fifteen, at most 49, before the poses last changed).
- **The dots are drawn flat on the finished render, not put in the scene.** 21 120 of them
  is 21 120 geoms and `tip_sequence.render` is handed 64. What that costs is the depth test,
  so it is done by hand and exactly: a dot is drawn only if the ray from its landing point
  to the eye misses the workpiece (`tip_sequence.buried`'s test, vectorised). The part
  hides the ground behind it, as it should. Visibility always uses the final camera;
  the original implementation calculated it before fitting the frame and reused a
  stale mask after the eye moved.
- **The colour is the deleted plan view's own orange**, which is what
  `torque/code/demo_scene.py` already spends on this quantity. It is not the setup page's
  green work region, and it no longer has to be told apart from that page's orange-red
  contact target: `render`'s `draw` is left empty here, so the target is not painted at all.
  Two warm marks on one panel read as one set, and the landings are what this page is for.

## Known issues

- **Nothing on the panel says where the hull would have to reach.** An earlier version drew
  the smallest three feet that contain the landings, and a plan view beside them; both were
  cut as clutter. Three unrestricted vertices can enclose any bounded planar landing
  set. This page draws the assembly's required landings; it does not choose an actual
  footprint or construct support geometry.
- **`big_tip.sweep` is re-run to get the rows**, so this page rewrites `slides/poses`'s own
  three PNGs as a side effect. They come out byte-identical whichever script writes them,
  and that is now checked — it was NOT true for a while: this page rebound `tip_sequence`'s
  shared `TITLES` at import, so the setup pages went out with this page's single column
  head. The rebinding now happens only around this page's own `page` call.
- **Nothing draws the set as a REGION any more.** `coverage.py` did — its outline, its
  convex hull, and the ring of feet that answers it — and it is deleted with the plan view.
  This page is the point cloud and there is no longer a page for the shape or for the
  requirement; `slides/setup/equations/equations.md` carries both in algebra.
- **Contact points and rim directions are sampled:** 220 area-uniform points over the
  stored work region and 96 azimuths at each. The drawn magnitude is `t = K = 0.5`.
- **The fold and visibility boundaries are not drawn.** The diagnostic tests the full
  cone against `d·r = 0.5 r_z`; it does not trace that curve or the boundaries created by
  occlusion. The point cloud is therefore not a complete boundary certificate.
