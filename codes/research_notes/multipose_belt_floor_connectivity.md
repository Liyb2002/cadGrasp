# 多姿态共享腰带：地板避碰与可连接性

2026-09-21。按用户要求进行的数学推导；子 agent 独立推导，主 agent 检查反例、厚度条件和现有构造代码。该推导本身未运行 baseline 搜索。后续实现更新：`slides/baseline_algo/step3_scheculer/connection.py` 已将后平面充分构造接入单 pose 候选筛选与尺寸调整，并独立重建检查连接实体扫掠。多地板只用于反例测试；实际搜索仍每个 pose 独立求解，不是多姿态共享结构求解器。

## 结论先行

- 把每个 pose 的地板变换到同一个物体坐标系，所有地板上方半空间的交集 P，就是共享蓝块必须留在其中的凸区域。姿态包括固定高度；任意抬高物体会改变问题。
- 各接触面都不碰地板、各头有共同退出方向，不足以保证存在有厚度的共享连接。还要检查头之间在物体外侧、P 内部的可连通性。
- 对现有“沿共同方向伸出脖子，再在物体后方接框架”的构造，可给出一个简单充分条件：后连接平面的位置有下界（越过物体）和上界（某个 pose 下不能伸到地板以下），二者之间有余量，并满足真实厚度的脖子扫掠条件，就能构造地板无碰撞的连通框架。
- 这个条件应当用于比较“接触组合＋共同退出方向”，而非等蓝块完成后才筛。区间为空仅排除该构造，不能宣称任意腰带都不存在。
- 装入 d0 与退出 u 的符号相反。所有 pose 的地板约束用于蓝块的最终状态；首次装配的中间运动只需检查实际初始场景，不能无意中要求在每个任务姿态都能装蓝块。

## 当前图示的只读核对

读取 `slides/belt_test/code/fixture_geometry.py` 的三个姿态，将同一蓝块逐顶点代入拉回的地板平面，并与刚体变换后的世界 z 坐标核对，误差阈值 1e-12 m。最低高度分别为：

| 姿态 | 蓝块最低离地距离（mm） |
|---|---:|
| pose_2 | 16.967611 |
| pose_2 + 25° | 13.091206 |
| pose_4 | 101.895232 |

这是当前完整蓝块在三个最终姿态下的地板避碰检查，不是任意接触组合的存在性证明，也不证明承载或搬运过程。

以下保留完整推导与适用条件。

---

# Shared blue connector across multiple poses: geometric derivation

Status: analytic derivation under explicit assumptions; no baseline rerun. The current posed blue example was checked separately by the parent agent. This report does not certify loads, manufacturing, or robot motion.

## 1. The exact multi-pose floor condition

Let the closed solid object be O in object coordinates. Fixed task pose k is T_k(x)=R_k x+t_k. World floor is z=0. Define unit normal a_k=R_k^T e_z and offset b_k=e_z^T t_k. The height of an object-coordinate point x in pose k is

    h_k(x)=a_k·x+b_k.

The same blue solid B avoids all floors precisely when

    B ⊂ P,       P = intersection_k {x:h_k(x)≥0}.

Also require int(B)∩int(O)=empty, with designated contact allowed on the object surface. P is convex. For physically legal poses, O⊂P too. The pose translations are fixed: if arbitrary extra vertical lift is allowed, any bounded B can eventually be raised above every floor and this constraint becomes trivial.

For a triangular mesh solid, minimum height occurs at a vertex. Hence checking all blue vertices against all K planes is exact for that represented polyhedral solid (up to floating point), not surface sampling. It is not an existence test for an as-yet unknown connector.

## 2. Contact clearance alone is insufficient

Analytic counterexample: O=[-1,1]^3. Four legal resting orientations pull the ground planes back to x=-1, x=1, y=-1, y=1, giving

    P=[-1,1]^2 × R.

Choose small contact patches about (0,0,1) and (0,0,-1). Every point of either sufficiently small patch has strictly positive clearance from all four floor planes. Nevertheless int(P)\O consists of two disconnected components: z>1 and z<-1. Any continuous path joining these components crosses z=0; there its x,y coordinates place it inside O. A positive-thickness connector therefore does not exist. Paths along x=±1 or y=±1 are only zero-width boundary paths: thickening them penetrates object or a floor.

This disproves: "all contacts individually clear every floor implies a shared connected belt exists." It also illustrates the quantifier error in constructing separate connectors for separate poses: one B must work in the intersection of all allowed spaces.

The upper and lower head solids can both withdraw tangentially along +x without penetrating the cube. Thus a common head withdrawal direction alone does not rescue the conclusion. A connector joining their external ends would have to extend beyond x=1 and violate one pulled-back floor. This assumes initial installation placement can be selected, as in the current paper scope; it does not require the assembly sweep to respect every task floor simultaneously.

## 3. Exact topological statement and finite-thickness version

For a strictly clear connector centerline, use the OPEN free space

    E=int(P)\O.

Given fixed attachment points p_i in E, there exists a path-connected curve network in E containing all p_i iff the points lie in the same path-connected component of E. Necessity is immediate. Sufficiency: connect every p_i to p_1 by a path and take their finite union. Open subsets of Euclidean space have path-connected components, so connected-component computation has the intended meaning here.

Every such finite network can be chosen polygonal and compact. Its distance to closed E^c is then strictly positive. Thus some sufficiently thin tubular connector exists. This establishes existence of SOME positive thickness, not a prescribed manufacturing thickness and not mechanical adequacy.

Contact surfaces themselves lie on ∂O, not in E. First instantiate the contact heads H_i and identify outward attachment ports. Each H_i must itself be object-compatible and contained in P; the connecting network must intersect head material with positive volume. One cannot simply insert a contact surface point into E or inflate a surface path isotropically into O.

For prescribed round beam radius r and clearance c, define centerline domain

    E_r={x: h_k(x)>r+c for all k, dist(x,O)>r+c}.

Head material is allowed to meet O at its designated contact; these clearance inequalities apply to the connecting beams/ports, not to every point of every contact head. Let A_i⊂E_r be permissible center positions whose radius-r connector overlaps head H_i with positive volume. A sufficient condition is that one component of E_r intersects every A_i. It is also necessary within the stated architecture of radius-r tube networks routed through E_r. It is not a universal necessary condition for arbitrary variable-thickness or noncircular solids. If full heads can themselves bridge centerline components, include head connectivity explicitly in the graph instead of imposing an overly restrictive all-ports-in-one-component rule.

Use a voxel/grid graph only as a discretized algorithm. An accepted graph path needs continuous thick-edge collision and plane checks. Failure of one grid, one thickness, or one connection architecture is not proof of general impossibility.

## 4. Does O⊂P yield an easier shell theorem?

It does not yield universal connectivity, as the cube example proves. A useful sufficient condition is available with regularity assumptions. Suppose relevant object boundary is a smooth embedded surface with an outward tubular collar, and the contact attachment neighborhoods can be joined along a compact surface path entirely in ∂O∩int(P). That path has positive minimum floor clearance. A sufficiently small outward offset and sufficiently thin connector follow the path without hitting O or any floor. This proves a local-shell sufficient condition, not necessity for general nonconvex O.

For a convex body O with nonempty interior, the topology simplifies further. Pick c∈int(O). Along each ray c+tu, both O and P occupy intervals beginning at c. The allowed exterior segment is (rho_O(u),rho_P(u)). Directions with nonempty segments correspond exactly to ∂O∩int(P). Radial projection from E to ∂O∩int(P), with a positive continuous radial offset for a reverse section, identifies their path components. Consequently for convex O, the strict-floor portion of its surface provides the component criterion for arbitrarily thin external connections. Assumptions about valid outward attachments still apply.

For a smooth strictly convex O and finitely many tangent floor planes, each floor touches at only one point; a sphere-like boundary with finitely many points removed is path connected. Arbitrarily thin floor-clear connections can then exist between non-ground contact neighborhoods. This does not yield a usable minimum thickness: the bottleneck can be arbitrarily narrow. Faceted convex objects can have whole floor-contact faces whose removal disconnects the surface, exactly as in the cube example.

## 5. A particularly useful sufficient condition for current rear-plane construction

Current code slides/baseline_algo/step6_connect_support/direction_first.py:loose_frame extends each head along a common WITHDRAWAL unit direction u to a rear plane, then links terminal joints with a tree. This yields a simple conservative analytic test that uses the convexity of P.

Assume:

1. Fixed heads are valid and wholly inside P.
2. Chosen attachment-root centers p_i have floor clearance at least r+c.
3. The radius-r+c neighborhood of each forward ray p_i+s u, s≥0, does not enter O; Alternatively, certify the actual thick neck solids over their entire withdrawal sweep against O; a static neck check alone is insufficient. This is stronger than merely checking a zero-thickness ray.
4. Only object, floors, and the chosen straight withdrawal matter; no omitted work-area or fixture obstacle is silently included in this theorem.

One clean way to obtain assumption 3 is to choose a root ball of radius r+c wholly inside an already certified contact-head solid, and certify that complete head's entire withdrawal sweep. The root-ball extrusion is then a subset of the head sweep. A head-center ray alone is insufficient. A finite head sweep must extend at least to the selected terminal; once the complete moving head lies beyond a separating plane, further outward motion is safe. If no such root ball fits, use the actual cross-section and its certified extruded solid instead.

Let H_O(u)=max_{x∈O} u·x. Choose rear plane u·x=q. Terminal i is

    y_i=p_i+(q-u·p_i)u.

A sufficient lower bound is

    q_min=max(H_O(u)+r+c, max_i u·p_i).

For every pair (i,k) with a_k·u<0, the descending-floor direction gives the upper bound

    q ≤ u·p_i + (h_k(p_i)-r-c)/(-a_k·u).

Let q_max be the minimum of those bounds, or +infinity if no such pair exists. If the roots are initially clear and q_min<q_max, choose q strictly inside the interval.

Proof:

- Height on each neck center segment is affine. Both endpoints satisfy h_k≥r+c, hence the whole segment does; no intermediate floor sampling is necessary.
- All y_i lie in the eroded convex halfspace intersection P_{r+c}. Therefore every straight tree edge between terminal centers lies in P_{r+c}. Its radius-r neighborhood clears every floor.
- Every terminal/tree center has u-coordinate q>H_O(u)+r+c. Hence the complete thick rear frame lies beyond a support plane of O and cannot hit O.
- Continuing the rear frame in +u increases this separating coordinate, so it remains object-clear during withdrawal. Necks are safe by assumption 3. The assembled blue is therefore removable in +u and insertable by reversing it.

This is a SUFFICIENT condition for a specific rear-plane-and-necks construction, not a universal existence theorem. q_min>q_max only rules out this construction for these roots, u, and radius. Another bent connector, different direction, thinner beams, or other contacts could still work.

A still simpler sufficient special case is a_k·u≥0 for every task floor. Then the common outward direction never approaches any pulled-back floor, q_max=+infinity, and sufficiently far rear-plane connection is guaranteed under the other assumptions. This condition is often conservative: finite necks may remain safe even when some a_k·u<0. Insertion d0=-u has the opposite inequality sign.

Actual code uses world-aligned cubic terminal joints and convex tapered beams, not round tubes. For a cube with half-width r, replace plane offset r by its support radius r||a_k||_1, and the rear separation allowance by r||u||_1. Alternatively test actual convex-part vertices; floors are still exact endpoint/vertex tests. A generic spherical-radius formula must not be pasted unchanged into cube-joint code.

Furthermore current loose_frame shrinks each head about its center by factor 0.85 for the neck root, extrudes that cross-section, and raises terminal z if needed. The current world-z adjustment is not an all-pose guarantee. The analytic interval is a proposed extension of this code, not a statement that the implementation already enforces it. New terminal adjustments must preserve every pulled-back plane, or be followed by exact actual-part checks.

## 6. Preserving the complete-blue d0 constraint

For finite withdrawal +u over length L, the forbidden material-placement set due to O is

    S_u=O ⊕ { -s u : 0≤s≤L }.

B can withdraw iff its interior avoids the appropriate obstacle interior throughout the sweep; use explicit contact conventions at s=0. Replace the object obstacle in the free-space/clearance construction by this direction shadow, then intersect with P. All-head withdrawal only says the individual head solids avoid the shadow; it does not imply their external attachment locations connect in the remaining allowed domain.

If initial-pose floor must also remain clear during withdrawal, it adds its own affine endpoint constraint along s. Do not impose every task floor on every intermediate d0 state unless that stronger assumption is actually wanted. The static shared blue still must satisfy all task floors.

## 7. Minimal algorithm change

1. Transform all fixed pose floors into the object frame once.
2. Filter candidate contact head solids by all floor planes before expensive contact-set search.
3. For each surviving Step3 contact-set/common-direction candidate, compute admissible root/neck/rear-plane interval; test actual-thickness heads, necks, terminals and beams.
4. A successful interval and construction provide an explicit connected witness. Compare successful candidates by actual material or other declared objectives.
5. If no rear-plane witness exists, either try another Step3 set/direction or run a broader routing graph in the all-pose, direction-shadow free space. Do not turn this method-relative failure into a general impossibility claim.

Useful reporting: minimum blue clearance over all poses, chosen thickness, component/route witness, continuous full-blue withdrawal result. Treat bearing capacity and work-area access as separate constraints.
