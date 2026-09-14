# 历史讨论：独立支撑的非向下合力

**当前决定已更新：所有接触头和底座连接为一个刚体，局部下压允许。** 本文的逐块条件和论文段落针对先前的独立实体假设，不作为当前逐接触区的限制。当前连接与运动检查见 [Step 5](../step5_connect_support/README.md)。

2026-09-08。本页给出评分的修订定义及对照实验。当前 `contribution.py`、尺寸优化和 scheduler 的主结果仍采用旧的固定接触反力模型；本轮没有把对照报告写成新模型下的贪心结果。

## 1. 按独立支撑约束同一组反力

固定姿态和接触设计 `S`。加工力作用于 `pt`，质心为 `c`，向上单位向量为 `z_hat`。以下方程使用 SI 单位；代码将力除以工件重量 `mg`，力矩相应除以 `mg`。

\[
 b(pt,F_{\rm push})=
 \begin{bmatrix}
 mg\hat z-F_{\rm push}\\
 -(pt-c)\times F_{\rm push}
 \end{bmatrix}.
\]

`G_S` 的列为接触位置与单位内法向构成的六维反力生成元，包括工件原地面支点。`lambda_i >= 0` 为各接触的法向力；同一组 `lambda` 必须同时满足工件的力和力矩方程。令 `I_j` 为独立支撑实体 `j` 对应的接触索引。

无自重、无锚固、未与其他支撑连接时，该支撑的竖直平衡要求

\[
R_j=\sum_{i\in I_j}\lambda_i n_{i,z}\ge0.
\]

`R_j` 是地板对该支撑的总法向力。该不等式是一项必要条件，可以在尚未构造底座时检查；不能替代逐块的完整力矩、摩擦与插入验证。若多个接触区连成一个共同实体，应按该实体分组。

修订后的单条需求判定为

\[
\chi_S(b)=\mathbf 1\!\left[
\exists\lambda\ge0:\quad G_S\lambda=b,\quad
\sum_{i\in I_j}\lambda_i n_{i,z}\ge0\quad\forall j
\right].
\]

每条载荷可以重新选择反力；设计 `S` 在整个载荷域上保持不变。不能分别求一个满足工件平衡的解和另一个满足支撑不等式的解。

对候选 `C`，Step 3.1 的联合覆盖与新增贡献分别为

\[
\widehat{\mathrm{coverage}}(S\cup\{C\})
=\frac1N\sum_{k=1}^N\chi_{S\cup\{C\}}(b_k),
\]
\[
\Delta(C\mid S)
=\frac1N\sum_{k=1}^N
\left[\chi_{S\cup\{C\}}(b_k)-\chi_S(b_k)\right].
\]

因为 `S` 固定，两种排序等价。加入约束后仍然是线性可行性问题，每个独立支撑增加一行不等式。

## 2. 必须同步修改的实现

- **Step 3.1：** 保留每个反力变量的支撑归属，在同一个 LP 中添加逐块不等式。当前 `merge_columns` 会跨块去重；有逐块约束时不能丢掉归属，可只在同一块内部去重。只约束全部支撑的竖直合力没有作用，因为工件平衡已经固定了该总和。
- **Step 3.3：** 所有尺寸都调用同一个新判定。旧块的反力也随载荷重新分配，几何仍固定；效率定义继续使用联合覆盖除以总面积。
- **缓存和批量加速：** 旧通过掩码、反力基和连续证书需要作废或重验。复用一个反力基时，除了系数非负和六维回代，还必须检查每块的竖直合力。
- **Step 3 连续验收：** 外包盒或球冠外包顶点的每个反力证书都要满足逐块条件。原始无限制供给锥的包含证书不能继续作为新模型的证明。
- **Step 4：** 构造实际底座后，用同一组工件接触反力验证每个独立实体的完整受力与地面接触；还要检查实体连接及插入。

局部负竖直法向不必直接删除。同一独立实体内部的向上、向下反力可以组合，只要其总竖直分量非负。对于全部内法向严格向下的整块，不等式迫使其所有反力为零。

如果希望保留现有六维锥分类器，可将每块受限的供给锥单独转换为生成元：保留 `n_z >= 0` 的原始生成元，并对同一块中每个 `z_p > 0`、`z_m < 0` 的生成元对加入

\[
\widetilde g_{pm}=(-z_m)g_p+z_p g_m.
\]

这些组合的总竖直力为零。对负竖直载荷逐一分配该块内的正竖直载荷，再保留剩余正载荷，即可分解任何满足逐块条件的非负反力解。此转换与单块的一条竖直不等式等价；配对必须限于同一实体，生成元数可能按正负项数量的乘积增长。因此直接带不等式的 LP 是更直接的初始实现。

## 3. 不向下压，仍可能覆盖整个任务载荷域

需求竖直分量满足

\[
 b_z=mg-F_{{\rm push},z}\in[0.5mg,1.5mg],
\]

因为 `|F_push| <= 0.5mg`。因此所有任务都需要净向上的接触合力；向上的加工力可以通过减小原有向上支撑反力来平衡，无需自动增加向下夹紧力。能否同时满足水平力与力矩，仍由接触几何决定。

逐块非向下约束使全部供给落在 `F_z >= 0` 半空间中，所以不可能正张成整个 `R^6`。目标应写成固定设计对给定连续载荷域的覆盖，不能写成任意扰动力矩的 force closure。Force closure 的全六维定义参照 [Modern Robotics 12.2.3](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-3-force-closure/)。

本次实验采用更严格的 **每个接触反力都没有向下分量**，即仅允许 `n_z >= 0` 的反力生成元：

| 物体 | 原已选组合在该限制下的采样覆盖 | 全部合格候选池的采样覆盖 | 连续外包盒证书 |
|---|---:|---:|---|
| A1-f | 49.2645% | 32,768 / 32,768 | 64 / 64 顶点有精确非负反力解 |
| B | 27.8992% | 32,768 / 32,768 | 64 / 64 顶点有精确非负反力解 |
| C5 | 4.5135% | 32,768 / 32,768 | 64 / 64 顶点有精确非负反力解 |

原已选组合中的每块法向竖直分量均同号，因此对这些特定组合，上述逐点限制与逐块非向下合力限制的可用反力集合相同。一般含混合正负法向的接触区没有这个等价性。

连续盒覆盖当前全部工作网格位置、完整 30° 球冠（包含被遮挡方向）与 `0–0.5mg` 力度区间，盒边界向外保留 `1e-9` 的缩放裕量。64 个顶点的反力使用原始位置、法向和质心的精确有理数重新求解，逐一检查系数非负及六个平衡方程。

这是候选池接触能力的充分证据。尚未证明这些接触块可以同时摆放、不会重叠、满足逐块落地力矩或可以插入，也没有给出新约束下的贪心块数。证书用到的不同候选区域数不是最小支撑块数。

历史对照数值保留在上表；对应的 `passive_capacity.json` 已不在当前输出中，不作为本轮验证依据。

```sh
# 历史容量实验入口已删除；当前模型使用整体不上抬约束，见 ../baseline_algo.md。
```

## 4. 可用于论文方法部分的英文段落

> We model each support as an independent, massless, unanchored rigid body resting on a unilateral ground contact. In addition to workpiece wrench equilibrium and nonnegative contact reactions, we require each support to exert a nonnegative net vertical force on the workpiece. This condition follows from the support's vertical equilibrium: its resultant vertical contact force equals the nonnegative ground normal reaction. A load is counted as covered only if the same reaction vector satisfies all these constraints. Candidate supports are ranked by the marginal fraction of covered task loads when evaluated jointly with the previously selected supports. Our objective is coverage of a prescribed, gravity-inclusive load domain rather than force closure over the entire wrench space. Since the process-force magnitude is bounded by half the workpiece weight, the required resultant vertical reaction remains positive. The vertical condition is necessary but does not replace subsequent per-support wrench balance, ground-friction, and insertion checks.

该段是修订方法的写法。待 Step 3.1、Step 3.3 和连续验收完成同步实现并重跑后，才能把新方法的贪心结果写进实验结论。
