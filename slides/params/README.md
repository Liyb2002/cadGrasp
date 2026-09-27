# Four design parameter groups

## 一件共享支撑，两个姿态示例

2026-09-26 更新：第 3、4 栏使用重新设计后的 `slides/reuse/data.js` 中完整 `fixture` 网格，替换旧四臂示意。第 1、2 栏和全部公式保留原图内容。目录只输出一张含公式的图。

2026-09-23 用户确认：第 1、2、4 栏统一为两个 pose，按上下两行对应。图中 Pose 1、Pose 2 来自当前 [reuse](../reuse/README.md) 的前两个任务，即 B/pose_2、B/pose_3；这是展示编号，不改变数据目录中的 pose ID。第 3 栏只有一件共享实体 V。

- [唯一的 overview](optimization_overview.png)：四组参数、示意图和约束公式。

![What we solve for](optimization_overview.png)

Parameters 指设计对象，不是算法超参数。顶部用通用任务索引 k，任务总数不限定为两个；下方仅用两个 pose 举例，并标注 1、2。设计参数为：

\[
\left(\{A_{\rm obj}^{k},A_{\rm floor}^{k},d_k\}_{k=1}^{N},V\right),\qquad N=\text{任务数}.
\]

## 四栏

1. **Object contacts — A_obj^k**：每个 pose 各自选出的工件接触区域。两行显示该任务的工件姿态和接触片；力／力矩需求计算方法与公式不变。
2. **Ground contacts — A_floor^k**：每个任务要选择的支撑实际接地区域。两行分别显示该任务自己的工件、地面需求散点与有限样本包络；包络不等于实际地面材料。地面需求计算方法和力矩公式不变，不对两个任务的点求共同凸包。
3. **Shared support geometry — V**：一件连通刚性实体，在不同摆放下实现所有输入任务选出的接触区域，图中展示两个任务。不再分接触模块和独立底座。
4. **Insertion directions — d_k**：每个 pose 固定摆好的同一件 V，物体分别沿 d_1、d_2 水平装入。浅色物体表示装入前，实色物体表示落座，箭头指向装入方向。没有初次装蓝块的 d_0 或 dock。

## 第 3 栏：实现接触，且没有体积冲突

第 3、4 栏公式统一使用支撑自身坐标系，不加波浪号。T_k 表示支撑在第 k 个任务的摆放；该任务世界坐标中的接触区域和工件实体经 T_k^{-1} 变换后，直接记为 A_obj^k、A_floor^k 和 Obj_k。装入方向也用同一坐标系表达。图中世界位置可改变，但 V 的自身几何不变。

\[
A_{\rm obj}^{k}\subseteq\partial V,\qquad
A_{\rm floor}^{k}\subseteq\partial V,\qquad \forall k.
\]

“Cover”指 V 的表面实现选定接触区域，连接材料将它们组成一件实体，不是把点包在实体内部。工件自身与地面的接触不要求 V 再覆盖。需求散点不是必须填满的材料，A_floor^k 指根据需求选出的支撑实际接地区域。

每个任务终态中，完整 V 与物体不能发生体积穿透，允许表面贴合：

\[
\operatorname{int}(V)\cap\operatorname{int}(\mathrm{Obj}_k)=\varnothing,\qquad \forall k.
\]

全部臂和连接材料都参与检查。本轮不预设零冗余；各 pose 分别求解、对齐合并、再优化，可以作为 baseline，但合并后的避碰、装入和承载仍须验证。

## 第 4 栏：物体沿方向装入固定支撑

支撑先摆好并保持不动，移动的是物体。令 d_k 为指向落座终点的单位方向，l_k 为有限行程；以落座物体 Obj_k 为基准：

\[
\operatorname{Sweep}(\mathrm{Obj}_k,d_k)
=\{x-t d_k:x\in\mathrm{Obj}_k,\;0\le t\le l_k\}.
\]

整个行程须满足：

\[
\operatorname{Sweep}(\mathrm{Obj}_k,d_k)\cap\operatorname{int}(V)
=\varnothing,\qquad \forall k.
\]

允许表面贴合，不允许进入支撑内部；实际检查还须避免穿过地面。当前最后一段在世界坐标中水平平移，物体朝向与高度不变。退开状态为 Obj_k-l_k d_k，箭头与真实插入方向一致。两个方向不受旧 dock 接口轴约束。

图采用 reuse 侧向装入示意，行程 90 mm。显示方向与起终点不表示已经验证连续无碰撞；第 3 栏终态约束不能代替 sweep 约束。

## 力学与数据来源

第 1、2 栏沿用 [合并公式图](../combined_equations.png) 的三条参考力学公式与符号（[生成代码](../tools/combined_equations.py)、[原公式说明](../setup/equations/equations.md)），对每个任务、每个允许载荷分别应用。

\[
\int_{\rm supp\_obj}\; \mathbf{F}_{\rm supp}\,dA=mg\,\hat z-\mathbf{F}_{\rm push}.
\]

\[
\int_{\rm supp\_obj}\; r_{\rm supp}\times\mathbf{F}_{\rm supp}\,dA=-r_{\rm push}\times\mathbf{F}_{\rm push}.
\]

\[
\int_{\rm sys\_floor}\; F_{\rm supp}\,r_{\rm supp}\times\hat z\,dA=-r_{\rm push}\times\mathbf{F}_{\rm push}.
\]

supp_obj 包括工件原有地面接触；sys_floor 包括整个系统的地面接触。力臂从工件质心量起，参考式沿用支撑无自重假设。工件的力与力矩由同一反力场满足；地面式使用标量法向压力，不代表完整摩擦承载认证。

- 第 1、2 栏使用已确认图中的两个面板，保存在 `approved_panels.npz`，并保留原始来源记录。原先的独立单 pose 接触文件已清理，重画不恢复旧实验，也不以新的头组替换已确认图示。缓存中的任务输入哈希须与当前 setup 一致。
- 第 1 栏原本显示各任务独立 baseline 的 `step3_scheculer/final_contacts.npz` 中实际裁切三角片。这些是历史选择，不表示新 V 已实现这些区域。
- 第 2 栏原本读取各自 `step4_floor_contact/floor_contact.npz`，将散点与质心一起平移，并用原载荷经 `pressure_centers` 重算核对。图中仍保留当时每任务 32,769 个需求点的抽样与全部凸包顶点；不将它解释为当前 baseline 的新结果。
- 第 3、4 栏直接读取当前 `reuse/data.js` 的完整 `fixture`，不再读取旧 `spine/arms`。第 4 栏使用同一文件中的工件、工作区及 `fixtureR/fixtureT`，核对其任务旋转和工作面与 `objects/B/tasks/pose_k/setup.npz` 一致；对共享网格只作刚体变换，插入方向取支撑局部 x 轴在世界中的方向。
- 图是设计要求示意。reuse 的几何及有限静力检查见其自己的记录；本次重画不新增认证。JSON 区分本次实际输入哈希和前两栏的历史来源，记录新的支撑摆放与方向。

## 生成

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/params/draw_overview.py
```

只生成 `optimization_overview.png` 和配套 `optimization_overview.json`；运行成功后删除旧精简版 `overview.png`（若存在）。`approved_panels.npz` 是前两栏的重画缓存，不是另一张导出图；有缓存时，即使删除最终 PNG 和 JSON，也能重新生成。首次建立缓存可读取已有图及其记录，或在旧独立 baseline 输入齐全时重画前两栏。运行只读 reuse 和任务输入，不执行它们的生成器或搜索。
