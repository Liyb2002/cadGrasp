# Subdivide task：把一大块 area 分成容易支撑的 task

2026-10-24 从 `whole_task_2/` 复制而来，自成一体，不 import `whole_task_2/`。Step3–5 沿用原算法；改动只有三处：

- 新的 Step1（`step1/subdivide.py`）。
- Step4 combine 按 Step1 的座位组抽样。
- run 名到物体的映射（`step1/common.object_of`）。

Step5 在 2026-10-28 换成 `whole_task_2/step5` 的版本（B+03_kuka 成品所用：`combined.py`、`peel.py`、`frame.py`、`render_joined.py`、`transparent.py`、`video/raster.py`，只把包名换成 `subdivide_task`）：大接触面从外向内削、修边后重建成圆滑的面，地脚是每张桌面规划一次的扁平锥。原来 10-24 那版削面的性能问题，在 whole_task_2 的这一版里已经用另一种方式修好（删除工具先挖掉要保留的棱柱，删除后只留最大块）。下面"结果"一节的数字来自旧版 Step5。

## 输入

- **big task**：`objects/B/big_tasks/<name>.json` 中的 `face_ids`。本次跑 big_08、big_14、big_20（面积都超过 60%，三块，两只耳朵全覆盖）和 whole_shape（整个物体）。
- **工位**：`robots_config/random/07_kuka.json`。KUKA LBR Med 14，75 mm 工作立方体，针向下 45°，允许偏转 20°。底座、工作窗口与 02_kuka 相同。
- 工具沿用 Step1：针长 100 mm、针半径 2 mm、头半径 20 mm。

## Step1：怎么分

一个 task 是物体的一个摆放（旋转 + 平移）加上在这个摆放下加工的面。每个面都用 Step1 原有的精确检查认证：针、头、KUKA 逆解与连杆、桌面，针尖要在工作立方体内。KUKA 随物体升降，所以高度和水平位置一样可以自由选择。

1. **座位组**。一个组对应物体的一个朝上方向 u。组内 task 都保持 u 朝上，只差绕竖直轴的转角和平移。这样在物体坐标里，它们的重力方向相同，针都从上方来，所有载荷都压向物体下部，一个从下面托住的支撑就能承受全部。每组只加工托住区（沿 u 最低的 25%）再往上 4% 以外的面，所以组内任何 task 都不会切到托住区。组数取能覆盖 area 的最少朝上方向：在 240 个采样方向上做集合覆盖。
2. **task**。各组的候选 = 转角（每 10°）× 立方体位置（网格步长为边长的 1/6，每个转角取面积最大、彼此重叠不多的 12 个）。用最少的候选覆盖 99% 的面积（部分集合覆盖：先贪心，再整数规划），逐个精确认证。某个面认证失败，就从该候选中去掉，重新求覆盖。覆盖不够时，补充瞄准未覆盖面的新候选。所有组的 task 一起竞争每个面，所以一个组难以够到的缝隙会交给别组的 task。
3. **分面**。用 Potts 图割互斥分配：每个面的代价是针的偏角加上它在该组中的高度（越低越贵），边界越短越好。没有 task 认证的少量面（不超过 1%）交给最近的 task，记为未认证。

`data/<run>/step1_reach/plan.json` 的 `subdivision` 字段，以及 `output/<run>/step1/subdivision.json`，记录分成了几个组、几个 task 和覆盖率。`step1.2/groups.png` 按组上色。

## Step4：按组合并

combine 的随机合并照旧，只是抽样会参考座位组：如果一个 task 所在的组在支撑中已经有座位，它被抽中的概率提高到 8 倍，并以 90% 的概率用 Rotational reuse 进入本组座位；所在组还没有座位的 task，以 20% 的概率尝试 Rotational reuse，否则用 Juxtaposition 新开一个座位。

## 运行

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
.venv/bin/python -B -m subdivide_task.run_big big_08          # Step1 -> Step3 -> Step4 keep-out -> combine -> Step5 -> evaluation
.venv/bin/python -B -m subdivide_task.run_big big_08 --start step4_combine
```

数据在 `data/<run>/`，图和 Step5 成品在 `output/<run>/`，模板与 `whole_task_2/output` 相同。

使用过程的视频在 `idea/demo/<run>.mp4`，每个 30 秒。代码 `idea/demo/code/video.py` 来自 `whole_task_2/idea/demo`：KUKA 按 task 加工，Franka 在 task 之间搬运。

```bash
.venv/bin/python -B subdivide_task/idea/demo/code/video.py --run big_08
```

这份代码相对原版改了两处：
- 视频按 run 名命名；
- 兔子中转用的姿态如果搬不通，就换下一个，最多试 6 个。

## 结果（2026-10-25）

四个 run 都从 Step1 跑通到 evaluation。每个 run 的 3 个不同支撑都通过了 Step5 全量验收（每个 task 32,768 组配对力和力矩、连续出口、KUKA 在每组存储关节下的检查、地脚包住压力凸包），并从磁盘重建复验通过。

| run | area | 组 | task | 认证覆盖 | Step4 座位（每座 task 数） | R / J | Step5 成品 | 包围盒 / 物体包围盒 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| big_08 | 63.9% | 2 | 12（A7 B5） | 99.26% | 3（4, 3, 5） | 9 / 2 | 251.4 cm³ | 2,643 cm³（1.13×） |
| big_14 | 67.5% | 2 | 13（A6 B7） | 99.09% | 2（6, 7） | 11 / 1 | 267.6 cm³ | 3,015 cm³（1.29×） |
| big_20 | 60.5% | 3 | 11（A5 B3 C3） | 99.00% | 3（5, 3, 3） | 8 / 2 | 273.4 cm³ | 2,768 cm³（1.19×） |
| whole_shape | 100% | 3 | 15（A7 B7 C1） | 99.31% | 3（7, 7, 1） | 12 / 2 | 317.3 cm³ | 3,478 cm³（1.49×） |

- **认证覆盖**：剩下不到 1% 的面交给最近的 task，记为未认证。
- **R / J**：最佳支撑的合并链中 Rotational reuse 与 Juxtaposition 的次数。
- **成品体积与包围盒**：取 r00（最佳）。
- **每个 run 都只用 1 个支撑朝向**：座位数等于组数，或只多 1 个。
- **Rotational reuse 的成功率**：搜索中所有尝试里，big_08 为 205/224，big_14 为 241/277，big_20 为 152/184，whole_shape 为 311/336。对照：`whole_task_2` 的 B+8pose 是 1/165；B+03_kuka、B+8pose、B+9pose 都是每个 task 一个座位，全部用 Juxtaposition，包围盒为物体的 1.32–2.28 倍。两边的工位和 task 不同，只能作定性对照。
- **用时**：Step1 1.5–25 分钟，Step3 2–5 分钟，keepout 1–2 分钟，combine 17–30 分钟，Step5 11–23 分钟（削面修复之后）。
