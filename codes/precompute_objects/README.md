# 对象数据集预计算

当前 21 个物体各保存 **30 个 pose、20 个互不重复的组合**：2、3、4、5、6 个 pose 各 4 组。每个 pose 固定工作面和 **32,768 个载荷**；算法读取这些输入，不重新生成姿态或撒点。

```bash
# 全部物体；生成、验证、发布成功后清理该物体的旧输入
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/run.py --jobs 4

# 单个物体
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/run.py B

# 已完成物体复用；失败物体继续生成
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/run.py --resume --jobs 4

# 全量独立复核与目录清理检查
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/verify.py

# 检查各算法的所有输入、样本复用、六姿态适配与旧输入拒绝
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/check_consumers.py baseline_algo
```

## 保存格式

```text
objects/<name>/
  mesh.stl                     # low-poly 网格，m
  meta.json                    # 几何、质量等物理元信息
  poses.json                   # 30 个姿态、参数、网格哈希
  sets.png                     # 20 个 set；每组并排展示其 pose
  pose_sets.json               # 20 个组合、完整有向兼容矩阵、各大小合法组合数量、文件哈希
  poses/
    pose_1/                    # 至 pose_30
      setup.npz                # T_world_mesh、COM、支点、工作面掩码、参数与输入哈希
      setup.json               # 工作面检查及细分次数
      needs.json               # 世界坐标几何与六维载荷定义
      samples.json             # 固定加工位置、加工力、六维需求及采样参数
      floor_contact.npz        # 原始载荷、地面需求、法向力、支点与力矩原点
```

`pose_sets.json` 的 `sets` 中，每项包含 `id`、`size`、`poses` 和 `passed`；`poses` 引用 `pose_<i>`。`directed_violating_counts[i][j]` 表示来源 pose i 的需求映射到目标 pose j 后穿地的样本数。一个组的所有有向计数均为零才兼容。组合是固定的预计算输入，随机种子只改变算法读取这些组合的顺序。

## 生成与验收

`search.py` 从完整三维旋转池生成接地姿态。每个 pose 保留唯一原始顶点接地、质心投影偏离支点至少 1 mm，姿态重力方向两两至少相差 18°。`work_regions.py` 固定连通、可见、朝上、离地的工作面，面积占比 6–10%。载荷保持 K=0.5、30° 半角、原固定种子；`loads.py` 是共享的采样与物理输入实现。

搜索先以 1,024 个载荷筛选兼容图，再对候选组合使用全部 32,768 个载荷复验。先寻找七姿态兼容子集以保证至少四个不同六姿态组合，再补齐 30 个彼此有足够差异的姿态。最终组合枚举基于全部载荷；各大小选四组，优先减少已使用姿态的重复。候选预算耗尽或组合不足即失败，不降低要求，不发布部分数据。

地面需求由六维反力与世界原点力矩计算。需求点按 `T_target @ inverse(T_source)` 随工件变换，目标高度须 ≥ −1e−9 m。地面无限延伸；凸包顶点只用于搜索加速，最终独立复核检查全部样本。通过只表示固定配准、无自重支撑模型下的采样地面必要条件，后续接触、完整实体、装卸、强度和机器人运动另行求解。

发布前在临时目录完成数据生成、原始加工力的独立力矩核对、姿态差异与组合验收，再替换该物体的数据。失败保留原输入。旧 `tasks/`、轨迹、视频、overview、历史输入及搜索缓存在成功发布后删除。根目录的 `cases.json`、`index.json` 是有效清单；`_simulation_assets/` 保留机器人代码仍使用的碰撞几何。

## 算法衔接与已有结果

baseline 与 DSL 的 Step1、registry、task reader 和 Step0 是适配层。Stage1 可复制数据集文件以兼容既有阶段布局，内容逐字节一致；不同计数／种子及不一致的既有输出会被拒绝。Step0 从保存的 20 个组合选取需要的组，必要时发布阶段诊断图，不重复姿态搜索或载荷采样。

旧 baseline/DSL 支撑结果仍保留在原输出目录，属于此前姿态版本。新旧 `pose_1` 等编号代表不同姿态，输入哈希检查会拒绝旧载荷；新实验需要重新初始化输出。本次没有重跑支撑设计。

本次执行结果见 `batch_report.json`；独立复核见 `verification.json`，共 630 个 pose、420 个组合、20,643,840 个载荷、18,900 个有向 pose 对。生成数据遵循仓库现有策略，仅保存在本地，不加入 Git。

只重绘各物体的组合总览：`OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/draw_sets.py --jobs 4`。也可在命令后指定 `B` 等物体。每个物体只生成一张 `sets.png`，20 个组合按大小分组，所有 pose 使用同一视角与比例，橙色表示保存的工作面。

## Low-poly 与 Step2

`lowpoly.py` 使用保留拓扑的二次误差边折叠，再局部等形细分过大的平面三角形，所有物体及工作面网格最多 5,000 面。原本较粗的网格保留形状。闭合性、连通分量、Euler 数、体积误差与双向采样距离均检查；不把采样距离称为严格 Hausdorff 上界。结果见 `lowpoly_report.json`。更换网格后需重建全部姿态、载荷、组合与图。

Step1 的需求与固定载荷由 `loads.py` 生成。Step2 的计算实现位于 `heads/`、`head_geometry.py`、`head_directions.py`；baseline/DSL 仅保留适配层。运行 `run.py` 会继续生成 Step2；`--skip-heads` 可先只生成姿态与载荷。单独生成或续跑：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/precompute_heads.py --workers 20
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/verify_heads.py
```

每个 `poses/pose_<i>/step2/` 保存：

- `geometry.npz`：200 个确定性候选中心、0.5%／1%／2% 面积各 200 个头的接触三角形、原始面编号、面积、相对 COM 的六维力／力矩生成元，以及有限连接路径图的节点与连通分量。
- `entries.json.gz`：实际拟合尺寸、局部头碰撞／地面检查、路径见证、完整有限方向菜单，以及每个方向的连续全程扫掠检查、通过／阻挡／未决记录。
- `candidates.png`：三个面积的候选中心和通过检查的前八个头，两个视角；橙色工作面、绿色通过、红色未通过。这里只展示单头结果，不代表所示头一起满足载荷或共有退出方向。
- `report.json`：输入与计算源代码哈希、配置、统计和以上文件哈希。输入或实现改变会拒绝旧缓存，续跑会重建。

Baseline 原生单 pose 池读取水平菜单对应的子集；DSL 初始化读取完整水平及向下菜单，并复用原始力／力矩生成元。变换头到另一 pose 时去掉原坐标生成元并重新计算；不能直接沿用旧 COM 的力矩。组合相关的工作面／地面排除、共同退出方向、共同连接路径、联合反力 LP、整套支撑体装卸仍在运行时求解。离散菜单未找到退出只表示没有该菜单的见证，未宣称不存在连续方向解。

所有当前数据集图均随 low-poly 和 Step2 更新。已有算法输出是旧姿态的历史结果，不把新图片覆盖到旧支撑结果上。`_simulation_assets/` 属于机器人流程保留的原始碰撞资产，与这里新的简化网格分开记录。

环境依赖：`requirements.txt`。退出／可达体计算中的约束 Delaunay 三角剖分需要 Shapely ≥ 2.1。

全程扫掠的凸多面体裁剪有 Numba 加速版本；与原 Python 几何判定使用同一正面积和边界容差规则，随机凸体及接触边界对照测试检查两者一致。未安装 Numba 时退回原实现。另有全局支撑面的分离平面捷径：只有整个工件和完整连续扫掠被同一平面分开才通过，内凹／内部面仍做完整碰撞计算。

只重绘候选头图而不搜索 pose、不拟合接触面、不重新检查退出：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/draw_heads.py --workers 4
# 单 pose 示例
OPENBLAS_NUM_THREADS=1 .venv/bin/python codes/precompute_objects/draw_heads.py --object B --pose pose_1
```

极细的等面积分割三角形可能让原 `closest_point` 算式产生非有限值；撒点现在确定性回退到同一叶三角形的重心。`finite_center_migration.json` 记录了修复前后逐点重放；仅中心和原始面编号完全相同的缓存复用旧几何与退出证明，发生变化的缓存重新计算。

本次全量 Step2 验收见 `heads_verification.json`：21 个物体、630 个 pose、378,000 个候选头，187,320 个候选头通过各自 pose 下的单头检查；21 张 `sets.png` 和 630 张 `candidates.png` 的哈希、格式及输入一致性通过。原始 pose／组合／20,643,840 个载荷的独立复核仍见 `verification.json`。这里的单头通过不等于联合载荷或完整支撑体通过。
