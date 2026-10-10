# 三个角度的逐 pose 需求输入（2026-10-10）

每个 `objects/<name>/poses/pose_<i>/` 只保存三套角度输入：

| 加工力圆锥半角 | 目录 | 配对力／力矩样本数 |
| --- | --- | --- |
| 15° | `angle_15/` | 32,768 |
| 30° | `angle_30/` | 32,768 |
| 60° | `angle_60/` | 32,768 |

不再按 grounded／airborne 分需求。同一朝向下，工件、质心和载荷作用点一起平移时，绕质心的需求不变；是否着地改变的是**可用地面反力**。地面接触应按优化后的实际摆放判断，不由这些需求文件决定。

每个角度目录包含 `needs.json`（连续需求域）、`setup.npz`（原生参考位姿，无接地状态字段）、`samples.npz`（原有配对载荷与完整采样参数）、`sample_metadata.json`（单位／种子／分布）和 `variant.json`（来源及指纹）。`need_wrench` 顺序是 `[Fx,Fy,Fz,tau_x,tau_y,tau_z]`，力矩绕工件质心，单位为 `mg` 与 `mg*m`。加工力大小仍为 `[0,0.5mg]`，重力保留；力和力矩共同产生，不独立采样。参考几何沿用原生 pose；它的参考高度不意味着优化时必须接地。

每套仍采用原面积／立体角／均匀力大小采样及物体自遮挡拒绝，固定种子20260907。**这次整理直接复用已有三个角度的全部数组，不重新采样。**全开角分别30°、60°、120°。30°是当前工况参数，不是平衡方程推导出的常数；不同角度是不同连续域及有限样本，不假定样本逐行嵌套。

角度目录不再保存固定高度的世界原点需求、地面压力中心或工件地面接触标记。Step5应按最终世界摆放重新计算系统—地面的力／力矩，必要时计入支撑自身重力；运行时的原点转换工具为 `wrench_at_world_origin`。物理推导见 [物体不着地](../../slides/obj_supp/airborne_equations.md)。

原根级 `setup.npz`、`setup.json`、`needs.json`、`samples.json`、`floor_contact.npz` 逐字节保留；后三项已从旧目录的符号链接恢复为独立文件。根级 `floor_contact.npz` 是原生参考输入，保留供已有 reader／pose-set 分类使用，不能据此给抬高后的工件添加地面反力。Step2、原生姿态与已有集合保留。旧算法仍读取原30°默认输入；旧兼容分类不构成其他角度或新位置的验收。

索引仍为逐 pose 的 `load_variants.json` 和总的 `objects/load_variants.json`，格式升级为 v2，只有三个角度，没有状态维度。旧六个数值目录删除；仅旧代码和元数据留档，避免另存重复样本。

在仓库根目录运行：

```sh
# 将已有六套存档合并；已完成项校核后复用，无重新采样
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m codes.precompute_objects.collapse_load_variants --jobs 4
# 重放全部六维平衡、角度边界、来源数组指纹及原始文件哈希
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m codes.precompute_objects.verify_load_variants --jobs 4
# 以后为没有角度存档的新原生数据生成三套输入；已有项校核后复用
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m codes.precompute_objects.load_variants --jobs 4 --resume --run-name load_angles_v2
```

生成和整理支持 `--objects B`、`--poses 1 2`。不再提供 airborne 高度或 grounded 参数。整理记录位于 `data/load_angles_collapse_20261010/`，全量结果和校核分别为 `batch.json`、`verification.json`。此前六状态批次记录属于历史，不能作为当前目录清单。

```python
from codes.precompute_objects.load_variants import read_variant
case = read_variant('B', 'pose_1', half_angle_deg=60)
wrenches = case['arrays']['need_wrench']  # (32768, 6)，绕质心
assert case['ground_state_independent']
# 地面接触是否可用由实际布局决定，而不是需求文件决定。
```

# 当前 pose set 三分类（2026-10-06）

每个 objects/<name>/ 保存三个互斥的集合 JSON：

| 文件 | 含义 | 21 个对象合计 |
| --- | --- | --- |
| pose_sets.json | 合法且存在非零共同方向 | 217 |
| no_common_direction_pose_sets.json | 合法且不存在非零共同方向 | 455 |
| illegal_pose_sets.json | 地面载荷不兼容，不论有无共同方向 | 30 |

合法指原固定 32768 个载荷的组内地面兼容矩阵无违反，不表示完整夹具通过。第二类合并原有 203 组与新增 252 组。所有 702 组互斥、无丢失、无重复，法向/共同方向见证及反例证书已复核。清单在 objects/pose_set_categories.json，复核在 objects/pose_set_category_verification.json。

shared reader `dataset.read_pose_groups(name, category='legal')` 读取两类合法集合（每对象 32 组）；category 也支持三类名称、illegal、all。`dataset.read_sets(name)` 保留原每对象 20 组的顺序与成员，供已有算法/图片的兼容读取；include_supplemental=True 包含全部 32 个合法组。当前 Co-optimize 单组 solver 读取全部三类，批次和历史渲染读取原集合，避免把未构造的新组误认成已有输出。

姿态、几何和载荷未改变，但集合文件哈希因重组改变；原集合 JSON 归档在 codes/precompute_objects/data/pose_sets_before_categories_20261006。重组不构成旧设计的重新求解或验收。

以下记录生成数据集及此前审计的格式和历史统计；原先 pose_sets.json 的“20 组”语义现在由 reader 兼容层提供。

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

## 共同退出方向检查与补充反例

运行 `.venv/bin/python codes/precompute_objects/common_directions.py`，检查所有物体现有的 `pose_sets.json` 和 `illegal_pose_sets.json`，并写入每个物体的 `common_direction_audit.json`、`no_common_direction_pose_sets.json`。共同方向仅指所有 pose 的原生地面半球约束交集，不表示扫掠、承载或完整夹具可行。严格内部、仅边界和不存在非零方向分别记录；不能把 LP 得到零向量当作共同方向。

2026-10-06：21 个物体 / 630 个 pose。正式 420 个集合中 217 有严格共同方向、203 无非零共同方向；另 30 个历史 illegal 集合中 25 有、5 无。未出现仅边界的现有集合。按正式集合大小：2 pose 84/84 有共同方向，3 pose 84/84，4 pose 40/84，5 pose 8/84，6 pose 1/84。

每个物体新增 12 个反例：4、5、6 pose 各 4 组，总计 252 组。全部与原有集合不同，且保存的全部原始载荷地面兼容矩阵中，组内有向违反数均为零。反例附带正权重法向零和、法向矩阵满秩证书：如果所有法向与方向点积非负，正加权和又为零，则每个点积只能为零，满秩迫使方向为零，从而不存在非零共同方向。

补充文件不改变原有 pose、载荷、20 个正式集合或其哈希。根目录总表：`objects/common_direction_audit.json`；独立复核记录：`objects/common_direction_verification.json`（630 个 setup 变换逐一匹配，702 个方向见证/反例证书通过）。补充集合需要显式读取，现有算法不自动把它们算成新的正式验收集合。
