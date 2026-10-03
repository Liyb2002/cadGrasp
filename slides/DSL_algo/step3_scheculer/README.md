# Active operation DSL

[Algorithm and acceptance](operation_dsl.md). Run `run_operation_batch.py --jobs 3` to optimize all B sets, audit and publish Step4. Results: 81 → 72 physical heads, 10/10 complete fixtures. The copied latest baseline constructor is in `../step4_connect_support/baseline_current/`.

## Historical stage documentation

# Active: feasible-incumbent DSL (V5)

The active algorithm is [feasible_dsl.md](feasible_dsl.md), run with `step3_scheculer/run_feasible_dsl.py`. Each pose retains its independent heads and exit path. Optional head sharing and soft path alignment are accepted only with a complete feasible fixture witness; rejected proposals preserve the incumbent. The V4 descriptions below are historical.

# Active joint shared-head experiment (V4)

The active implementation is described in [the experiment README](../README.md).
Step3 uses one canonical physical-head program for the whole group, global
head deletes/merges, all-pose GPU numerical descent and joint exit-space selection.
Active results are in `dsl_shared/` and `dsl_shared_support/`; old `dsl/` results
optimized pose heads independently and must not be called shared-head results.
Step3 retains the heavy full-construction witness contract. Construction uses
one in-memory geometry acceptance; Step4 copies the same mesh.

---

The following notes describe earlier copied/independent experiments.

> Active V3: `run_dsl.py` accepts a case only after force checks, baseline root-path connectivity and a full exported construction witness pass. Contact-only proposals are not Step3 successes. See [the experiment guide](../README.md). The text below is copied baseline history.

2026-09-29 当前修改：**各 pose 仍独立选头、独立受力和退出，但增加整组地面余量筛选。** 头的构造输入按零厚度接触面表示；其全部有限三角面顶点变换到组内每个 pose 后，最低地面高度必须至少为 **2 mm**（数值容差 `1e-10 m`，恰好 2 mm 通过）。平面高度是仿射函数，因此检查全部三角形顶点等价于检查完整面，不使用中心点替代。只筛选原来的 200 个候选，不重定位、裁小或扩大头，不添加跨 pose 的受力、退出或工作面限制。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_independent.py B \
  --poses pose_3 pose_6 --floor-poses pose_3 pose_6 --floor-clearance-mm 2 \
  --output-root slides/baseline_algo/output/B/pose3+6/step3_scheculer/independent_poses_floor2mm \
  --jobs 1
```

每组独立结果根目录必须显式指定，避免同一个 pose 在不同组错误复用未筛选的旧结果。新 schema 为 `independent_single_pose_floor_margin_v1`；主报告及候选表都保存整组 pose、原始输入哈希、2 mm 条件和每个头的逐 pose 最低高度。最终验收重新计算选中面。保留全部 32,768 原始载荷、1% 接触面积、200 候选、3–4 个头及最多十条链。

本地退出和细路径暂时继续用原有限厚度头作**保守探测**；零厚度直接送入旧实体凸包会退化。`normal_depth_m` / `local_geometry_probe_depth_m` 记录这个探测体厚度，**不再要求 Step4 保留这个旧实体体积**；`head_model=zero_thickness_contact_surface` / `construction_thickness_m=0` 明确新构造输入。Step4 从这些完整接触面造实际有体积的连接支架，并重新检查全部真实材料。2 mm 面筛选只修复头部地面余量，不能保证任意连接体、退出或受力通过。组模式不重建用户已删除的单 pose `heads.png`。

以下为被当前修改替代的独立求解与固定配准记录。

2026-09-29 历史独立入口：`run_independent.py B --jobs 2` 默认覆盖 B 的全部 20 个 pose；没有 `--floor-poses` 的旧模式不施加整组地面筛选，结果位于 `output/B/independent_poses/pose_<i>/`。

独立结果只认证本 pose 的头和受力，不认证 Step4 连接实体。Step4 需要确定物体与共享支撑在不同 pose 下的相对摆放，然后检查闲置材料的真实位置。不能把闲置头通过物体坐标变换重新贴到物体上，也不能拿下面旧固定配准结果判断新方案。旧 Step0 的跨 pose 地面检查不是本次独立求解的前置条件。

全部 20 个 B pose 已跑完：18 个达到 32,768/32,768；pose2 为 31,533/32,768，pose20 为 32,551/32,768，两者用完十条链，保留四头的最好结果。[完整表](../output/B/independent_poses/report.md)。这不是 Step4 完整实体的通过结果；独立头的后续连接入口见 [Step4](../step4_connect_support/README.md)。

以下为已被替代的固定配准实验记录。

2026-09-29 历史入口：先运行 `../run_sequential_batch.py B --n 3` 的 Step0，随机试不同组合直到地面需求全部合法。Step0 通过后才发布 Step1 并准备 Step2 候选；该入口仍保留用于复现旧实验。

2026-09-29 退出检查修正：每增加一个物理头，检查它在**所有 pose** 下的完整直线退出，并与此前保留下来的方向集合求交；包含闲置头和还未开始求解的 pose。任意 pose 的方向集合为空就拒绝候选，不进入覆盖评分。切换 pose、选择共享头时也继承方向集合，只有新搜索链才重新初始化。结束时独立重放全部已选头的整组退出，把实际保留的方向交给 Step4；连接体、地脚仍在 Step4 构造并检查。方向目录依旧是现有有限水平方向，不能把搜索失败称为任意运动下无解。

同一批从 Step3 重跑：`../run_sequential_batch.py B --from-step 3 --seed 20260929 --jobs 2`。复用保存的 Step0–2 输入和精确候选曲面；旧 Step2 未保存的方向／路径证书在内存中重算，不生成新候选、不修改候选文件。只替换现有八组的 Step3/4 输出，不创建历史归档。当前共享 Step0 缓存已删除，因此输入引用经哈希核对后指向各组原有 Step1；本轮汇总放最后一组的 `step4/data/batch_report.md`。

本轮八组已完成：**pose3+6 用 6 个头全覆盖，并在 Step4 构造连接实体（258.579 cm³；后续 Step4 完整退出通过，但联合载荷验收失败）**；其余七组在十条链内未全覆盖。全部选中头组在所有 pose 的退出均通过独立重放；265 次加头的方向继承记录已复核，36 项相关测试通过。[八组结果和图片](../output/B/pose2+9+13+15+17/step4/data/batch_report.md)。

# Step3：逐 pose 求解，选一个最佳已有头，再补当前 pose

最新用户决定（2026-09-28）：baseline 使用 [run_sequential_k.py](run_sequential_k.py)。先求第一个 pose；后续 pose 从此前**所有已选物理头**中，选择对当前 pose 单头覆盖增量最大的合法头，再添加当前 pose 自己的头。每个 pose 优先用 3 个头，未全覆盖则补第 4 个；共享头计入后续 pose 的 3–4 个总数。每个 pose 满足全部 32,768 个原始样本后才进入下一个，先前头组和几何保持不变。每头固定面积 1%，不改尺寸或末尾扩展。每组最多十条独立 top5 搜索链，第一条全任务通过即停止。

最新几何修正：**每个头都必须避开所有输入 pose 的工作面和地面，不论它在哪些 pose 参与受力。** 候选生成先扣除工作面并集，并将接触曲面裁到所有地面以上 1.5 mm；固定 1% 面积拟合后，再用原始来源头实体检查所有地面。不能在某个 pose 把头设为 inactive 来豁免这些条件。最终 `geometry.all_pose_head_check` 重查每个选中头与每个 pose，记录工作面交集、接触面最低高度和实体最低高度。各有效组的插入／基本路径、覆盖评分不变；不要求每个头都贡献每个 pose，完整支架装卸与落脚可行性仍由 Step5 处理。

新结果使用 `sequential_k_global/from_<顺序>/`，原 `sequential_k/` 保存为历史。当前重跑命令如下，只处理仍保留的四组新结果，不恢复已删除的组合，也不覆盖旧 pose1+3 做图结果：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cadgrasp-mpl-head-previews \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/run_sequential_batch.py B --existing-groups --workers 2
```

当前评分只看正在求解的 pose 的原始载荷覆盖增量。每轮按增量取 top5，并按增量归一化抽样；零增量均匀抽样。共享头选择比较单头与原工件地面支点的增量，并按 ID 打破并列；不只比较紧邻前一 pose 的头。一个已有头可以被多个后续 pose 依次选中，但每个新 pose 只继承一个旧头。

此前批量入口随机选六个组合，组合种子 `20260928` 得到：`8+10`、`6+7`、`6+9+10`、`1+9+10`、`4+5+6+8`、`5+7+8+9`。每组按数字升序求解，每任务 200 个候选；不根据结果更换抽中的组合。原始输出在各阶段的 `sequential_k/from_<顺序>/`。下述六组数据尚未使用所有 pose 的头部排除规则；新版本不复用这些候选和成功状态。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/run_sequential_batch.py B --seed 20260928
```

Step5 已支持 N 组接触的唯一物理头配准、所有有序地面对检查、最近合法地面身体和多 pose 展示。前置条件失败会保留诊断和明确失败状态，不输出虚构实体；默认仍关闭最终完整受力／退出审计。顺序搜索、多 pose 共享实体与原回归共 49 项测试通过。用户保存的 `slides/co_design_algo/` 副本只读，未修改。

### 全 pose 头部排除后的四组重跑（2026-09-28）

四组均已处理到 Step5 入口，批量运行约 451 秒。所有已选头都通过全部 pose 的工作面／地面检查；3/4 组完成 Step3 全覆盖，0/4 组生成完整支架。

| pose 组合 | 不同头数 | 各 pose 有效头数 | 搜索链数 | Step3 | Step5 |
| --- | ---: | --- | ---: | --- | --- |
| 1+9+10 | 8 | 3、4、3 | 1 | 全覆盖 | 固定配准落脚条件失败 |
| 4+5+6+8 | 10 | 3、3、4、3 | 2 | 全覆盖 | 固定配准落脚条件失败 |
| 5+7+8+9 | 11 | 3、3、4、4 | 10 | pose9 为 32379/32768，其余全覆盖 | 接触输入未完成 |
| 6+9+10 | 9 | 4、4、3 | 4 | 全覆盖 | 固定配准落脚条件失败 |

三组完整接触方案的地面冲突数量与旧运行完全一致：固定摆放下，脚位限制和整件地面需求并不随头的选择改变。新增检查修复了头本身占用其他工作面／穿地的问题，但不代表脚和连接结构也能实现。第四组只是在十链、每 pose 至多四头的搜索预算内未成功，不能据此证明不存在其他头组合。

86 项相关回归通过。独立复查了 135 个头／pose 组合、458752 个保存覆盖标记和 24 个原始地面对条件，旧头图保留，co-design 副本的 9764 个文件未改变。当前[批量结果](../output/B/pose6+9+10/step4/data/batch_summary.json)、[总图](../output/B/pose6+9+10/step4/data/batch.png)、[独立复查](../output/B/pose6+9+10/step4/data/global_head_review.json)在最后一组的 Step5 data 内。每组外层的 `heads.png` 和 `head_details.png` 已更新为新选头。

### 历史六组结果：仅检查有效任务的头（2026-09-28）

六组均已处理到 Step5 入口。**5/6 组完成全部 pose 的 Step3 覆盖；0/6 组生成完整实体。** 通过的每个 pose 都是原始 `32768/32768`；Step5 的五次拒绝均来自固定共享头配准下的地面需求冲突。本批没有进入身体增长，也未运行最终完整实体审计，不能把这些接触解称为完整支撑。

| 依次求解的 pose | 不同头数 | 各 pose 有效头数 | 搜索链数 | Step3 | Step5 |
| --- | ---: | --- | ---: | --- | --- |
| 8 → 10 | 6 | 3、4 | 10 | 未全覆盖：32768、23446 | 接触输入未完成 |
| 6 → 7 | 6 | 4、3 | 5 | 全覆盖 | 固定配准地面冲突 |
| 6 → 9 → 10 | 9 | 4、4、3 | 1 | 全覆盖 | 固定配准地面冲突 |
| 1 → 9 → 10 | 8 | 3、4、3 | 1 | 全覆盖 | 固定配准地面冲突 |
| 4 → 5 → 6 → 8 | 9 | 3、3、3、3 | 2 | 全覆盖 | 固定配准地面冲突 |
| 5 → 7 → 8 → 9 | 12 | 4、3、4、4 | 1 | 全覆盖 | 固定配准地面冲突 |

[六组图](../output/B/pose5+7+8+9/step4/data/batch.png)仅显示选中的接触与失败状态，没有身体或脚。[批量记录](../output/B/pose5+7+8+9/step4/data/batch_summary.json)和同目录 `review_check.json` 区分每一阶段并绑定实际来源哈希。Step3 导出时已从零重算原始覆盖 mask，作独立 LP 抽查；另外重放所有链的共享头选择、top5 随机数、固定接触几何和 Step5 有序地面对证据。

## 历史：所有 pose 同时选头

2026-09-28 最新范围：用户已停止十任务搜索并要求删除其整个输出目录；十任务记录和下文提到的该组验证文件已一并删除，不再自动重跑。当前指定组为 **pose1+3+4+6**，其旧六头预算试跑尚未通过：覆盖数为 **4,782 / 3,154 / 32,768 / 32,768**。该组 Step5 已运行构造前置检查，固定共享接触配准后的地面条件也未通过，未生成实体；见该组 `step4/data/report.json`。旧配对 Step5 图保持不动。

2026-09-28 当前入口为 [run_joint.py](run_joint.py)。取消先为第一个 pose 选三头、再强制共享一个并补两个的顺序；从第一轮同时计算所有输入 pose。每个 pose 的头数、总头数和共享数量由搜索产生。旧入口和结果保留为历史对照，下面的 3+2、逐轮尺寸优化、末尾扩展规则不适用于当前入口。

不指定 pose 时，默认读取 `objects/<object>/tasks.json` 的全部任务共同搜索；`--all-poses` 显式表达同一行为。`--poses` 可指定一个任务子集，只有显式使用 `--pairs` 才运行抽样双任务实验。B 当前注册了 pose_1～pose_10，因此全任务运行是一次十任务搜索，不是五次双任务搜索。

当前已选状态记为 S，任务 k 的原始载荷覆盖率为 c_k；加入候选 h 后的新增覆盖比例为 delta_k，分母仍是该任务的全部 32,768 个原始样本。评分为：

```text
value(h | S) = mean_k((1 - c_k) * delta_k)
```

例如覆盖 80% 的任务新增 10 个百分点，贡献为 `0.2 * 0.1 = 0.02`；覆盖 20% 的任务增加相同覆盖，贡献为 `0.8 * 0.1 = 0.08`。未覆盖比例作乘法权重，不作分母。它是即时启发式，柔性照顾落后任务；不保证优先推进每一个瓶颈、最少头数或未来成功。

- 每任务生成 200 个固定为工件总面积 1% 的候选，合并为公共池。每个候选保留来源任务的真实接触曲面与头实体；跨 pose 只作刚性变换，不重新挤出另一实体。
- 每轮分别检查候选加入各任务现有头组后的局部几何、共同水平退出方向和基本路径。将该头加入所有兼容任务；不兼容的任务保持原头组。允许候选只贡献一个任务，不要求它在所有任务均为合法有效接触。已经选中的头和任务分配保持不变；这个“全部兼容任务启用”的规则尚不枚举其他任务子集。
- 每任务对加入后的整组接触重新求反力，沿用原地面接触与共享不上抬约束。delta_k 是整组覆盖的增量，不能相加各头独立覆盖。所有任务每个原始样本通过才成功，不增加纯重力门槛或连续验证。
- 默认十条独立链，每轮按新的 value 取 top5，并以同一 value 归一化抽样；全部为零时均匀抽样。并列按候选 ID 排序。每轮权重更新，已全覆盖任务权重为零。
- 默认四个独立进程分担候选评分，主进程统一排序和抽样，`--workers 1` 可串行运行。只增加接触时，旧反力见证可令新头反力为零而保留，因此使用原分类器的 `known_covered` 接口复用已覆盖样本；已经全覆盖的任务不重复求候选 LP。导出时仍用全部原始载荷从零重算覆盖 mask 并作独立 LP 抽查。
- 完成空集合的全候选评估后，缓存实际候选及各 pose 的单头方向／路径分量集合，后续组合法性仍重新求交；缓存绑定输入、几何代码和候选文件哈希。首轮力学评分另外绑定评分代码哈希，各链可复用同一空状态评分，但使用自己的随机数。任何依赖变化会使对应缓存失效。
- 只增加头：不删除、不替换、不改尺寸，末尾也不扩展。没有每 pose 或总头数的固定目标。不能重复选择已有中心，因此有限候选耗尽会终止；`--max-heads` 可设置总计算预算，预算停止不代表无解。
- Step3 只输出有效接触集合。闲置实体冲突、共享支撑摆放、身体、地脚和完整装卸保留到 Step5。Step4 沿用原载荷的地面需求计算。现有 Step5 仍读取历史 3+2 数据，不能把新结果直接当作完整实体或已适配的新 Step5 输入。

```sh
# 全部已注册 pose 一起搜索；B 为十任务，无头数上限
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_joint.py B --all-poses

# 两个 pose 同时搜索；默认十条链，每任务 200 个候选，无固定头数
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_joint.py B --poses pose_1 pose_3

# 相同入口支持四个 pose 子集，同样不限制头数
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_joint.py B --poses pose_1 pose_3 pose_4 pose_6
```

结果位于 `output/B/pose1+3/step3_scheculer/joint_weighted/` 等目录；四任务为 `pose1+3+4+6/`，数字排序，输入顺序不指定搜索先后。各链记录逐轮所有候选的各 pose 增量、未覆盖权重、value、top5 概率、随机数、激活任务和几何筛选原因，以及最终接触 NPZ 与原始样本 mask。导出接触后重算全部 mask 并作独立 LP 抽查。顶层 `schedule.json` 记录实际参数、各任务头 ID、候选来源与头深度；`complete` 仅表示运行完成，设计是否成功看 `result.passed`。Step4 生成 `joint_result.json` 和 `joint_overview.png`，不修改旧实验或 Step5。

### 2026-09-28 实现验证

加入原始样本的精确剩余候选池失败证明和 `--resume` 后，相关测试合计 **63 项通过**。断点恢复绑定输入、候选、参数和代码，重放随机数并复核已保存前缀；失败证明只针对冻结的当前前缀，不表示其他选择也无解。

加入覆盖证明复用与准备缓存后共 **58 项相关测试通过**，包括已完成任务免重算、缓存曲面的刚性变换与方向／路径求交、输入／候选／几何代码变化时拒绝缓存、力学代码变化时禁用旧评分。另对正式十任务运行保留的前两轮，重放全部 4,000 个候选的十任务几何判断，结果一致；八个第二轮候选的四进程评分与原串行十任务覆盖数和 value 一致。保留的原运行记录及代码哈希位于十任务 Step3 的 `joint_weighted/history/before_coverage_reuse/`。

正式加速运行还逐项复现了前两轮全部 4,000 条候选记录的各 pose 覆盖数、value、top5 概率与抽样结果。对应文件哈希与核对结论保存在十任务 Step4 的 `joint_weighted/coverage_reuse_equivalence.json`；测试日志为同目录 `implementation_tests.log`。

全任务入口更新后共 **54 项相关测试通过**。新增测试确认默认入口一次传入全部十个任务、没有头数上限，只有显式 `--pairs` 才抽样配对；另以必须选十个不同头才能覆盖的十任务案例确认选头不会在三头或五头处停止。`progress.json` 逐轮保存当前链、头数与所有任务覆盖数；多任务结果图自动换行。

同日代码复核后共 **50 项测试通过**，补充五任务真实 LP／共享不上抬、三任务完整运行与导出、重跑失败状态回归。修复了同目录重跑在初始化时失败仍可能留下旧 `complete=true` 报告的问题：CLI 在加载输入前将本次 Step3/4 报告标为未完成、旧检查标为未复核；只有本次完成后才发布完成报告，旧数据文件不作为新运行的通过证据。候选文件改在运行开始后导出，单独创建求解器不再替换既有候选文件。

对下表保存结果另行重放全部 22 轮的评分、top5 概率与随机选择，重新计算 262,144 条最终载荷分类并作独立 LP 抽查；逐接触核对来源曲面的刚性变换、原半径和只增前缀，全部一致。记录为各组 Step4 的 `joint_weighted/review_check.json`，绑定原 `schedule.json` 哈希。本次没有重新搜索；保存结果继续保留状态管理修复前的原代码哈希，不能改写成新搜索结果。共享头实体的跨任务扫掠逻辑未在这次复核中改变，完整支撑仍不属于 Step3 验证范围。

47 项针对性测试通过，包含五任务动态头数／共享分配、剩余比例乘法权重、top5 排名与概率、权重更新、零增益互补、有限池终止、固定尺寸与新增前缀、原输入复用，以及旧 sequential/pair/terminal-expansion 回归。

用原始 32,768 载荷完成两次小规模端到端试跑，种子均为 `20260926`；这是接口与规则验证，不是默认 200 候选／十链的成功率实验：

| 输入 pose | 每任务候选数 | 链数 | 总头数预算 | 最佳链覆盖数 | 结果 |
|---|---:|---:|---:|---|---|
| 1、6 | 24 | 2 | 8 | 9,434 / 32,768；32,768 / 32,768 | 两链均预算耗尽 |
| 1、3、4、6 | 12 | 1 | 6 | 4,782；3,154；32,768；32,768（各自总数 32,768） | 预算耗尽 |

两次均已重算导出接触的完整样本 mask、独立 LP 抽查和 Step4 方程；另重放全部 22 个选头轮次的 value／概率、检查只增前缀、输入原字节与来源／输出哈希。记录在各组 `step0_pose_selection/joint_weighted/joint_check.json`，图为同目录 `joint_overview.png`。检查通过不等于设计通过，本次真实数据试跑尚无所有 pose 全覆盖的组合。

## 历史：顺序式及双任务共同头组实验

2026-09-26 目录整理：配对现在直接位于 `output/B/pose1+3/`、`pose1+4/`、`pose1+6/`、`pose2+8/`、`pose6+9/`，下一层为阶段目录。Step1 输入分别位于每对的 `step_1_needs/pose_<i>/`；顺序式结果为 `step3_scheculer/sequential_3plus2/from_<first_pose>/terminal_expansion/`。配对目录按数字排序，`from_*` 保留先求哪个 pose 的区别。全部既有实验已迁移并更新引用，没有重跑搜索或改变载荷。

<a id="sequential-3plus2"></a>

## 历史实验：先 pose1 三头，再共享一个、为 pose2 补两个

2026-09-26 用户指定直接 hardcode 顺序算法，入口为 [run_sequential.py](run_sequential.py)。先运行十条独立 pose1 particle，各用 top5 增量加权采样选三个头；在每条可继续的三头链内，选出对 pose2 单独覆盖增量最大的几何合法头，再为 pose2 用同样的 top5 规则增加两个新头。零增量 top5 均匀抽样，共享头并列按 ID 决定。每个完整组合恰好五个不同中心，两套接触三元组只共享一个头。

- 每个任务各生成 200 个固定 1% 候选，按本任务工作面、地面和几何筛选；不先排除另一任务的工作面。各自的三个有效接触保留共同水平退出方向及基本路径连通性。
- 共享头从 pose1 的实际几何刚性变换到 pose2，检查其工作面、地面净空、实际头部实体扫掠及路径。头部实体沿用来源任务的厚度和外形，不按另一任务重新挤出不同形状；受力只使用同一块接触曲面。
- pose1 评分只使用其三头，pose2 评分只使用共享头和两个新头。每个被计入覆盖的原始样本仍需满足六维平衡与共享不上抬；原工件—地面接触保留，单独纯重力不作为额外门槛。
- 沿用固定 1% 选头和末尾补全：五头布局完成后，仅当两边都严格超过 98% 时尝试最多到 1.10% 的扩展；共享头改变时，两套接触同步改变并重查。pose1 三头若不超过 98%，提前失败；禁用补全时必须在 pose1 已全覆盖才转移。
- 十条 pose1 particle 各自至多接一条 pose2 延续，不额外展开成一百条。最终必须两套三头各自覆盖全部原始 32,768 载荷，才记为接触组合成功。同时报告成功链数和去重后的五头组合数。
- 这次只验证指定的有效接触集合。闲置的另两个部位、完整共享刚体的摆放、地脚和整件装卸仍待 Step5；没有把“不出力”直接标记为“可接地”。因此成功数不能与原来“五头在两边都接触并共同退出”的实验当作同约束消融。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_sequential.py B --pairs 5 --seed 20260926 --particles 10 --workers 2
```

`--poses pose_6 pose_1` 可指定搜索顺序；`--no-expansion` 关闭末尾补全。输出在各 pair 既有阶段的 `sequential_3plus2/from_<先求的pose>/terminal_expansion/` 或 `fixed_only/`。逐链保存两个阶段的候选评分、抽样随机数、共享头比较、实际三头接触和覆盖 mask。Step4 保存两套地面需求与 `sequential_overview.png`；图只展示该任务实际采用的三头，同色表示同一头，图例标记共享头。

### B：同五对任务、每对十条 particle 的结果

种子 `20260926`，每任务 200 个固定 1% 候选，顺序为表中箭头方向。**5/5 对均找到接触解，50 条 particle 共 11 条通过，各对内去重后也是 11 个不同五头组合。** 其中 10 组固定 1% 已全覆盖，1 组在末尾小幅扩大后补齐；没有重新抽取任务或更换种子。

| 顺序 | 第一任务三头直接全覆盖 | 转入第二任务的链 | 最终通过／不同组合数 | 其中补全成功 | 图 |
|---|---:|---:|---:|---:|---|
| 1 → 6 | 5/10 | 5 | 2/10 | 1 | [图](../output/B/pose1+6/step0_pose_selection/sequential_3plus2/from_pose_1/terminal_expansion/sequential_overview.png) |
| 1 → 4 | 5/10 | 5 | 3/10 | 0 | [图](../output/B/pose1+4/step0_pose_selection/sequential_3plus2/from_pose_1/terminal_expansion/sequential_overview.png) |
| 1 → 3 | 5/10 | 5 | 3/10 | 0 | [图](../output/B/pose1+3/step0_pose_selection/sequential_3plus2/from_pose_1/terminal_expansion/sequential_overview.png) |
| 6 → 9 | 3/10 | 8 | 2/10 | 0 | [图](../output/B/pose6+9/step0_pose_selection/sequential_3plus2/from_pose_6/terminal_expansion/sequential_overview.png) |
| 2 → 8 | 1/10 | 9 | 1/10 | 0 | [图](../output/B/pose2+8/step0_pose_selection/sequential_3plus2/from_pose_2/terminal_expansion/sequential_overview.png) |

第一任务未全覆盖但严格超过 98% 的链也可继续，最终仍须补齐两边全部载荷；因此“转入第二任务”可能多于第一任务直接成功数。此次 11 个最终成功组合的第一任务在选三头后均已全覆盖。唯一补全成功的是 `1 → 6` 的 particle 009：第二任务由 32,753/32,768 补至全覆盖。其余失败结果保持失败。

可直接用于后续 Step5 的一个例子是 `1 → 4` 的 particle 003：pose1 使用 `pose_1_C041、pose_1_C163、pose_1_C088`；pose4 使用共享的 `pose_1_C088` 及 `pose_4_C106、pose_4_C156`。各自仅三头，两边均为 32,768/32,768，无末尾扩展。其实际接触数据已导出为该 pair 的 `final_contacts_pose_1.npz` 和 `final_contacts_pose_4.npz`。

新增 9 项测试通过，覆盖逐任务三头评分、最高贡献共享头、top5 和零增益回退、两新头追加、方向交集、共享尺寸同步、同一实际共享实体的扫掠及顺序输出隔离。运行中对全部 50 条链从导出接触重算完整 mask、执行独立 LP 抽查，并核对共享接触几何、实际面积和 Step4 地面方程。各对的 `sequential_check.json` 均通过，本次没有数值重试。结果表示接触组合可行，尚未认证五个部位连接后的同一刚体和实际地脚。

下方为此前的双任务共同五头算法及结果。

<a id="two-pose-baseline"></a>

## 当前规则：每轮固定 1%，超过 98% 才尝试终止补全

2026-09-26 最新决定：Step3 每轮只选头，**每个头固定为工件总表面积的 1%**。选头停止后，如果尚未完成且**两姿态各自严格超过 98%**，才尝试小幅扩大现有头来补齐剩余样本；任一姿态不超过 98% 则直接失败。已全覆盖的链保持原样。不以覆盖／面积比评分，也不在完成后缩面积。接触面积最小化留到后续 structural geometry 阶段，未增加共享实体构造、候选上界诊断、换头策略或 value network。

2026-09-26 用户最终确定：每个 pose 固定使用 Step1 的 32,768 个采样载荷，全部通过即通过；不运行连续载荷域验证，不搜索或追加反例。每个样本仍包含重力，并须满足六维平衡和共享合力不上抬条件。单独的零加工力重力检查只作诊断，不另设通过门槛。插入和基本连通性要求保留。此规则覆盖此前关于连续证明的要求。

当前入口为 [run_pairs.py](run_pairs.py)，五头预算对照入口为 [head_budget.py](head_budget.py)。同一组工件表面接触片随工件变换到两个任务，各自求反力；不搜索共享 V 的独立摆放或接触角色切换，止于 Step4。

- 200 个共同候选中心；排除两个工作面集合和两个地面净空带。在真实曲面上一次拟合 1% 面积，相对拟合容差 `1e-4`；无法达到目标面积或目标尺寸下几何不合法则淘汰，不缩小头来挽救。
- 每对 10 条独立 sampling，每轮从合法 top5 按正的覆盖增量加权抽样；全部增量为零时均匀抽样。
- 贡献为两个 pose 覆盖增量的等权平均。只改善一个 pose 也可以选择；中间状态不要求已经完成另一任务。
- 每轮新增一个候选，所有已选头的中心、半径和实际三角片保持不变。面积只记录，不参与评分或最终链排序；双任务完成优先，其次平均覆盖，并列按链编号。
- 选头结束后才调用 [terminal_expansion.py](terminal_expansion.py)，不会在中间轮次因超过 98% 而提前调尺寸。严格门槛用每姿态整数样本数判断，不用平均覆盖或显示时四舍五入的百分比。
- 补全保持中心和头数不变，目标面积依次为工件总面积的 1.01%、1.02%、1.05%、1.10%，上限为原 1% 面积增加 10%。每一级轮流扩大合法头，优先采用实际覆盖增量最大的提案；允许暂时零增益的扩大，以便多个头共同产生贡献，但不允许丢失已覆盖的任何样本。不缩小、不新增头；此有限启发式不保证发现范围内所有可行尺寸组合。
- 每次扩大重新检查真实接触片、头部扫掠、两任务各自的共同水平插入方向和基本路径连通性，并评估全部原始载荷。全覆盖即停止；达到扩展上限仍有失败样本，状态为 `terminal_expansion_exhausted`，保留失败结果。
- 每个被计入覆盖的样本同时满足力、力矩平衡及 `sum(head_force_on_workpiece_z) >= 0`；原始物体—地面接触力不计入该求和。
- 每次加头检查两任务各自的共同水平插入方向，并使用固定头实体的完整相对平移扫掠证据；三维路线图要求全部头能接入共同连通分量。加头后的固定样本覆盖不得减少。
- 每个 pose 的 32,768 个样本全部通过后立即成功停止。不会再调用连续验证、增加载荷或因额外工况改判。Step4 也只计算这 32,768 个载荷对应的地面散点。

基本 path 检查不认证连接杆厚度、完整连接实体的插入或整体 V 的承载。加工射线体积仍按既有模型关闭。默认最多 3 个头；可显式提高预算。

```sh
# 随机一对，默认三头预算
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_pairs.py B

# 相同五对，五头预算；当前结果放在 fixed_area_1pct/terminal_expansion/heads_5/
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/head_budget.py B --pairs 5 --seed 20260926 --particles 10 --max-heads 5 --workers 2
```

数值 LP 报错时，[retry_pairs.py](retry_pairs.py) 在同一原方程上重试并记录证据，不放宽容差；五头入口已使用此机制。数值重试不添加载荷，不执行连续验证。

## B：终止补全结果

在同五对 B、五头上限、每对十条已保存的固定 1% 链上执行终止补全，**不重新运行 sampling**。先复现补全前的完整覆盖 mask 和接触数据，再执行新步骤；共 4 条未完成链满足两姿态各自超过 98% 的门槛。

| Pose 对 | 尝试补全链数 | 最佳链补全后覆盖率 | 未覆盖数：补全前 → 后 | 通过链数 | 图 |
|---|---:|---|---|---:|---|
| 1 + 6 | 0 | 100% / 100% | 0 / 0 → 0 / 0 | 1/10 | [图](../output/B/pose1+6/step0_pose_selection/fixed_area_1pct/terminal_expansion/heads_5/pair_overview.png) |
| 1 + 4 | 1 | 100% / 99.9969% | 0 / 1 → 0 / 1 | 0/10 | [图](../output/B/pose1+4/step0_pose_selection/fixed_area_1pct/terminal_expansion/heads_5/pair_overview.png) |
| 1 + 3 | 0 | 14.8987% / 100% | 27,886 / 0 → 27,886 / 0 | 0/10 | [图](../output/B/pose1+3/step0_pose_selection/fixed_area_1pct/terminal_expansion/heads_5/pair_overview.png) |
| 6 + 9 | 3 | 100% / 99.6155% | 0 / 131 → 0 / 126 | 0/10 | [图](../output/B/pose6+9/step0_pose_selection/fixed_area_1pct/terminal_expansion/heads_5/pair_overview.png) |
| 2 + 8 | 0 | 88.0768% / 100% | 3,907 / 0 → 3,907 / 0 | 0/10 | [图](../output/B/pose2+8/step0_pose_selection/fixed_area_1pct/terminal_expansion/heads_5/pair_overview.png) |

总结果仍为 **1/5 对、1/50 条链通过**，本次小幅补全没有增加成功链。(1,4) 接受了 15 次逐级更新，部分提案被共同方向／路径条件或法向包角拒绝，最后仍差 1 个载荷。(6,9) 最佳链五个头都扩大到约 1.10%，补上 5 个载荷；其余两条尝试链分别补上 2 个和 12 个，均未完成。接近 100% 的样本比例不保证小幅扩大可以覆盖剩余载荷；这些结果只说明本次有限扩展未找到全覆盖解。

34 项针对性测试中 33 项通过，1 项因历史 B/pose_2 单 pose 候选数据缺失而跳过。全部 50 条链的原 mask 复现、最终导出 mask 重算、逐姿态门槛、中心和头数不变、扩展上限、几何检查、独立 LP 抽查及 Step4 方程核对通过；各对记录见 `terminal_expansion_check.json`。本次补全运行未触发数值重试。

重放保存链的补全：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/run_terminal_expansion.py B --max-heads 5
```

## 历史对照：B 固定 1%，尚未加入终止补全

按种子 20260926 的同五对任务、每对十条独立 top5 sampling、五头上限重跑；仍只检查每 pose 的原始 32,768 个样本。新结果位于各阶段 `fixed_area_1pct/heads_5/`。候选面积要求也收紧为固定 1%，因此与下表旧版的比较同时包含取消尺寸搜索及排除不足 1% 候选，不能解释成只改变尺寸目标的消融。

| Pose 对 | 最佳链覆盖率 | 未覆盖样本数 | 通过链数 | 图 |
|---|---|---|---:|---|
| 1 + 6 | 100% / 100% | 0 / 0 | 1/10 | [图](../output/B/pose1+6/step0_pose_selection/fixed_area_1pct/heads_5/pair_overview.png) |
| 1 + 4 | 100% / 99.9969% | 0 / 1 | 0/10 | [图](../output/B/pose1+4/step0_pose_selection/fixed_area_1pct/heads_5/pair_overview.png) |
| 1 + 3 | 14.8987% / 100% | 27,886 / 0 | 0/10 | [图](../output/B/pose1+3/step0_pose_selection/fixed_area_1pct/heads_5/pair_overview.png) |
| 6 + 9 | 100% / 99.6002% | 0 / 131 | 0/10 | [图](../output/B/pose6+9/step0_pose_selection/fixed_area_1pct/heads_5/pair_overview.png) |
| 2 + 8 | 88.0768% / 100% | 3,907 / 0 | 0/10 | [图](../output/B/pose2+8/step0_pose_selection/fixed_area_1pct/heads_5/pair_overview.png) |

固定面积版 **1/5 对、1/50 条链通过**；旧尺寸优化五头版为 2/5 对、3/50 条链。五对最终导出的最佳链均为五头；(1,4) 仍有一个固定样本失败，不能按四舍五入后的 100% 判通过。候选合格数依次为 101、94、104、106、108；失败结果不表示其他接触组合无解。

本次测试 27 项中 26 项通过，1 项因历史 B/pose_2 单 pose 候选数据缺失而跳过。对全部 50 条链，从保存的实际接触重新计算全部固定样本 mask，核对逐轮尺寸不变、覆盖非减、实际面积、独立 LP 抽查及 Step4 点集；记录见每对 Step4 的 `fixed_area_check.json`。生成的图使用本次新结果。搜索期间 (2,8) 有一次原方程 LP 数值重试，未放宽容差、未添加载荷，证据保存在该 pair 的 `numerical_retries/`。

## 历史：逐轮调尺寸版本按固定采样规则更新后的结果

保留种子 20260926 抽取的五对及每对 10 条链。已对保存的三头／五头最终头组重新计算全部原始样本，与保存的原始 mask 逐项一致，并复核独立 LP。此次只更新验收，不重新运行选头搜索。

| Pose 对 | 三头上限覆盖率 | 五头上限覆盖率 | 五头上限通过链数 | 当前结论 | 图 |
|---|---|---|---:|---|---|
| 1 + 6 | 35.14% / 100% | 100% / 100% | 1/10 | 通过 | [图](../output/B/pose1+6/step0_pose_selection/heads_5/pair_overview.png) |
| 1 + 4 | 61.16% / 98.66% | 100% / 98.41% | 0/10 | 尚未全覆盖 | [图](../output/B/pose1+4/step0_pose_selection/heads_5/pair_overview.png) |
| 1 + 3 | 11.42% / 83.81% | 13.53% / 99.99% | 0/10 | 尚未全覆盖 | [图](../output/B/pose1+3/step0_pose_selection/heads_5/pair_overview.png) |
| 6 + 9 | 100% / 97.96% | 100% / 100% | 2/10 | 通过 | [图](../output/B/pose6+9/step0_pose_selection/heads_5/pair_overview.png) |
| 2 + 8 | 99.91% / 87.28% | 89.68% / 100% | 0/10 | 尚未全覆盖 | [图](../output/B/pose2+8/step0_pose_selection/heads_5/pair_overview.png) |

三头预算仍为 0/5 对通过；五头预算为 **2/5 对通过，共 3/50 条链**。(1,6) 的五头组合按当前规则通过，过去的连续域反例不参与当前判定。(6,9) 的两条通过链分别使用 4 个和 5 个头；当前按面积选择五头方案，不以头数最少为目标。表中的未通过只表示本次保存头组未覆盖全部固定样本。

## 输出与历史记录

当前输出为 `output/B/pose<i>+<j>/<stage>/fixed_area_1pct/terminal_expansion/`，显式五头预算再使用 `heads_5/` 子目录。新运行 `schedule.json` 使用 `shared_object_attached_heads_two_pose_terminal_expansion_v5`，通过状态为 `both_samples_passed`。记录 `during_selection_area_optimization_performed=false`、终止补全策略、尝试链数及补全链数；`area_optimization_performed` 仅表示是否尝试终止尺寸调整。选头各轮仍保存固定半径和实际面积，每条链另存 `terminal_expansion.json` 记录全部提案、几何拒绝原因和覆盖。Step4 为每任务恰好 32,768 个地面需求点。

固定面积原搜索的核验保存在其 Step4 的 `fixed_area_check.json`。终止补全重放另存 `terminal_expansion_check.json`：复现原链全部 mask、检查逐姿态门槛、固定中心和头数、扩展面积上限，从导出接触重算全部最终 mask，并执行独立 LP 抽查及 Step4 方程核对。核验文件的 `passed` 表示数据与规则核对通过，设计是否覆盖两任务仍看 `schedule.json`。

绘制本次结果：

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/draw_pair.py B pose_1 pose_6 --terminal-expansion --max-heads 5
```

旧尺寸优化输出仍保留在各 pair 根目录及原 `heads_5/` 子目录，未加补全的固定面积版在 `fixed_area_1pct/heads_5/`。v3、v4、v5 schema 都可读。

旧连续验证搜索的固定采样重新判定读取 `sample_result.json`，各粒子为 `sample_schedule.json`，固定 mask 为 `sample_coverage.npz`。这些文件只适用于对应历史搜索；v4 和当前 v5 直接读取各自的 `schedule.json` 和 `coverage.npz`。历史 Step4 重新验收记录为 `sample_result_check.json`。

可用 [relabel_pairs.py](relabel_pairs.py) 重放此验收更新：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python \
  slides/baseline_algo/step3_scheculer/relabel_pairs.py B
```

下面是旧单 pose / SMC 的历史说明，其中连续验证和通过条件已被本页当前规则取代，不作为当前双 pose 的运行指令。

## 2026-09-22：默认流程移除完整搜索审计

按用户确认，`run_all.py` 不再执行 Step3 的 `audit.py`，不重放所有分支、候选排名、抽样与各轮覆盖。搜索中的受力、几何和连续载荷域验收保持不变；搜索后的 `verification.py` 保留最终载荷验证，并只对最终输出构造实际接触头与有厚度连接，检查一条共同方向的完整扫掠。该实体检查失败会使运行报错，不把构造条件通过当作实体已通过。结果写入 `verification/scheduled_contacts_verification.json`。

`--resume` 通过输入／代码／输出文件哈希和完成状态复用搜索，不再依赖 `audit.json`；复用后仍执行最终验证和结果绘图。完整审计脚本保留为手动调试工具，旧审计文件仅为历史记录，Step3 批量结果不再用它表示本次验收。此改动不调整 Step2、Step4 或 Step6 的检查。历史耗时中的完整审计费用不再属于当前默认 Step3 流程，新的耗时需另行实测。


## 2026-09-22：SMC-inspired 粒子搜索

新增 `--search-mode smc --trajectories 10 --search-seed 0`。这是启发式 population optimization，不是对某个后验分布的严格 SMC，也没有全局最优保证。默认 `top5` 及原 `greedy` 保留以便比较；SMC 不改变载荷、接触力模型、几何约束或连续域完成条件。

一个粒子保存**完整的部分接触组合**：中心、优化后的半径、实际三角面、共同退出方向、同方向的有限厚度连接证据，以及该分支发现的连续域反例。每层分配 10 个扩展名额，每个名额只新增一块接触，复用原评分、整组尺寸优化及验证。父节点文件不可变，子节点直接引用祖先轮次；不会从第一轮重复计算父链。最多 3 层、30 次新增接触优化，实际数量与各种评价次数写入报告。

同一父状态的全候选评分只算一次，兄弟分支复用它，但使用独立随机种子选候选。兄弟已尝试的候选从提案中排除，再从**剩余合法候选的 top5** 按边际覆盖增益抽样，零增益统一回退；因此首层可以探索 10 个不同接触，而不是在最初 5 个中重复。它与旧独立 top5 基线不只相差重采样，也增加了这一显式去重探索规则。若可用候选少于名额，跳过重复评价并记录原因，实际独特粒子数可能小于 10，不虚构新样本。

完成尺寸优化后，用所有粒子共享的**原始载荷集合的绝对联合覆盖数**排序，同覆盖时面积小者优先。各分支反例仍约束本分支的求解和验收，但不用于跨粒子的覆盖排名；也不使用新增覆盖量作为重采样权重。按这个排序赋予温和的线性权重：第一层最好／最差权重比最多 2，第二层最多 4。相同覆盖与面积得到相同权重。每轮保留一个最佳粒子扩展名额，10 名额中另有 2 个均匀探索名额，其余按权重抽样；副本在下一层选不同候选。

连续域已经验证通过的解进入 archive，后续不再添加接触。它们始终参与最终选择。失败几何、候选耗尽、连续验证未决的分支停止；未决沿用旧停止策略，表示未取得证书而非物理不可行。最后一层不再重采样。最终优先选择 archive 中面积最小的解；若无解，则在全部实际评价过的节点中保留原始覆盖最高、面积较小的部分结果。若该最佳结果是历史中间节点，顶层状态为 `population_search_exhausted`，原节点状态保留在 `selected_particle_status`。

顶层 `schedule.json` 保存 `particles`（父子谱系、独立种子、兄弟排除候选、状态与耗时）、`layers`（权重、有效样本量、各名额来源）、`verified_archive`、`round_trajectories`、`evaluation_counts`。评价数量区分实际全候选评分轮数、候选组合分类数、新增接触优化次数、尺寸覆盖评价数和连续验证次数；不把这些计数冒充底层 LP 或几何测试次数。`population_progress.json` 保存运行进度。全部文件仍位于各 case 的既有 Step3 阶段目录。

2026-09-22 数值修复：首轮真实 B/pose_2 批跑暴露了既有连接检查对近乎平行初始地面方向的除零型病态：变换后点积约 `5.4e-17` 被当成真实上升斜率，将 2.5 mm 的关节抬升需求放大成 `4.65e13 m` 的后平面，随后实体凸包构造失败。现对单位方向与地面法向的点积使用 `64 × machine_epsilon × max(1, sum(abs(u_i n_i)))` 的保守误差保护，代入斜率下界。所有外挤距离非负，所以此修复只收紧地面不等式，不把微小下行放宽为水平，不加最大结构尺寸，也不使用 Qhull 扰动掩盖问题。该旧三头组合现在在候选／尺寸检查时直接无连接证据；一个已通过的真实两头组合仍能构造正体积连接。Step1–2 不依赖此源码，缓存保持有效；全部十例 Step3 按相同搜索预算统一重跑，修复前中间数值不混入最终成功率。新增 2 项回归覆盖实测旋转法向／切向方向和保守下界。

可选的手动审计重放候选提案、随机数、重采样、谱系、完成 archive 和最终选择，并对每个独特实际轮次审计一次物理与几何证据。继承轮次仅在已审计后复用，避免按所有叶子重复审计共同前缀。`--resume` 只复用完整、参数和哈希一致的 population；中断的 SMC 搜索会重新开始，不声称支持中途恢复。B/pose_1～10 当时仅运行到 Step3，并执行了独立审计；这些历史结果不代表 Step5/6 或机器人抓持已通过。当前默认验收见页首更新。

```sh
/Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step3_scheculer/run_all.py B \
  --pose pose_1 --from-step 3 --through-step 3 --search-mode smc \
  --trajectories 10 --search-seed 0 --no-round-drawings --primal-first-retry
```

SMC 新增 7 项测试覆盖统一载荷比较、覆盖优先与面积排序、温和权重与探索名额、随机重放、兄弟去重、只前进一步、谱系路径隔离、已完成解保留和末层不再重采样。Step3 调度器目录共 108 项测试通过；真实案例结果须另看逐例审计。


2026-09-21 当前运动模型更新：蓝块在 `objects/<object>/poses.json:rest` 的初始躺姿安装。方向仍以任务坐标存储，但地面条件使用 `T_initial_from_task` 拉回的平面；接触三角面须离初始地面至少 1.5 mm。取消用加工工作面外侧半球限制初次安装方向的旧启发式。连接实体须同时满足初始和任务姿态的静态地面约束，初次装入扫掠只针对初始场景。Step5 的整体 docking 方向独立，不能复用 d0 作为固定底座的运动方向。受力和力矩仍在任务姿态验证。


## 2026-09-21：单 pose 的连接与插入联合筛选

新增 [`connection.py`](connection.py)。候选加入前，先检查所有实际头是否能沿一个共同退出方向伸出有厚度的连接颈，再在越过物体的后平面上用梁树连为一体；连接实体必须在地板上方。根部严格位于头内部，梁与关节有实体重叠，不把零宽路径或点接触算作连接。工作面不能被选作接触；加工射线体积仍按既有策略默认关闭。

对每个已认证头部方向，预计算物体投影、真实头几何和有限厚度连接的后平面位置区间。组合时取下界最大值、上界最小值；无可行方向的候选以 `no_common_connection_witness` 淘汰，不参与受力评分。缓存按实际几何签名区分，尺寸变化必须重新检查，不能继承旧连接证书。缩小头只能继承满足嵌套条件的头部扫掠证书；连接方向仍需重新求交。旧 greedy 尺寸搜索也只接受通过该检查的尺寸。

这是一种**充分构造检查**，不是任意三维绕行路径的完备判定。失败仅表示当前尺寸、有限方向表及该连接形式中未找到证据，可能淘汰其他结构可以连接的组合。首次计算需要构造几何与投影，不能统称 instant；冻结的 B/pose_2 两头几何测得首次约 30 ms、缓存后约 0.052 ms（100 次中位数），仅为该案例的局部检查耗时，不是完整搜索耗时。

原 `common_withdrawal_directions` 保留纯头部方向交集；`common_connected_directions` 是同时允许连接的子集。`connection` 保存构造参数和实际几何签名，`head_connection_witness_verified` 必须为真才能通过 Step3。当前默认最终验证重新构造最终头组的连接，并检查实际头部与连接实体的退出扫掠；完整搜索历史审计只按需手动执行。Step5/6 只使用联合通过的方向，但实际底座与完整支撑仍须分别验证，早期连接证据不能替代最终结构认证。

本次相关测试通过：调度器目录 98 项、尺寸优化目录 35 项（包括新增连接检查与完成门槛测试）。这些测试不替代每个案例的独立审计。

当前仅逐个 pose 独立求解。连接检查器现在同时接入初始地面和当前任务地面；其他任务姿态的地板仍未接入多 pose 联合优化。连通性、有限厚度、地板反例、尺寸变化、评分前淘汰及后续方向限制见 `test_connection.py`。


默认 `--search-mode top5 --trajectories 10 --search-seed 0`（总入口 `run_all.py`；单独运行 `scheduler.py` 使用 `--seed`）。每条链最多新增 3 块接触，不替换、不重采样、不训练网络。`--search-mode greedy` 保留原单链、只调新块的基线，便于对照。

每轮沿用完整的 Step3.1 候选评分和几何／纯重力／共同退出方向筛选。从合法候选中取新增覆盖量最高的 5 个，按 `gain_i / sum(gain)` 抽样；全为零时在这 5 个中均匀抽样。数量不足 5 个就使用全部合法候选。排名并列按候选 ID，记录概率、随机数及每条链的独立种子。第一轮评分共享；后续各链独立评价，不把其他链的反例、尺寸或覆盖提示带入当前链。

抽中后，先调新块，再依次调全部旧块；固定中心、轮流调整半径，最多两遍（`--sizing-sweeps 2`）。未满覆盖时优化整组覆盖率／总面积；某次标量搜索找到满覆盖半径后，选择保持满覆盖的较小半径，后续坐标也必须保持满覆盖。每次重建该坐标的固定几何及覆盖缓存，按其他实际接触重新计算可用退出方向；缩小旧块可以释放方向。该方法是局部坐标优化，没有多维全局最优保证。

每轮整组面积搜索共享 `--sizing-budget 96` 次覆盖评价预算，按剩余坐标轮次分配；几何裁切、纯重力检查和导出后的独立验证另计耗时。这个预算不是整条搜索的 LP 次数，也不包含全候选评分。半径下界仍严格大于物体面积的 0.5%，共同退出及受力模型保持一致。

2026-09-18：每个坐标改用真实面积上的局部割线和固定代表载荷的平衡残差代理，最多四次完整覆盖评价；全部载荷和最终连续域验证仍决定是否接受。省去每次坐标更新的全局半径边界二分；合法嵌套缩小可继承头部方向证书，但须重新检查连接证据，每轮结束仍更新完整方向集合。详见 [面积优化](../step3.3_optimize_contact/README.md)。

共享的原始反力 LP 关闭 HiGHS presolve：这类问题方程少而反力列多，相关方程消除可能比实际求解更慢。方程、非负约束、容差、数值回退及原坐标残差验收保持原样。代表性冻结候选的完整 32,768 载荷分类已逐项对比，结果一致；全案例结果仍以重新运行的独立审计为准。

若可选的连续域对偶凸包因退化面合并长时间运行，可显式使用 `--hull-timeout 60`。它先尝试原始 primal 证书，再在独立进程计算凸包；单次超时记录为 `unresolved`，不能算作物理失败或成功。证书记录限时和各次结果。此选项可能错过耗时更长才能找到的反例／证书，默认不启用，且须完整重跑，不能与 `--resume` 混用。其余几何、样本与原始方程验证保持原样。

每条链的文件位于各既有 Step3 阶段目录的 `trajectory_000/round_001/` 等子目录。顶层 `step3_scheculer/schedule.json` 汇总各链，选择连续域验证通过且总面积最小的结果；若都失败，按相同原始载荷集合的覆盖数选择部分结果。`final_contacts.npz` 和方向记录继续供后续步骤读取。手动审计工具可重放每条链的抽样、尺寸更新、实际几何和验证证书，默认运行不调用。断点续跑核对种子、搜索参数及代码／输入哈希，参数不匹配时重新运行对应链。

例如只运行 Step3：

```bash
/Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step3_scheculer/run_all.py B --pose pose_1 --from-step 3 --through-step 3 --search-seed 0 --no-round-drawings
```

以下为原 greedy 模式的规则及两种模式共享的物理模型说明。

Schedule 最多选 **3 块**，不回溯、不增设第 4 块。每轮先筛共同插入方向及同方向连接证据，再检查该组合在纯重力下的物体平衡和支撑不被上抬；通过硬筛选后才按合法工作载荷的覆盖量排名。部分覆盖允许，零增益不自动淘汰，但硬条件不能靠覆盖率补偿。缩小接触半径时增加纯重力平衡的可行半径下界，导出实际三角面后再次验证。

达到 3 块仍未通过连续域覆盖时返回 `contact_limit_reached`；候选全被硬筛掉时返回 `candidates_exhausted`。保留已选头、逐轮图、覆盖率和失败原因。此处的硬受力条件是已约定的整体不上抬必要条件，并非对尚未设计的实体底座作不倾覆保证。

每个载荷现在还要求 `sum(head_force_on_workpiece_z) >= 0`：所有头受到的物体反作用力合计不能向上抬起支撑。只加总头，不包含物体脚底；所有头共享一个约束，允许局部下压。力与力矩平衡、单边接触和这条约束在同一个 LP 中联合求解。评分、半径优化、连续域证明和独立审计均使用此模型。

实现上保持 Step1 的六维需求，求解时补第七个零：头部生成元的第七维为法向的 z 分量，物体脚底生成元为零，另加 `[0,0,0,0,0,0,-1]` 的非负松弛变量表示底座总法向反力。旧结果没有此约束，不能作为新模型的完成证书。该必要条件只排除整体上抬，真实底座的摩擦、翻倒及完整平衡仍由后续检查。

2026-09-12：全部接触头最终属于同一个打印刚体，方向模式为 `common_rigid_withdrawal_3d`。不再使用“每个头分别能插入”的资格条件。

每轮开始，将 Step2 候选的 `certified_directions.ids` 与已选实际头的共同方向集合求交。会堵死最后一个共同方向的候选不参与 Step3.1 评分；其余候选沿用联合覆盖评分、Step3.2 贪心排名。当前集合排除朝初始地面下方的方向；工作面法向只用于排序参考，不限制初次安装所在半球。

Step3.3 仍只优化本轮圆的半径，旧头几何固定。半径上限同时受真实几何和共同方向约束；每个接受尺寸必须至少保留一个与旧头共享的完整扫掠方向。导出接触三角面后重新计算方向，并与旧头求交，不能直接沿用 Step2 初始尺寸的方向。每块面积仍须严格大于物体面积的 0.5%。

`round_*/candidate_filter.json` 保存已选编号、选择前方向集合、各候选加入后的集合和过滤原因。优化目录和最终 `insertion_directions.json` 保存实际头的几何签名、各头阻挡/未决方向、共同剩余方向及代表向量。所有 ID 均引用同一个不可变三维方向表，退出是支撑沿 `+d` 运动，装入反向。

共同方向保证的是：在声明的数值容差内，所有实际接触头沿该直线完整退出且不穿地面。它不保证尚未构造的框架、底座和连接件可行。方向表是有限三维搜索，表耗尽不证明整个连续球面或带旋转路径无解。

完成条件仍是优化后所有存储载荷覆盖，再通过连续载荷域验证；真实反例加入需求后在剩余轮次继续贪心；连续验证未决时停止为 `continuous_validation_inconclusive`，不当成可行或全局无解。候选耗尽仍生成明确的部分结果。审计重放每轮方向交集，并独立检查全部剩余方向的所有实际头扫掠。

此阶段保存可重建的头部连接证据，不导出最终腰带。Step4 算法不变；当前 Step5 从联合通过的 d0 中构造完整蓝块，另选独立对接方向设计固定底座与矩形接口；Step6 验证并展示初次安装、整体搬运和对接。

地面遵循“摩擦足够”的假设：原始工件落地点使用四射线单边摩擦锥，`|Fx|+|Fy| <= 64 Fz, Fz >= 0`，作为 Step3 的有限充分摩擦见证；Step5 对独立模块另行检查接口与底座平衡，并可使用既有的更高摩擦档位。该数值是有限的充分摩擦假设，不是材料实测值。评分、半径优化及独立连续复核使用相同模型；没有增加地面接触点、拉力或自由力矩。头与工件仍为无摩擦法向接触，最终支撑自身平衡由 Step5 检查。

查看逐轮方向变化：`python slides/baseline_algo/step3_scheculer/draw_directions.py B --pose pose_1`。`withdrawal_directions.png` 中绿色为剩余方向，红色为本轮新增锁定，灰色为此前锁定；第一幅显示当前方向表的初始锁定；新模型只按初始地面过滤，工作面法向不再锁定整个半球。

最终接触与共同方向合图：`python slides/baseline_algo/step3_scheculer/draw_result.py B --pose pose_2`。同姿态的 `step3_scheculer/selected_contacts_directions.png` 用不同颜色高亮实际优化后的接触面，箭头显示同一个已认证退出方向，方向球显示全部共同剩余方向；装入沿箭头反向。图中不标接触编号，遮挡面以透视方式显示。只读取并核对已保存的接触及方向，不重跑选块或尺寸优化；离散方向点不代表连续角域认证。

单例评分较慢时，可设置 `CADGRASP_SCORE_WORKERS=6` 再运行 `run_all.py ... --resume`。独立进程预计算原始分类与独立 LP 验证，原始评分函数按候选编号写结果；保留串行检查点，新轮次额外记录 `parallel_scoring.py` 源码哈希。该加速不改变评分目标、候选数量、载荷样本或几何检查。

连续验证的可选对偶凸包很慢时，可加 `--primal-first-retry`（`run_all.py` 和批量入口均支持）：先尝试已有的切向外包多面体原始反力证书，失败再走原来的连续验证及反例搜索。直接证明失败不会被当作物理反例；成功证书仍由原审计独立重放。报告记录 `primal_first_retry.py` 的源码哈希。该选项不改接触搜索或承载模型。

并行模式下，旧的每轮 `scoring_seconds` 只计原评分函数读取预计算结果并写出的耗时，不含预计算；运行总耗时应读取调度器或批量台账的 `elapsed_seconds`，不能拿该逐轮字段比较串行/并行速度。

2026-09-13：`schedule.json` 新增 `wall_timings`。`3.1` 包含方向筛选、并行评分预计算及结果写入；`3.2` 为实际选择调用；`3.3` 包含尺寸优化、方向更新及逐轮绘图，另保存这些内部子项；`3.validation` 为连续域检查。父项与子项不能重复相加。断点续跑记录本次读取/计算的时间，不把历史缓存中的耗时再次计入。完整阶段（当前含最终验证和结果图，不含手动审计）见总入口输出的 `timing.json`。

## B：2026-09-22 新流程重跑

以下为历史记录。旧 B 输出已按要求于 2026-09-26 清除；此表不代表当前双 pose 实验结果，原 Step6 结果链接已移除。

本轮按要求复用 pose_1～4 的 Step1～4，重算紧凑闭环底座及 Step6 视频。圈只围绕需求凸包与原始支点，候选按实际接地凸包面积从小到大验收；不是任意形状下的全局最小证明。

| Pose | 实际接地凸包面积（cm²） | 相对上一版缩减 | 几何／完整承载 | KUKA 逐帧 IK |
|---|---:|---:|---|---|
| pose_1 | 173.72 | 38.5% | 通过 | 通过 |
| pose_2 | 240.55 | 78.4% | 通过 | 通过 |
| pose_3 | 176.25 | 44.7% | 通过 | 通过 |
| pose_4 | 216.28 | 80.5% | 通过 | 通过 |

四例均重放当前几何及三刚体承载证据，充分地面摩擦见证分别为 256、1024、256、256，属于既有理想摩擦菜单，不是实测材料值。视频为 1600×900、24 fps、15 秒，地面与镜头固定；机械臂松开后显示三组保存的合法任务力。逐帧 IK 用于演示，不认证机器人碰撞、抓持保持或动力学。

此前 pose_1～5 的 Step3 连续覆盖均通过；pose_5 曾在旧底座搜索中未取得完整承载证书，本轮未重跑其 Step5～6。pose_6～10 仍为更早的整件支撑流程，不能并入当前结果。

复现本次依赖重算及视频：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step3_scheculer/rerun_common_directions.py \
  --from-step 5 --through-step 6 --workers 2 \
  --case B:pose_1 --case B:pose_2 --case B:pose_3 --case B:pose_4 \
  --no-round-drawings --primal-first-retry
```
