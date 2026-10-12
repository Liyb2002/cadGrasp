# 搜索加速与计时

本轮已获准跑十组快速比较，并明确跳过最终验收。新的搜索、过程图和共同采样框体积比较放在 [output/B](output/B/README.md)，与下面先前的单集合精确验收测速分开。

十组已完成：20 次独立纯搜索累计 **21.1 分钟**，双进程墙钟 **12.1 分钟**；单次约9–178秒。20/20 整组采样载荷通过，搜索和最终阶段均没有精确实体评估。20份过程图、20份最终姿态图另用快速材料占用网格绘制，所有已选布局及候选选择记录保留在各方法目录。共同空间采样框比较的体积，whole 与 incremental 各5组更小。

当前入口 `code/run_reuse_first.py` 默认 `--evaluation sampled`。在搜索中不重建完整支撑；最终导出保持原实体、接触、全部载荷与 WORK 验收。原 Co-optimize 生产源码及已发布结果未修改。

## 改了什么

- `delta_guidance.py` 保留材料覆盖／切除计数，以及每个任务的接触覆盖／锁定计数。Direction 更新 blocker 的列；Translation／Juxtapose 还更新移动任务的接触行。候选不修改当前状态；仍被其他任务锁住的面积不能恢复。
- `fast_search.py` 保留这些状态及反力查询缓存。反力基只在其全部原接触列仍存在时复用，并回代原七维方程；不改变压力容量、地面模型或原载荷。
- 接触同时检查与精确算法一致的 `normal·local_exit <= 1e-9`。仅凭近表面的射线查询会误保留少数前向面；这一问题在最终核对中发现并修正。每个原三角面保留内部采样，另加靠近角点的采样以覆盖力臂范围。
- Juxtapose 每轮默认最多初筛 96 个不同候选，保留不同 guest／host 并采样多个偏移／方向。各操作仍有多个步幅。全过程不做逐候选 Boolean。
- 最终优先验收选中的状态，数值未解或验收失败时，最多尝试三个已保存的整组采样可行状态。所有尝试保存在 `final_validation_attempts.json`；没有完整实体验收就不输出成功报告。
- 精确构造去重相同 body／sweep／包裹，复用已经完成的交集诊断，并记录分阶段时间。改变几何容差或绕过 WORK 都不在此修改中。

## pose1–10 实测

常规预算：8 轮 frontier、每轮 3 个入选候选、1 轮分支调整、2 轮减体积、96 个初筛候选。此表是先前仅测一个集合的单线程计时；本轮十组使用相同搜索预算，双进程运行且不做最终验收。

| 顺序 | 从头纯搜索 | 入选采样评估 | 搜索中精确评估 | 最终验收／导出 |
|---|---:|---:|---:|---:|
| whole / joint | 174.54 s | 47 | 0 | 23.94 s，通过 |
| incremental | 44.50 s | 26 | 0 | 首次 WORK 数值未解 49.14 s；前一状态 58.38 s，通过 |

单个入选候选通常约 1–3 秒；接触不变时可以直接复用。两份最终实体均通过全部 327,680 条原载荷以及完整退出／WORK，实体体积分别为 130.781、116.668 cm³。这是测速搜索的结果，不替换已有更小的已发布支撑，也不说明全局最优。增量的中间加入前缀仅经过采样检查，没有逐个做完整实体再生验收。

数据：[whole 搜索](output/speed_delta_v4/pose1-10/joint/search_report.json)、[incremental 搜索](output/speed_delta_v4/pose1-10/incremental/search_report.json)、[最终验收计时](output/speed_delta_v4/validation_timings.json)、[增量回退](output/speed_delta_v4/validation_fallback_timings.json)。各运行保留启动参数、代码快照和原输入哈希。

此前 joint 末尾两个各卡住 180 秒的布局，当前采样评估分别为 **1.18 s、1.32 s**，不调用 Boolean。见 [同一布局的采样计时](output/speed_delta_v4/timeout_candidate_sampling.json)。这表示搜索绕开了昂贵实体运算，并不表示原 Boolean 已在一秒内精确求解。

较早 v1–v3 是诊断试跑，不能用未修正前向面判断的计时替代 v4。原旧算法 13.9／8.2 分钟的记录仅是已有可行解之后的减体积续跑，不与这里的从头搜索计算加速比。

## 批量时间估计

对同一 B shape、7–10 个以上 pose、相近预算：10 个集合 × 两种顺序，纯搜索先估 **30–60 分钟**。最终实体构造与验收通常再花几十秒／结果，数值回退会增加时间；搜索加最终验收先按 **1–2 小时**预留。过程图渲染尚未计时，不包含在该估计中。固定搜索预算不能保证每个新集合在此时间内解决。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u slides/pose_set_search/code/run_reuse_first.py \
  --case pose1-10 --mode both --out slides/pose_set_search/output/new_delta_run \
  --iterations 8 --finalists 3 --branch-rounds 1 --volume-rounds 2 \
  --screen-budget 96 --final-candidates 3
```

加 `--search-only` 只测搜索、不做最终导出；`search_report.json` 明确没有最终验收。`--evaluation exact` 保留逐候选完整实体评估。28 项检查通过，包括增量／全量采样一致性、共同遮挡、平移影响其他任务、前向面限制、反力基失效、无 Boolean 搜索、最终 WORK 拒绝和回退，以及原模型不变量。
