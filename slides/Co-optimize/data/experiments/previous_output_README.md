# Co optimize 输出

主算法是 sampling 搜索共同退出趋势，再以真实承载缺口引导梯度局部调整。当前主版本为 `step4.2/hybrid_clearance_fast.py`。

## 主结果

| 入口 | 内容 | 已保存结果 |
| --- | --- | --- |
| [current/B](current/B/) | 当前混合算法，B 的完整 30 组 | 承载与退出 28/30；2 组初始通过，26 组恢复 |
| [current/other_objects](current/other_objects/) | 同一版本，其余 20 个对象各固定一组 5 poses | 20 组均结束，6/20 通过 |
| [current/B_previous_fast](current/B_previous_fast/) | 净空数值修复前的 fast 版 | 承载与退出 28/30，独立历史对照 |

进入每个入口先看 `batch.json`，再看组目录中的 `report.json`、`remaining_support.obj` 和搜索日志。`current` 是相对软链接，不复制结果。原始结果仍保留在 `physics_guided_clearance_fast_batch/`、`generalization5_ready/`、`physics_guided_fast_batch/`，以保持主结果来源路径有效。

B 未通过：`pose1+4+7+12+21+27`、`illegal/pose4+7+12+21+23+27`。搜索预算耗尽不构成无解证明。

其他对象通过：C3、C5、C6、D1、D3、D4。其余 14 组中，6 组达到时间预算，6 组 unresolved，2 组完成有界搜索但未通过。单组状态与原因见批次记录。

通过范围为每个 pose 的全部原始载荷和带 1% 净空的完整退出。连通、安装后支撑的接地覆盖、强度及机器人运动尚未作为通过条件。

## 前置数据与缓存

顶层 `B/`、`A*/`、`C*/`、`D*/`、`cuboid_baseline/` 保存注册、包裹壳、接地围边、Step4.1 输入以及早期阶段结果；它们不是当前混合算法的 Step4.2 主结果。`physics_guided_cache/` 是距离场缓存。两类路径均由主结果引用，保持原位。

## 历史实验

- [history/comparisons](history/comparisons/)：连续模型、Step7、串行混合、原始并行、final、adaptive、balanced 的独立批次。
- [history/generalization](history/generalization/)：被替代或停止的其他物体测试及性能探查。
- [history/diagnostics](history/diagnostics/)：review、pilot、continuation、旧连续优化输出和诊断汇总。
- [history/logs](history/logs/)：原先散落在顶层的日志。

历史文件内容与哈希保持不变；其报告内旧绝对路径、相对来源路径仍是当时的记录。移动后的路径由 [history/migration.json](history/migration.json) 的 old/new 前缀映射定位。历史脚本中的旧默认输出路径不应作为当前结果入口。
