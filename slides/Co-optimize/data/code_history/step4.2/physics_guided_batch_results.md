# Step4.2 连续算法：30 组批量结果

本轮 20 个正常组、10 个非法组全部运行结束。所有组从 Step4.1 的初始化方向起步，不读取旧 Step4.2 优化方向；每组最多 8 轮、完整轨迹距离场指导、保留每侧 1% 余量，不优化连通性。

**通过 3/30（10%）：正常组 0/20，非法组 3/10。只有 1 组由新算法从失败状态修复；另 2 组初始化已可行。**

成功修复组为 `illegal/pose1+2+29`：初始载荷覆盖为 `[641, 32740, 32767]`，四次接受更新后达到 `[32768, 32768, 32768]`，耗时 167.42 秒；退出与余量构造通过，单连通未通过，按本轮口径计成功。初始可行组为 `illegal/pose11+19` 和 `illegal/pose5+13+27`。

其余 27 组未通过：24 组真实目标线搜索停滞，3 组初始化几何构造数值未解。数值未解不代表物理不可行。最后一组六 pose 已把五个 pose 补齐，另一个剩 513 个失败载荷，仍严格计失败。

这批结果说明当前连续松弛在部分组能提供有效方向，但尚不能作为可靠的独立求解器；先前采用旧算法方向的单组演示不能代表从初始化起步的表现。

27 份完整结果均通过原输入、执行源码与产物哈希核对。原始 30 组运行汇总留存于 `batch_runner_original.json`；最终汇总修正了终点体积列表的类型处理，优化轨迹和每组算法报告保持原样。

| Pose set | 本轮结果 | 初始失败载荷总数 | 最终失败载荷总数 |
|---|---|---:|---:|
| [pose3+15](../data/experiments/history/comparisons/physics_guided_batch/B/pose3+15/run.log) | 构造数值未解 | — | — |
| [pose19+28](../data/experiments/history/comparisons/physics_guided_batch/B/pose19+28/report.json) | 线搜索停滞 | 8883 | 5072 |
| [pose8+21](../data/experiments/history/comparisons/physics_guided_batch/B/pose8+21/run.log) | 构造数值未解 | — | — |
| [pose2+20](../data/experiments/history/comparisons/physics_guided_batch/B/pose2+20/run.log) | 构造数值未解 | — | — |
| [pose1+12+29](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+12+29/report.json) | 线搜索停滞 | 41378 | 1574 |
| [pose4+5+7](../data/experiments/history/comparisons/physics_guided_batch/B/pose4+5+7/report.json) | 线搜索停滞 | 90399 | 90399 |
| [pose8+10+19](../data/experiments/history/comparisons/physics_guided_batch/B/pose8+10+19/report.json) | 线搜索停滞 | 96069 | 96069 |
| [pose18+23+24](../data/experiments/history/comparisons/physics_guided_batch/B/pose18+23+24/report.json) | 线搜索停滞 | 97946 | 91329 |
| [pose8+9+13+30](../data/experiments/history/comparisons/physics_guided_batch/B/pose8+9+13+30/report.json) | 线搜索停滞 | 131072 | 131072 |
| [pose2+3+4+7](../data/experiments/history/comparisons/physics_guided_batch/B/pose2+3+4+7/report.json) | 线搜索停滞 | 100559 | 100559 |
| [pose1+11+14+27](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+11+14+27/report.json) | 线搜索停滞 | 131072 | 131072 |
| [pose5+6+23+29](../data/experiments/history/comparisons/physics_guided_batch/B/pose5+6+23+29/report.json) | 线搜索停滞 | 131072 | 131072 |
| [pose1+4+7+9+24](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+4+7+9+24/report.json) | 线搜索停滞 | 163840 | 163840 |
| [pose5+6+11+23+29](../data/experiments/history/comparisons/physics_guided_batch/B/pose5+6+11+23+29/report.json) | 线搜索停滞 | 163840 | 163840 |
| [pose12+16+19+21+27](../data/experiments/history/comparisons/physics_guided_batch/B/pose12+16+19+21+27/report.json) | 线搜索停滞 | 163840 | 163840 |
| [pose6+10+13+17+30](../data/experiments/history/comparisons/physics_guided_batch/B/pose6+10+13+17+30/report.json) | 线搜索停滞 | 163840 | 163840 |
| [pose1+2+4+5+6+7](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+2+4+5+6+7/report.json) | 线搜索停滞 | 196608 | 196608 |
| [pose4+5+8+9+19+23](../data/experiments/history/comparisons/physics_guided_batch/B/pose4+5+8+9+19+23/report.json) | 线搜索停滞 | 196608 | 196608 |
| [pose1+6+11+13+14+17](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+6+11+13+14+17/report.json) | 线搜索停滞 | 196608 | 196608 |
| [pose1+4+7+12+21+27](../data/experiments/history/comparisons/physics_guided_batch/B/pose1+4+7+12+21+27/report.json) | 线搜索停滞 | 196608 | 196608 |
| [illegal/pose2+29](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose2+29/report.json) | 线搜索停滞 | 27 | 27 |
| [illegal/pose11+19](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose11+19/report.json) | 初始可行 | 0 | 0 |
| [illegal/pose1+2+29](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose1+2+29/report.json) | 优化修复 | 32156 | 0 |
| [illegal/pose5+13+27](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose5+13+27/report.json) | 初始可行 | 0 | 0 |
| [illegal/pose1+2+6+15](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose1+2+6+15/report.json) | 线搜索停滞 | 131072 | 75761 |
| [illegal/pose7+11+13+19](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose7+11+13+19/report.json) | 线搜索停滞 | 31770 | 31489 |
| [illegal/pose1+2+6+7+29](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose1+2+6+7+29/report.json) | 线搜索停滞 | 161106 | 161106 |
| [illegal/pose5+6+13+15+27](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose5+6+13+15+27/report.json) | 线搜索停滞 | 6611 | 6357 |
| [illegal/pose4+7+12+21+23+27](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose4+7+12+21+23+27/report.json) | 线搜索停滞 | 196608 | 196608 |
| [illegal/pose5+12+20+24+27+28](../data/experiments/history/comparisons/physics_guided_batch/B/illegal/pose5+12+20+24+27+28/report.json) | 线搜索停滞 | 108461 | 513 |

[完整批次记录](../data/experiments/history/comparisons/physics_guided_batch/B/batch.json) · [算法说明](physics_guided_README.md)
