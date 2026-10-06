# B 非法 pose set：当前算法实验

Step3.3 已更新为原始需求点的最小凸包围边。本文的 Step4 图片、承载和连通结果保留自此前圆环版本；未针对新的 Step3.3 模型重跑，不能作为新模型的接受结果。

输入：[illegal_pose_sets.json](/home/yli581/Desktop/cadGrasp/objects/B/illegal_pose_sets.json)。10 组，2–6 个 pose 各两组；复用原始姿态、工作面与全部 32,768 个需求，不重新采样。

**完整包裹承载检查 10/10 通过；Step4.2 承载、合法物体退出及连通 9/10 通过。** 已核对 34 个 pose 实例的 1,114,112 个保存需求通过记录。

**这不表示原来的非法性已消除。** JSON 的失败条件是原生地面需求点映射到其他 pose 后落到地下；所有有向失败计数已复现。当前 Step4.2 尚未把安装后支撑的所有地面约束、真实接地材料覆盖纳入接受，因此两种结论可以同时成立。下表的地下深度是已有模型的描述性坐标诊断，不是新增接受或导出模型回放。

| Pose set | 原有有向非法样本数 | Step4.2 | 原始需求通过数 | 剩余块数 | 支撑最大地下深度 mm | 图 |
|---|---:|---|---:|---:|---:|---|
| pose2+29 | 1 | PASS（当前阶段） | 65,536/65,536 | 1 | 6.857 | [退出图](pose2+29/step4/step4.2/exit_motion.png) |
| pose11+19 | 32,768 | PASS（当前阶段） | 65,536/65,536 | 1 | 52.900 | [退出图](pose11+19/step4/step4.2/exit_motion.png) |
| pose1+2+29 | 1 | PASS（当前阶段） | 98,304/98,304 | 1 | 6.857 | [退出图](pose1+2+29/step4/step4.2/exit_motion.png) |
| pose5+13+27 | 83,929 | PASS（当前阶段） | 98,304/98,304 | 1 | 44.977 | [退出图](pose5+13+27/step4/step4.2/exit_motion.png) |
| pose1+2+6+15 | 1 | PASS（当前阶段） | 131,072/131,072 | 1 | 8.316 | [退出图](pose1+2+6+15/step4/step4.2/exit_motion.png) |
| pose7+11+13+19 | 95,972 | PASS（当前阶段） | 131,072/131,072 | 1 | 58.416 | [退出图](pose7+11+13+19/step4/step4.2/exit_motion.png) |
| pose1+2+6+7+29 | 1 | PASS（当前阶段） | 163,840/163,840 | 1 | 6.857 | [退出图](pose1+2+6+7+29/step4/step4.2/exit_motion.png) |
| pose5+6+13+15+27 | 126,984 | PASS（当前阶段） | 163,840/163,840 | 1 | 50.930 | [退出图](pose5+6+13+15+27/step4/step4.2/exit_motion.png) |
| pose4+7+12+21+23+27 | 1 | force_recovery_search_unresolved | 未全部通过 | — | 未接受 | [未决结果](pose4+7+12+21+23+27/README.md) |
| pose5+12+20+24+27+28 | 137,648 | PASS（当前阶段） | 196,608/196,608 | 1 | 44.977 | [退出图](pose5+12+20+24+27+28/step4/step4.2/exit_motion.png) |

搜索使用保存的自身 +z 初始化、共同方向趋势与连通块承载检查，初始物体位置固定。每组 force / connectivity 两个搜索阶段各设 180 秒预算；超时为 UNRESOLVED，不证明无解。

独立输出目录为 `output/B/illegal/`；原 20 组结果和原始 objects 输入保持不变。内部 `data/` 保存来源、失败计数复现、逐 pose 原始需求掩码和搜索记录。959 个受保护原始输入的哈希一致。

复现：先运行 `run_illegal.py --jobs 2`，再运行 `illegal_recovery.py --jobs 2 --seconds-per-phase 180`，最后运行 `summarize_illegal.py`。

下一步的初始位置平移，可以研究是否改善这些地面不兼容关系；必须同时重算真实接触、力矩、退出与地面条件。此批次没有启用位置平移。
