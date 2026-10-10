# Step4.2 B：七个原失败组完整结果

完整批次 7/7 通过当前力学、退出与每侧1%净空范围；每组六个 pose，各 32768 条原始载荷通过。小步方向搜索解决 2/7，计算分离与收缩保底解决 5/7。后五组的厘米量级位移不能计作微小平移成功。全夹具连通、安装接地/穿地及强度尚未验收，full_fixture_accepted=False。

| pose组 | 解决方法 | 最大平移 mm | 用时 s | 报告 |
|---|---|---:|---:|---|
| pose1+2+3+4+6+7 | 小步方向，零平移 | 0.00 | 268.6 | [report](pose1+2+3+4+6+7/data/report.json) |
| pose1+2+3+4+7+27 | 小步方向，零平移 | 0.00 | 54.0 | [report](pose1+2+3+4+7+27/data/report.json) |
| pose1+2+4+5+6+7 | 计算分离+收缩保底 | 159.25 | 1138.4 | [report](pose1+2+4+5+6+7/data/report.json) |
| pose1+2+4+5+6+11 | 计算分离+收缩保底 | 226.84 | 1246.9 | [report](pose1+2+4+5+6+11/data/report.json) |
| pose1+4+7+12+21+27 | 计算分离+收缩保底 | 542.80 | 782.6 | [report](pose1+4+7+12+21+27/data/report.json) |
| pose1+6+11+13+14+17 | 计算分离+收缩保底 | 236.88 | 969.5 | [report](pose1+6+11+13+14+17/data/report.json) |
| illegal/pose4+7+12+21+23+27 | 计算分离+收缩保底 | 544.77 | 1233.5 | [report](illegal/pose4+7+12+21+23+27/data/report.json) |

两个小步成功组：pose1+2+3+4+7+27 仅 pose2 改约2°，2步；pose1+2+3+4+6+7 最大累计方向变化约4.82°，7步。每个局部步至多1°，均未平移。其余组局部方向搜索有部分改善，仍未全部解决，报告 local_search_report.json 保留真实计数。

## 可复现与版本

configuration.json 为实际参数：iterations=12, candidates=4, finalists=2, workers=2, seed=42。runtime_sources 是启动时源文件；本批次早于后续候选比较和联合外侧事件修正。逐组 provenance.code 已按启动快照校正，原报告时文件哈希保存在 code_hashes_at_report_time，输入哈希不变。input_audit.json 检查全部输入哈希一致。

当前源代码给方向、平移各验收名额，完整比较入选候选，并给重合物体加入联合平移的外侧约束。这些修正只在额外困难组 pilot 中测试：balanced_tools_v2_pilot_B_20261007 与 joint_outward_event_pilot_B_20261007，四轮均未新增小步成功。其平移代理预测明显过度乐观，实际验收拒绝新锁定导致的退步候选。不能把本七组旧版本的结果作为修正版七组成功率。

122项测试通过。推导和运行入口见 [算法说明](../../../step4.2/algorithm.md)。历史试验和公开输出保持。
