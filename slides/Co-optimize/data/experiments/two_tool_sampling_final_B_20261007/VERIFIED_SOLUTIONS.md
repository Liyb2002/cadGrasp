# 多分支试验中最小的已验证平移布局

仅比较已实际通过的布局，不宣称全局最小。完整七组批次见 README.md。本表可能采用独立计算分离试验中更小的保底解，不代表局部微调成功。全夹具连通、安装接地和强度尚未验收。

| 组 | 方法 | 最大平移 mm | 原始报告 |
|---|---|---:|---|
| pose1+2+4+5+6+7 | 计算分离+收缩保底 | 149.51 | [report](../computed_layout_remaining_B_20261007/pose1+2+4+5+6+7/data/report.json) |
| pose1+4+7+12+21+27 | 计算分离+收缩保底 | 542.80 | [report](pose1+4+7+12+21+27/data/report.json) |
| pose1+2+4+5+6+11 | 计算分离+收缩保底 | 226.84 | [report](pose1+2+4+5+6+11/data/report.json) |
| pose1+2+3+4+7+27 | 小步方向，零平移 | 0.00 | [report](pose1+2+3+4+7+27/data/report.json) |
| pose1+6+11+13+14+17 | 计算分离+收缩保底 | 236.88 | [report](pose1+6+11+13+14+17/data/report.json) |
| pose1+2+3+4+6+7 | 小步方向，零平移 | 0.00 | [report](pose1+2+3+4+6+7/data/report.json) |
| illegal/pose4+7+12+21+23+27 | 计算分离+收缩保底 | 378.63 | [report](../computed_layout_illegal_check_B_20261007/illegal/pose4+7+12+21+23+27/data/report.json) |
