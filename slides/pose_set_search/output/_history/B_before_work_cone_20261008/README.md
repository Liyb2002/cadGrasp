# B：10 个集合，两种快速搜索

**工作射线约束不通过：现有 20 个支撑全部找到真实遮挡反例，不能作为可行解。** 原有“20 次采样载荷通过”仅是力／力矩检查，遗漏了从整个工作面积向外射出的完整路径。[遮挡诊断](work_ray_audit.md) · [逐结果反例](work_ray_audit.json)。

**20 次独立搜索已完成；不做最终验收。** 每个 pose 在采样支撑接触上检查全部原始 32,768 个载荷。

[对比图](comparison.png) · [可点击浏览](index.html) · [搜索清单](batch.json) · [相同预算](experiment.json)

图中的蓝色支撑和 `support.obj` 都是从保存布局直接重建的三角 mesh，保留全部当前物体、工作带、完整退出的切除。搜索结果保持不变。体积列使用重建 mesh 的材料体积；括号保留共同 131,072 点空间采样估计。

**每个 set 两个视频**：`process.mp4` 展示已选操作和材料增减，`result.mp4` 展示所有 pose 依次装入、取出，最后查看空支撑。固定单个等轴测视角，先 whole、后 incremental，中间短暂白场分隔，白底无文字。橙色面片为该 pose 的原始工作面积；放稳后停顿 1 秒，橙红色小箭头沿工作面内法线，表示可能施加的力。过程图每步只有一个等轴测视角；具体选择和步幅仍在各方法 README。

| 集合 | whole：mesh cm³（采样估计） / 搜索 s / 转动+J | incremental：mesh cm³（采样估计） / 搜索 s / 转动+J | 视频与过程图 |
|---|---|---|---|
| pose1+2+3+4+5+6+7 | 132.63（估计 132.60） / 46.2 / 5+2 (采样通过) | 146.89（估计 148.52） / 38.2 / 6+1 (采样通过) | [操作视频](pose1+2+3+4+5+6+7/process.mp4) · [使用视频](pose1+2+3+4+5+6+7/result.mp4) · [whole](pose1+2+3+4+5+6+7/whole/process.png) · [incremental](pose1+2+3+4+5+6+7/incremental/process.png) |
| pose1+2+4+5+6+7+11 | 218.93（估计 220.63） / 60.2 / 5+2 (采样通过) | 159.53（估计 158.15） / 63.6 / 5+2 (采样通过) | [操作视频](pose1+2+4+5+6+7+11/process.mp4) · [使用视频](pose1+2+4+5+6+7+11/result.mp4) · [whole](pose1+2+4+5+6+7+11/whole/process.png) · [incremental](pose1+2+4+5+6+7+11/incremental/process.png) |
| pose1+4+7+12+21+23+27 | 154.72（估计 155.81） / 26.6 / 6+1 (采样通过) | 129.01（估计 130.04） / 27.3 / 6+1 (采样通过) | [操作视频](pose1+4+7+12+21+23+27/process.mp4) · [使用视频](pose1+4+7+12+21+23+27/result.mp4) · [whole](pose1+4+7+12+21+23+27/whole/process.png) · [incremental](pose1+4+7+12+21+23+27/incremental/process.png) |
| pose1+2+3+4+5+6+7+8 | 132.63（估计 132.60） / 52.8 / 6+2 (采样通过) | 146.89（估计 148.52） / 40.3 / 7+1 (采样通过) | [操作视频](pose1+2+3+4+5+6+7+8/process.mp4) · [使用视频](pose1+2+3+4+5+6+7+8/result.mp4) · [whole](pose1+2+3+4+5+6+7+8/whole/process.png) · [incremental](pose1+2+3+4+5+6+7+8/incremental/process.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 159.14（估计 159.32） / 177.9 / 6+4 (采样通过) | 125.86（估计 124.84） / 44.9 / 9+1 (采样通过) | [操作视频](pose1+2+3+4+5+6+7+8+9+10/process.mp4) · [使用视频](pose1+2+3+4+5+6+7+8+9+10/result.mp4) · [whole](pose1+2+3+4+5+6+7+8+9+10/whole/process.png) · [incremental](pose1+2+3+4+5+6+7+8+9+10/incremental/process.png) |
| pose1+2+4+5+6+7+10+11 | 146.62（估计 146.09） / 62.6 / 6+2 (采样通过) | 187.42（估计 186.88） / 70.2 / 6+2 (采样通过) | [操作视频](pose1+2+4+5+6+7+10+11/process.mp4) · [使用视频](pose1+2+4+5+6+7+10+11/result.mp4) · [whole](pose1+2+4+5+6+7+10+11/whole/process.png) · [incremental](pose1+2+4+5+6+7+10+11/incremental/process.png) |
| pose1+3+7+10+14+18+23+27 | 94.24（估计 93.78） / 9.2 / 8+0 (采样通过) | 131.03（估计 132.14） / 106.8 / 2+6 (采样通过) | [操作视频](pose1+3+7+10+14+18+23+27/process.mp4) · [使用视频](pose1+3+7+10+14+18+23+27/result.mp4) · [whole](pose1+3+7+10+14+18+23+27/whole/process.png) · [incremental](pose1+3+7+10+14+18+23+27/incremental/process.png) |
| pose1+2+3+4+5+6+7+8+9 | 141.85（估计 141.75） / 109.3 / 6+3 (采样通过) | 129.38（估计 129.43） / 43.0 / 8+1 (采样通过) | [操作视频](pose1+2+3+4+5+6+7+8+9/process.mp4) · [使用视频](pose1+2+3+4+5+6+7+8+9/result.mp4) · [whole](pose1+2+3+4+5+6+7+8+9/whole/process.png) · [incremental](pose1+2+3+4+5+6+7+8+9/incremental/process.png) |
| pose1+4+7+9+12+16+21+23+27 | 142.72（估计 143.92） / 104.7 / 8+1 (采样通过) | 127.12（估计 127.14） / 32.7 / 8+1 (采样通过) | [操作视频](pose1+4+7+9+12+16+21+23+27/process.mp4) · [使用视频](pose1+4+7+9+12+16+21+23+27/result.mp4) · [whole](pose1+4+7+9+12+16+21+23+27/whole/process.png) · [incremental](pose1+4+7+9+12+16+21+23+27/incremental/process.png) |
| pose1+3+5+7+9+12+16+21+23+27 | 99.10（估计 101.38） / 56.7 / 9+1 (采样通过) | 167.68（估计 168.18） / 95.5 / 6+4 (采样通过) | [操作视频](pose1+3+5+7+9+12+16+21+23+27/process.mp4) · [使用视频](pose1+3+5+7+9+12+16+21+23+27/result.mp4) · [whole](pose1+3+5+7+9+12+16+21+23+27/whole/process.png) · [incremental](pose1+3+5+7+9+12+16+21+23+27/incremental/process.png) |

重建 mesh 材料体积更小：whole 5 组，incremental 5 组。这是一次固定预算的快速比较，没有全局最优结论。

每个方法目录的 `process.png` 无文字，仅显示依次选中的新支撑；`README.md` 保留操作步幅、拒绝回合、最难载荷和各 finalist 结果；`final_result.png` 显示同一蓝色支撑在每个 pose 中的摆放。每步布局在 `process_states/`，所有原始候选在 `trace.json`。

20 次纯搜索累计 21.1 分钟；双进程批次墙钟 12.1 分钟，另含图像生成。

保持 Co-optimize 数值代码和既有验收结果不变；本批次独立保存在这里。
