# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 24.164 → 60.097 cm³。
转动支撑复用 4 个pose；Juxtapose 5 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 200606 | 28.359 |  |
| 1 | selective_juxtapose | 180534 | 10.666 | pose_1: host pose_1→pose_27, 方向54.897°, 平移202.965mm |
| 2 | selective_juxtapose | 144267 | 16.138 | pose_1: host pose_27→pose_27, 方向4.516°, 平移9.749mm；pose_4: host pose_4→pose_4, 方向4.942°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向4.942°, 平移0.000mm；pose_9: host pose_9→pose_9, 方向4.942°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向4.072°, 平移0.000mm；pose_16: host pose_16→pose_16, 方向4.942°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向4.942°, 平移0.000mm；pose_23: host pose_23→pose_21, 方向38.076°, 平移114.476mm；pose_27: host pose_27→pose_27, 方向4.516°, 平移0.000mm |
| 3 | direction_after_commit | 142265 | 16.138 | pose_1: host pose_27→pose_27, 方向1.988°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向1.933°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向1.933°, 平移0.000mm；pose_9: host pose_9→pose_9, 方向1.933°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向1.975°, 平移0.000mm；pose_16: host pose_16→pose_16, 方向1.933°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向1.933°, 平移0.000mm；pose_23: host pose_21→pose_21, 方向1.951°, 平移0.000mm；pose_27: host pose_27→pose_27, 方向1.988°, 平移0.000mm |
| 4 | selective_juxtapose | 105183 | 19.369 | pose_27: host pose_27→pose_9, 方向38.948°, 平移205.056mm |
| 5 | direction_after_commit | 105183 | 19.369 | pose_23: host pose_21→pose_21, 方向5.000°, 平移0.000mm |
| 6 | direction_after_commit | 105183 | 19.369 | pose_23: host pose_21→pose_21, 方向0.500°, 平移0.000mm |
| 7 | selective_juxtapose | 53153 | 32.622 | pose_1: host pose_27→pose_27, 方向7.173°, 平移19.348mm；pose_4: host pose_4→pose_4, 方向0.609°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向0.609°, 平移0.000mm；pose_9: host pose_9→pose_9, 方向0.609°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向1.248°, 平移0.000mm；pose_16: host pose_16→pose_16, 方向0.609°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向0.609°, 平移0.000mm；pose_23: host pose_21→pose_9, 方向91.313°, 平移225.206mm；pose_27: host pose_9→pose_9, 方向0.609°, 平移0.000mm |
| 8 | selective_juxtapose | 8139 | 51.087 | pose_1: host pose_27→pose_27, 方向0.000°, 平移4.837mm；pose_7: host pose_7→pose_21, 方向11.292°, 平移176.966mm |
| 9 | selective_juxtapose | 0 | 79.377 | pose_4: host pose_4→pose_9, 方向8.688°, 平移187.703mm |
| 10 | feasible_volume_descent | 0 | 69.024 | pose_23: host pose_9→pose_9, 方向0.000°, 平移4.837mm |
| 11 | feasible_volume_descent | 0 | 64.422 | pose_4: host pose_9→pose_9, 方向0.000°, 平移4.837mm |
| 12 | feasible_volume_descent | 0 | 59.821 | pose_27: host pose_9→pose_9, 方向0.000°, 平移4.837mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。
