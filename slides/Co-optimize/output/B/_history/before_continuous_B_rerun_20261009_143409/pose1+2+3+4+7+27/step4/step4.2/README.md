# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 28.398 → 85.193 cm³。
转动支撑复用 4 个pose；Juxtapose 2 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 65619 | 33.015 |  |
| 1 | rotate_reuse_direction | 65619 | 33.227 | pose_1: host pose_1→pose_1, 方向1.474°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向1.484°, 平移0.000mm；pose_3: host pose_3→pose_3, 方向1.628°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向1.628°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向1.715°, 平移0.000mm；pose_27: host pose_27→pose_27, 方向1.135°, 平移0.000mm |
| 2 | rotate_reuse_direction | 65619 | 31.957 | pose_1: host pose_1→pose_1, 方向2.000°, 平移0.000mm |
| 3 | rotate_reuse_direction | 65619 | 30.581 | pose_1: host pose_1→pose_1, 方向2.000°, 平移0.000mm |
| 4 | selective_juxtapose | 33096 | 63.216 | pose_1: host pose_1→pose_4, 方向9.613°, 平移160.448mm |
| 5 | direction_after_commit | 33096 | 62.505 | pose_2: host pose_2→pose_2, 方向5.000°, 平移0.000mm |
| 6 | selective_juxtapose | 0 | 102.879 | pose_2: host pose_2→pose_7, 方向54.224°, 平移146.110mm |
| 7 | feasible_volume_descent | 0 | 95.085 | pose_1: host pose_4→pose_4, 方向0.000°, 平移4.837mm |
| 8 | feasible_volume_descent | 0 | 91.188 | pose_2: host pose_7→pose_7, 方向4.688°, 平移0.000mm |
| 9 | feasible_volume_descent | 0 | 86.512 | pose_2: host pose_7→pose_7, 方向4.396°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_1 | 0.000 | 1.474 |
| 1 | pose_2 | 0.000 | 1.484 |
| 1 | pose_3 | 0.000 | 1.628 |
| 1 | pose_4 | 0.000 | 1.628 |
| 1 | pose_7 | 0.000 | 1.715 |
| 1 | pose_27 | 0.000 | 1.135 |
| 2 | pose_1 | 0.000 | 2.000 |
| 3 | pose_1 | 0.000 | 2.000 |
| 4 | pose_1 | 4.837 | 68.271 |
| 5 | pose_2 | 0.000 | 5.000 |
| 6 | pose_2 | 19.348 | 70.584 |
| 7 | pose_1 | 4.837 | 0.000 |
| 8 | pose_2 | 0.000 | 4.688 |
| 9 | pose_2 | 0.000 | 4.396 |

<!-- compact-volume-results -->

真实体积优化继续结果：[最小支撑与所有方法](compact/README.md)。已检查最小材料 75.983cm³（原基线 85.193cm³）。本目录原基线文件完整保留，新网格和图见选中方法目录。
