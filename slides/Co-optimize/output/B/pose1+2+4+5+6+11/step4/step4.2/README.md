# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 35.879 → 99.636 cm³。
转动支撑复用 4 个pose；Juxtapose 2 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 86914 | 41.269 |  |
| 1 | rotate_reuse_direction | 86780 | 41.692 | pose_2: host pose_2→pose_2, 方向1.964°, 平移0.000mm |
| 2 | rotate_reuse_direction | 86780 | 41.375 | pose_2: host pose_2→pose_2, 方向2.000°, 平移0.000mm |
| 3 | selective_juxtapose | 30544 | 64.063 | pose_1: host pose_1→pose_1, 方向3.325°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向3.126°, 平移0.000mm；pose_4: host pose_4→pose_6, 方向27.218°, 平移183.283mm；pose_5: host pose_5→pose_5, 方向3.499°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向3.499°, 平移0.000mm；pose_11: host pose_11→pose_11, 方向8.932°, 平移0.000mm |
| 4 | direction_after_commit | 27821 | 64.063 | pose_2: host pose_2→pose_2, 方向5.000°, 平移0.000mm |
| 5 | selective_juxtapose | 0 | 128.754 | pose_11: host pose_11→pose_6, 方向79.641°, 平移51.635mm |
| 6 | feasible_volume_descent | 0 | 119.957 | pose_11: host pose_6→pose_6, 方向0.000°, 平移4.837mm |
| 7 | feasible_volume_descent | 0 | 111.160 | pose_11: host pose_6→pose_6, 方向0.000°, 平移19.348mm |
| 8 | feasible_volume_descent | 0 | 106.362 | pose_4: host pose_6→pose_6, 方向0.000°, 平移1.209mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_2 | 0.000 | 1.964 |
| 2 | pose_2 | 0.000 | 2.000 |
| 3 | pose_1 | 0.000 | 3.325 |
| 3 | pose_2 | 0.000 | 3.126 |
| 3 | pose_4 | 27.363 | 84.546 |
| 3 | pose_5 | 0.000 | 3.499 |
| 3 | pose_6 | 0.000 | 3.499 |
| 3 | pose_11 | 0.000 | 8.932 |
| 4 | pose_2 | 0.000 | 5.000 |
| 5 | pose_11 | 19.348 | 0.000 |
| 6 | pose_11 | 4.837 | 0.000 |
| 7 | pose_11 | 19.348 | 0.000 |
| 8 | pose_4 | 1.209 | 0.000 |

<!-- compact-volume-results -->

真实体积优化继续结果：[最小支撑与所有方法](compact/README.md)。已检查最小材料 76.971cm³（原基线 99.636cm³）。本目录原基线文件完整保留，新网格和图见选中方法目录。
