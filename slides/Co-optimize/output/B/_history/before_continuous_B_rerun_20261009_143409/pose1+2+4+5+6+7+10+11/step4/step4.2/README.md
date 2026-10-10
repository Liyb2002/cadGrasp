# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 5.021 → 129.704 cm³。
转动支撑复用 3 个pose；Juxtapose 5 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 200211 | 5.502 |  |
| 1 | selective_juxtapose | 169489 | 10.226 | pose_11: host pose_11→pose_6, 方向19.122°, 平移37.892mm |
| 2 | selective_juxtapose | 96224 | 61.330 | pose_1: host pose_1→pose_1, 方向4.598°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向18.576°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向3.825°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向4.598°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向4.598°, 平移0.000mm；pose_7: host pose_7→pose_5, 方向51.596°, 平移132.675mm；pose_10: host pose_10→pose_10, 方向4.471°, 平移0.000mm；pose_11: host pose_6→pose_6, 方向4.513°, 平移0.000mm |
| 3 | direction_after_commit | 95200 | 61.330 | pose_2: host pose_2→pose_2, 方向6.689°, 平移0.000mm |
| 4 | direction_after_commit | 95024 | 61.330 | pose_1: host pose_1→pose_1, 方向4.600°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向4.984°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向3.831°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向4.600°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向4.600°, 平移0.000mm；pose_7: host pose_5→pose_5, 方向4.922°, 平移0.000mm；pose_10: host pose_10→pose_10, 方向4.475°, 平移0.000mm；pose_11: host pose_6→pose_6, 方向4.555°, 平移0.000mm |
| 5 | selective_juxtapose | 62347 | 104.266 | pose_4: host pose_4→pose_6, 方向37.196°, 平移147.983mm；pose_7: host pose_5→pose_5, 方向0.000°, 平移1.209mm |
| 6 | direction_after_commit | 62347 | 103.439 | pose_1: host pose_1→pose_1, 方向2.477°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向0.532°, 平移0.000mm；pose_4: host pose_6→pose_6, 方向0.706°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向2.477°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向2.477°, 平移0.000mm；pose_7: host pose_5→pose_5, 方向1.341°, 平移0.000mm；pose_10: host pose_10→pose_10, 方向2.267°, 平移0.000mm；pose_11: host pose_6→pose_6, 方向2.562°, 平移0.000mm |
| 7 | selective_juxtapose | 2407 | 139.132 | pose_2: host pose_2→pose_6, 方向133.341°, 平移202.770mm |
| 8 | direction_after_commit | 2389 | 139.132 | pose_7: host pose_5→pose_5, 方向5.000°, 平移0.000mm |
| 9 | selective_juxtapose | 8 | 197.663 | pose_10: host pose_10→pose_6, 方向8.436°, 平移129.344mm |
| 10 | selective_juxtapose | 0 | 176.869 | pose_7: host pose_5→pose_5, 方向20.659°, 平移9.749mm |
| 11 | feasible_volume_descent | 0 | 151.997 | pose_2: host pose_6→pose_6, 方向0.000°, 平移19.348mm |
| 12 | feasible_volume_descent | 0 | 129.608 | pose_4: host pose_6→pose_6, 方向0.000°, 平移19.348mm |
| 13 | feasible_volume_descent | 0 | 122.202 | pose_2: host pose_6→pose_6, 方向0.000°, 平移4.837mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。
