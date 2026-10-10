# Step4.2：最终两工具布局

本页图片来自已完成的七组两工具实验，显示保存的完整最终蓝色支撑、灰色物体和各 pose 的最终退出方向。当前图片使用真实平移布局，并按该 pose 的原始安装坐标展示同一支撑。

- [最终支撑与全部 pose](final_result.png)：3×2 排列，带退出方向箭头。
- [退出方向](exit_directions.png)：按保存 pose 顺序横向排列。
- [退出扫掠路径](exit_sweeps.png)：透明显示实际直线路径的前100 mm，完整验收路径为500 mm，与 Step4.1 展示规则一致。

面板顺序：pose_1, pose_2, pose_4, pose_5, pose_6, pose_11。

| pose | 最终布局 | 退出路径 |
|---|---|---|
| pose_1 | [图](pose_1_final.png) | [图](pose_1_exit.png) |
| pose_2 | [图](pose_2_final.png) | [图](pose_2_exit.png) |
| pose_4 | [图](pose_4_final.png) | [图](pose_4_exit.png) |
| pose_5 | [图](pose_5_final.png) | [图](pose_5_exit.png) |
| pose_6 | [图](pose_6_final.png) | [图](pose_6_exit.png) |
| pose_11 | [图](pose_11_final.png) | [图](pose_11_exit.png) |

当前数值来源：[report](data/two_tool_final/report.json)、[原始完整支撑](data/two_tool_final/remaining_support.obj)、[平移和方向](data/two_tool_final/layout.npz)。此前求解报告 data/report.json 与历史图保留，不代表本轮新展示结果。这里只绘图，未重新求解或改变实验支撑。连通、安装接地和强度的验收状态保持。

[完整退出路径（3×2）](full_exit_paths.png) · [完整退出路径（横向）](full_exit_sweeps.png)：显示完整 500 mm 扫掠；每个 pose 的完整路径另存为 pose_*_full_exit.png。
