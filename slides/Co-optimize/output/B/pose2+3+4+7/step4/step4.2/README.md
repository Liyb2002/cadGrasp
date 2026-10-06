# pose2+3+4+7 Step4.2

PASS。Sampling 搜索大方向，物理反馈梯度局部调整。

- [过程图](process/process.png)：共享初始材料、原生退出方向、真实采样与梯度记录、保存的最终材料。
- [最终图](final_results/final_results.png)：同一支撑在全部 pose 下的安装与退出路径。
- [支撑 STL](final_results/support.stl)：单位毫米，原始物体坐标。

内部 NPZ、源模型、验收报告与搜索记录统一在 `data/`；[验收记录](data/report.json)。

真实优化过程：[MP4](process_actual/actual_sampling_process.mp4) · [GIF](process_actual/actual_sampling_process.gif) · [Poster](process_actual/actual_sampling_process_poster.png)。该组初始失败，第一条 `common floor cone` 全局提案瞬间跳变即成功；真实记录只有 1 个提案、0 个梯度步骤，没有方向插值。视频固定镜头：初始停留 2 秒，瞬间跳到成功 sample，最终停留 3 秒；画面只保留 pose 标签。视频时间仅用于展示，不代表优化计算耗时。最终画面直接读取保存的支撑 OBJ；复现数据在 [data/actual_sampling_process.json](data/actual_sampling_process.json)。

退出方向变化演示：[MP4](process/exit_direction_changes.mp4) · [GIF](process/exit_direction_changes.gif) · [Poster](process/exit_direction_changes_poster.png)。这是此前保存的几何演示；实际优化记录见过程图。
