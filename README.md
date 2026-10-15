# cadGrasp

共同设计一件刚性支撑，依次支持同一工件的多个任务姿态。保持原任务朝向和原始力／力矩需求，联合优化退出方向、工件座位及支撑安装位置，再构造、删减并认证实际实体。

当前算法、物理条件、各阶段入口和运行命令统一见 [Co-optimize 算法说明](slides/Co-optimize/README.md)。

- `objects/`：原始工件、姿态、工作面和需求输入。
- `slides/Co-optimize/step*/`：初始化、联合布局搜索与实体后处理。
- `slides/Co-optimize/vis_func/` 和 `evaluation/`：绘图及静态空间／材料评估。
- `slides/Co-optimize/data/` 和 `output/`：保存数值、模型、证书、来源和阶段图片。
- [三个当前 Franka 视频](slides/Co-optimize/idea/)：视频直接放在目录根部，代码、轨迹、资源和分镜在 `assets/`。
- [物体预处理](codes/precompute_objects/README.md)与[仿真](codes/simulation/README.md)。

搜索中的 `.search_runtime/` 是临时缓存；正常完成后清理，异常时保留用于同次检查点恢复。同步可用 `--exclude=.search_runtime/` 排除它。当前独立 Step4 批次的实际进度见 [progress.json](slides/Co-optimize/data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/progress.json)。
