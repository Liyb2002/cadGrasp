# Step4.2 采样与梯度混合算法

确认算法为 **sampling 搜索大方向，物理反馈梯度局部调整**。主入口是 [run.py](run.py)，使用 [solver.py](solver.py)。算法细节见 [algorithm.md](algorithm.md)。

Sampling 生成共同趋势并投影到各 pose 的合法退出半球；短梯度分支依据真实反力锥缺口，在方向切平面上用带约束的 SLSQP 调整方向。每次候选从 Step3.3 原始材料重新切除完整扫掠，允许恢复材料；真实实体和全部原始载荷决定最终验收。

B 的完整批次为 **28/30**，其中 2 组初始通过、26 组恢复。结果统一见 [output/B/README.md](../output/B/README.md)，每组位于 `step4/step4.2/`。每侧 1% 净空始终保留；连通、支撑接地覆盖与强度暂缓。

运行单组时传入 `--set`、Step4.1 的 `--directions` 及新的 `--out`。批次入口为 `run_batch.py`；默认新实验目录位于 `data/experiments/`。主结果不自动覆盖。

历史控制和其他物体测试在 `data/experiments/`，缓存在 `data/cache/`；均不属于 `output/`。原始结果迁移不构成对当前源码的新验收。

内部实现层位于 `helper_func/optimization/`；它们提供当前求解器使用的基类、采样和梯度方法，不是独立推荐算法。测试统一在 `tests/`，旧实验代码在 `data/code_history/`。

最终展示文件在各组的 `step4/step4.2/final_results/`，只包含 `final_results.png` 和 `support.stl`（毫米）。使用保存的实体和退出方向生成，不重新优化或改变原验收结果；渲染及导出记录放在该组的 `data/final_results.json`。生成命令：

```sh
.venv/bin/python slides/Co-optimize/vis_func/render_final_results.py
```

每组的 `process/` 保存过程图与演示；所有 NPZ、源模型和搜索记录集中在 `data/`。过程图生成入口：`vis_func/render_process.py`。验收报告位于该组 `data/report.json`。
