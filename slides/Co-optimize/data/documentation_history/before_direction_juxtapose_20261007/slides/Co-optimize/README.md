# Co-optimize：多次装卸、多配置共享支撑

同一件刚性支撑服务多个使用配置。每次抓取物体、沿该配置的路径装入、执行任务并取出；空支撑可以换摆放。一个支撑姿态可对应多个物体姿态，进一步可服务不同物体，同一支撑也可有多个摆放姿态。当前先完成单物体版本，多对多分配和跨物体共享是后续扩展。

核心是共同决定哪些材料提供承载、哪些空间必须为物体和装卸路径留空。详见 [研究定义与当前算法](algorithm.md)。

- [Step4.1](step4.1/README.md)：共同／相近合法装卸方向初始化。
- [Step4.2](step4.2/README.md)：方向与相对平移的多步幅候选、接触恢复和真实全部载荷验收。
- [操作示意](operation_demo/README.md)：direction 与 translation 的数学和几何示例。

Step4.1／4.2 已恢复为实际代码目录，原始输出路径和结果保留。当前小步搜索七组结果为 **2/7通过，5/7未通过，没有接受平移，没有分离保底**，见 [实验报告](data/experiments/multistep_two_tool_v2_B_20261007/README.md)。这些结果尚未完成整件支撑连通、安装接地合法性或强度验收。

在项目根目录运行：

```sh
.venv/bin/python slides/Co-optimize/step4.1/run.py --help
.venv/bin/python slides/Co-optimize/step4.2/solver.py --help
.venv/bin/python slides/Co-optimize/step4.2/run_placement_sampling.py --help
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

批量新实验使用全新 `data/experiments/` 目录，保留原始 Step4.1 和已发布输出。批量搜索默认 `multi-step`；单组 `solver.py` 默认历史 `contact-recovery`，要使用当前多步幅搜索需显式 `--search multi-step`。
