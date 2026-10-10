# Co-optimize：多次装卸、多配置共享支撑

同一件刚性支撑服务多个使用配置。每次抓取物体、沿该配置的路径装入、执行任务并取出；空支撑可以换摆放。一个支撑姿态可对应多个物体姿态，进一步可服务不同物体，同一支撑也可有多个摆放姿态。当前先完成单物体版本，多对多分配和跨物体共享是后续扩展。

核心是共同决定哪些材料提供承载、哪些空间必须为物体和装卸路径留空。详见 [研究定义与当前算法](algorithm.md)。

- [Step4.1](step4.1/README.md)：共同／相近合法装卸方向初始化。
- [Step4.2](step4.2/README.md)：新设计包含 **Direction、Juxtapose** 两个 operations；微调退出方向，或将停滞 pose 直接重叠放到已有支撑位置，确定材料在哪里补、怎样连接、补多少。微小 translation 不再是独立 operation。
- [Operations](operation_demo/README.md)：Direction 数学与示例、Juxtapose 固定摆放的删补几何视频；translation 标记 `none_applicable`，保留为历史参考。

Step4.1／4.2 已恢复为实际代码目录，原始输出路径和结果保留。**当前代码仍运行旧 direction／translation 搜索，Juxtapose 尚未实现。** 新候选需同时计算新增材料与删旧支撑的影响，验收全部配置。旧小步搜索七组结果为 **2/7通过，5/7未通过，没有接受平移，没有分离保底**，见 [旧实验报告](data/experiments/multistep_two_tool_v2_B_20261007/README.md)；这不是 Juxtapose 的证据，也尚未完成整件支撑连通、安装接地合法性或强度验收。

在项目根目录运行：

```sh
.venv/bin/python slides/Co-optimize/step4.1/run.py --help
.venv/bin/python slides/Co-optimize/step4.2/solver.py --help
.venv/bin/python slides/Co-optimize/step4.2/run_placement_sampling.py --help
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

批量新实验使用全新 `data/experiments/` 目录，保留原始 Step4.1 和已发布输出。现有旧搜索的批量默认 `multi-step`；单组 `solver.py` 默认历史 `contact-recovery`，使用旧多步幅搜索需显式 `--search multi-step`。这些命令尚不执行 Juxtapose。
