# Co-optimize：多次装卸、多配置共享支撑

同一件刚性支撑服务多个使用配置。每次抓取物体、沿该配置的路径装入、执行任务并取出；空支撑可以换摆放。一个支撑姿态可对应多个物体姿态，进一步可服务不同物体，同一支撑也可有多个摆放姿态。当前先完成单物体版本，多对多分配和跨物体共享是后续扩展。

核心是共同决定哪些材料提供承载、哪些空间必须为物体和装卸路径留空。详见 [研究定义与当前算法](algorithm.md)。

- [Step3.1](step3.1/README.md)：新的 **whole 初始化**，合并原注册和包裹。每组全部 pose 注册到首个参考 pose，生成贴合支撑，扣除全部工作面的完整向外禁区。
- [Step3.2](step3.2/README.md)：只画整组工作禁区的并集，复用 pose_set_search 的圆弧显示边界和 8 个环绕等轴测视角。
- [B 初始化结果](output/B/README.md)／[图片浏览](output/B/index.html)：沿用已有 42 个集合，原物体、工作面和载荷不变。
- [Step4.1](step4.1/README.md)：共同／相近合法装卸方向初始化。
- [Step4.2](step4.2/README.md)：新设计包含 **Direction、Juxtapose、Translation**；微调方向，或将停滞 pose 重叠到已有支撑位置并补材料，再微调位置，比较材料和占地代价。
- [Operations](operation_demo/README.md)：Direction 数学与示例、Juxtapose 几何视频、6 秒 Translation 和 8 秒 Combined。

本轮实现的是新的初始化。B 当前 Step3 目录只包含新 Step3.1/3.2；旧 Step3.3 和旧初始化完整保存在 [历史目录](output/B/_history/before_whole_initialize_20261008/)。原 Step4 文件保持原样，属于旧初始化下的结果，后续需要从新支撑重新生成。初始化时保留全部原始载荷的受力诊断，当前重合布局的失败不丢弃整个 pose set；退出、接地和 whole 搜索尚未在这批新支撑上运行。

**生产 Step4.2 仍是旧 direction／translation 搜索，新的 whole Direction＋Juxtapose＋Translation 尚待接入。** 独立的 [pose_set_search 测试](../pose_set_search/README.md)已经完成。本轮按 whole 路线建立生产初始化，不运行 incremental。旧小步搜索七组结果为 **2/7通过，5/7未通过，没有接受平移，没有分离保底**，见 [旧实验报告](data/experiments/multistep_two_tool_v2_B_20261007/README.md)；它不代表新的 whole 算法结果。

在项目根目录运行：

```sh
# 新初始化：沿用 B/output 的全部已有集合，生成两阶段图片
.venv/bin/python slides/Co-optimize/step3.1/run.py B --jobs 2
# 只重新画 Step3.2 禁区；不修改支撑
.venv/bin/python slides/Co-optimize/step3.2/run.py B
# 核查导出网格与完整工作禁区、原始输入及旧 Step4 文件
.venv/bin/python slides/Co-optimize/helper_func/audit_whole_initialize.py B
.venv/bin/python slides/Co-optimize/step4.1/run.py --help
.venv/bin/python slides/Co-optimize/step4.2/solver.py --help
.venv/bin/python slides/Co-optimize/step4.2/run_placement_sampling.py --help
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

批量新实验使用全新 `data/experiments/` 目录，保留原始 Step4.1 和已发布输出。现有旧搜索的批量默认 `multi-step`；单组 `solver.py` 默认历史 `contact-recovery`，使用旧多步幅搜索需显式 `--search multi-step`。这些命令尚不执行 Juxtapose。
