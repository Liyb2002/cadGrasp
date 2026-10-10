# Co-optimize：多次装卸、多配置共享支撑

同一件刚性支撑服务多个使用配置。每次抓取物体、沿该配置的路径装入、执行任务并取出；空支撑可以换摆放。一个支撑姿态可对应多个物体姿态，进一步可服务不同物体，同一支撑也可有多个摆放姿态。当前先完成单物体版本，多对多分配和跨物体共享是后续扩展。

核心是共同决定哪些材料提供承载、哪些空间必须为物体和装卸路径留空。详见 [研究定义与当前算法](algorithm.md)。

- [Step3.1](step3.1/README.md)：新的 **whole 初始化**，合并原注册和包裹。每组全部 pose 注册到首个参考 pose，生成贴合支撑，扣除全部工作面的完整向外禁区。
- [Step3.2](step3.2/README.md)：只画整组工作禁区的并集，复用 pose_set_search 的圆弧显示边界和 8 个环绕等轴测视角。
- [B 初始化结果](output/B/README.md)／[图片浏览](output/B/index.html)：沿用已有 42 个集合，原物体、工作面和载荷不变。
- [Step4.1](step4.1/README.md)：共同／相近合法装卸方向初始化。
- [Step4.2](step4.2/README.md)：新设计包含 **Direction、Juxtapose、Translation**；微调方向，或将停滞 pose 重叠到已有支撑位置并补材料，再微调位置，比较材料和占地代价。
- [Operations](operation_demo/README.md)：Direction 数学与示例、Juxtapose 几何视频、6 秒 Translation 和 8 秒 Combined。

新的生产 Step4 已接入 whole Direction＋Juxtapose＋Translation，全部 pose 从开始就参与约束。优先转动空支撑复用；停滞时选择性重新落座并微调。可行后减少实体材料，最终以实际网格和原始全部载荷复核。旧 Step4 完整保存在 [历史目录](output/B/_history/before_whole_step4_20261008/)，新 Step3、原始 B 输入与完成的 pose_set_search 实验保持原样。

已有 42 组／180 个 pose 实例的可行性基线全部完成，13 组原 Step4.1 承载失败得到修复；四组另用有记录的数值延续，因此不宣称相同主预算下 42/42。**材料目标还需要继续搜索。** 新增 `step4.2/compact.py` 从实际可行基线运行单路径减材料和多个布局／修复分支搜索，允许可行后重新选 host 或恢复转动复用；最后取实际通过且实体体积最小的一项。原基线完整保留。

[论文可解释的完整算法](step4.2/paper_algorithm.md) · [实体体积对照](output/B/compact_comparison.md) · [最小支撑图集](output/B/compact_index.html)。这轮先对六个已有集合比较两种延续方法，另有两组消融，不冒充全部 42 组的新体积优化。两种方法共同运行使用两份预算，不宣称 beam 必然优于单路径。

另已完成旧 `pose_set_search` 同一批十个 **7–10 pose** 集合：[新旧体积表](output/B/large_pose_sets_comparison.md)／[全部过程与最终图](output/B/large_pose_sets_index.html)。全部从新版整组初始化开始；主预算6/10通过，四组追加自身布局搜索后均通过，随后各自运行同基线的greedy和beam，共20份真实结果。独立重放5,439,488个原始载荷，18,969,720条工作射线无阻挡。相对新版自身可行基线，实体体积合计减少15.19%；按相近名义构造比较，7/10不大于旧whole、5/10不大于旧两方法较小者，连续8/9/10 pose仍明显偏大。旧版名义网格与新版真实接触核/净空网格的体积不可直接视为同一验收口径。原42组及六组小规模优化保持原样。

[新的 B Step4 结果](output/B/step4_results.md)／[图片浏览](output/B/step4_index.html)：Step4.1 沿用退出方向及 sweep 画法，Step4.2 有无文字的固定等轴测过程图及最终各 pose 图。实时批次状态见 `output/B/data/whole_step4_progress.json`；最终汇总为 `whole_step4_batch.json`。整件支撑接地、连通与强度尚未验收。

在项目根目录运行：

```sh
# 新初始化：沿用 B/output 的全部已有集合，生成两阶段图片
.venv/bin/python slides/Co-optimize/step3.1/run.py B --jobs 2
# 只重新画 Step3.2 禁区；不修改支撑
.venv/bin/python slides/Co-optimize/step3.2/run.py B
# 新 Step4 两阶段及图片
.venv/bin/python slides/Co-optimize/helper_func/whole_pipeline.py B --jobs 3
# 或从已有新 Step4.1 继续优化
.venv/bin/python slides/Co-optimize/step4.2/run.py B --jobs 3
# 从实际可行结果继续优化体积；可用 --sets 选择已有组
.venv/bin/python slides/Co-optimize/step4.2/compact.py B --mode both --jobs 2
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

旧 `solver.py`、`run_placement_sampling.py` 和 `run_selected_to41.py` 保留供历史实验，不是当前 whole 生产入口。历史七组两工具实验 2/7 的结果不代表本轮新算法；几何演示视频也不替代本轮实际载荷结果。
