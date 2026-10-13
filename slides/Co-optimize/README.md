# Co-optimize：多配置共享刚性支撑

最新Step5.2整块圆整删减已重跑全部52组Step5.2／5.3，均通过；Step4、Step5.1和全部pose角色冻结。删减及成品的整体BBX逐组与上一版一致，合计351.890791／432.466717L；最大差仅7.105e-15L。保留较宽的原材料带，替换逐格删减；51组采用6 mm圆整范围，1组保守保留整块裁边主体。每组`step5.2`／`step5.3`标准图已更新，新增同视角`comparison.png`，无HTML。[算法说明](aesthetic_pruning_algorithm.md) · [全部52组数量、BBX和前后图](aesthetic_pruning_results_20261013.md) · [本次BBX Evaluation](data/results/B/evaluation_aesthetic_pruning_all52_20261013_01_bbx.md)。下面的连通主体搜索与历史结果保留原数值身份。

最新连通主体流程已完成全部52组／263个object pose：Step4与Step5.1／5.2／5.3均52/52通过，最终复用123／Juxtapose140，123个支撑安装pose。BBX合计：无支撑254.408 → Step4主体371.058 → 删减351.891 → 长脚成品432.467L；成品相对Step4净增加16.55%。51组删减缩小BBX，31组成品增加、21组减少；最大增幅仍需关注。1882张当前PNG已核对，每组在自己的step5.2／step5.3目录，无HTML。[算法与验收](connected_core_algorithm.md) · [全部52组BBX及图片](data/results/B/evaluation_connected_core_postprocess_all52_20261013_final_bbx.md) · [全部组图](output/B/all_bbx_connected_core_postprocess_all52_20261013_final.png) · [执行、重试与完整审计](connected_core_results_20261013.md)。包含独立冷启动重试与本轮候选精修，不声称第一轮52/52或相同计算量。以下数字保留此前独立历史批次。
[与上一版V8及早期暖启动52/52版的全部组对比](data/results/B/comparison_v8_connected_core_20261013.md)：共同49组成品BBX下降6.47%，Step4上升1.18%；完整通过49→52，但新流程包含更多重试计算。


最新V8布局的Step5.1→5.2→5.3→Evaluation已完成：50个Step4通过集合中，49个取得实际完整实体证书，另2个原Step4失败集合保留失败。最终49组共238个object pose、129个支撑安装pose，原位复用129／Juxtapose109。用户关注BBX体积：相同49组Step4名义324.424→完整413.720L（+27.52%），无支撑基线239.082L；相对Step4已构造完整参考，BBX+0.24%。1275张逐组阶段PNG已更新。[全部52组BBX、角色数量与图片](data/results/B/evaluation_dynamic_postprocess_all52_20261012_01_bbx.md) · [结果与来源说明](dynamic_postprocess_results_20261012.md)。`pose1+2+4+5+6+7`的真实接触复查未通过，不能沿用Step4点云PASS当成成品证书。

最新实验按[动态换座的32候选算法V8](population_soft_algorithm.md)执行：所有pose注册原位初始化，不指定Juxtapose身份；每槽位先离散或差分变异，再一次调整。恢复旧摆放、方向及高度修复能力，N/3只影响换座抽样。入口为[run_population_soft.py](step4.2/run_population_soft.py)。全部52组／263个object pose已完成冷启动Step4，最多8轮：**50组完整通过、2组原受力失败**，无运行错误。通过组共244个object pose、131个实际支撑安装pose。469张PNG直接在`output/B/[set]/step4.2/`。[逐组Evaluation与图片](data/results/B/step4_population_soft_all52_20261012_final.md) · [结果与调整说明](population_soft_results_20261012.md)。

上一次固定32候选冷启动是33/52；本轮新增通过17组，无原通过转失败。共同通过33组的完整BBX合计516.607→262.283L，材料55897.107→6373.931cm³。二者同为32槽位、8轮上限，但不代表计算工作量相同。此前用户接受的暖启动52/52结果单独保留；本轮没有读取其布局，也不宣称体积最优。[旧冷启动结果及存档图片](data/results/B/step4_population32_all52_20261012_combined.md)。

当前Step5顺序为：Step5.1地面需求→Step5.2接触材料删减→Step5.3完整实体构造→Evaluation。原Step5.2构造代码移到Step5.3。新图直接放在每组`output/B/[set]/step5.2/`和`step5.3/`；旧两阶段输出按用户要求清理。下方52组统计保留此前确认的历史数值身份。

新Step5后处理已跑全部52组、263个pose：Step5.2 **52/52通过**，Step5.3 **51/52通过**。`pose1+2+4+5+6+7`的连接成品未保留全部原需求，图中标红为失败候选。[本轮逐组记录](data/results/B/pruned_all52_20261012_results.md) · [删减总览](output/B/all_pruned_supports_pruned_all52_20261012_final.png) · [完整支撑／失败候选总览](output/B/all_supports_pruned_all52_20261012_final.png)。本轮复用保存布局和Step5.1，没有重新搜索Step4或重跑Evaluation；与population32搜索分开记录。

用户已于2026-10-12确认满意当前全部52组结果。[已确认结果与讨论起点](accepted_results_20261012.md)记录最终批次、Evaluation、逐组object／支撑pose数量及图片存档，作为接下来讨论的参照。

当前完整算法、物理边界、Step4／Step5 公式与 Evaluation 指标统一维护在[仓库根 README](../../README.md)。实现采用 whole 联合优化：Direction、世界 XYZ Translation 与离散 Juxtapose，结合动态 base Oracle；允许 airborne，保持原任务朝向与需求。

最新全部52组已完成，每组图片直接放在 `output/B/[set]/step5.2/`，不使用 HTML。其中 `overview.png` 为完整支撑，`all_poses.png` 为全部任务姿态，`pose_N.png` 为单个物体姿态，`evaluation.png` 从左到右为无支撑／加 Step4／加完整 Step5.2；`illegal/` 保留目录前缀。共419张直接图片；[示例姿态图](output/B/pose1+2+4+19/step5.2/all_poses.png) · [示例Evaluation图](output/B/pose1+2+4+19/step5.2/evaluation.png) · [52件支撑总览](output/B/all_supports_material_output_all52_20261012_final.png)。全部52组共263个 object pose、112个支撑安装pose；每组只有一件共享支撑实体。[完整Evaluation与逐组pose数量](data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](data/results/B/evaluation_material_output_all52_20261012_final.csv)。最终批次为`material_output_all52_20261012_final`，保留本轮搜索与单组构造修正的来源；1150张阶段图及旧实验图片保留。

| 入口 | 内容 |
| --- | --- |
| [新算法设计](representation_algorithm.txt) | 共享材料表示、固定sample搜索、完整成本评价，以及已集成部分与后续材料规划器的接口 |
| [新混合布局搜索](step4.2/run_material.py) | 原位复用／Juxtapose组合采样、原接触端口匹配、地面高度提案和搜索内成品构造；原物理与法向Oracle保持 |
| [Step3.1](step3.1/README.md)／[Step3.2](step3.2/README.md) | 整组注册、贴合包裹、扣除工作禁区及单独展示 |
| [Step4.1](step4.1/README.md) | 共同／相近退出初始化 |
| [Step4.2](step4.2/README.md) | 连续梯度、离散跳步、动态 Oracle 与可行减材料 |
| [Step5](step5/README.md) | 全部地面需求凸包、确定性构造成品实体 |
| [Evaluation](evaluation/README.md) | 静态独立pose空间、共享state空间与三列整体BBX |
| [Operations](operation_demo/README.md) | Direction／Translation／Juxtapose 演示 |

历史 `base_oracle_v3`／`connected_base_v2` 批次全部52组通过，263个pose实例；10个≥7-pose集合均通过。该批次成品材料总量25,540.38cm³，相对Step4名义实体净增加8.88%；这些是旧条件下的保存结果，不代表新混合布局实验的结果。

[历史逐组数值表](data/results/B/evaluation_connected_base_v2.md)。各阶段 PNG 保留在对应集合和实验目录，B 目录不再提供 HTML 浏览页。

`output/`保留各步最终可视化，B 不使用 HTML；布局、mask、需求、模型和报告在`data/results/`。候选帧、临时布局、日志及已验证的执行副本用完即删。物理输入在`objects/`，原字节保持不变。文件生命周期实现见[artifact_store.py](helper_func/artifact_store.py)，每组直接图片的发布入口为[publish_output_images.py](helper_func/publish_output_images.py)。
