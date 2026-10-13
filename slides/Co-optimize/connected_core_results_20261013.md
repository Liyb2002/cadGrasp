# 连通主体、只删减与长脚：2026-10-13 全部组结果

本轮新流程已完成全部 52 组／263 个 object pose。Step4、Step5.1、Step5.2、Step5.3 均为 52/52 通过；最终 Evaluation 在全部组完成之后重新运行。原任务、原始需求、朝向和物理容差保持不变。

最终共享支撑安装状态共 123 个；原位复用 123 个，Juxtapose 140 个。每组是一件连通的共享支撑，在不同最终安装状态中依次执行任务；123 不是实体件数。

| BBX 指标 | 合计 L |
|---|---:|
| 无支撑 | 254.408145 |
| Step4 连通接触主体 | 371.057878 |
| Step5.2 只删减 | 351.890791 |
| Step5.3 长脚成品 | 432.466717 |

删减减少 19.167086 L；长脚增加 80.575925 L；成品相对 Step4 净增加 61.408839 L（16.5497%）。成品相对无支撑增加 178.058571 L（69.9893%）。这些是 52 件不同夹具的指标合计，不能解释成一件实体的体积。

Step5.2 有 51 组实际缩小 BBX，没有组增加 BBX。完整 Step5 相对 Step4：31 组增加，21 组减少；逐组净增幅中位数 7.3444%。最大净增加为 pose1+2+4+19，+14.740825 L（+207.4237%）；其次 pose12+16+19+21+27，+8.716359 L。不能用合计掩盖这些长脚代价。

[全部 52 组 BBX、pose 数量与逐组图片](data/results/B/evaluation_connected_core_postprocess_all52_20261013_final_bbx.md) · [BBX CSV](data/results/B/evaluation_connected_core_postprocess_all52_20261013_final_bbx.csv) · [逐 pose 角色 CSV](data/results/B/evaluation_connected_core_postprocess_all52_20261013_final_poses.csv) · [全部组图](output/B/all_bbx_connected_core_postprocess_all52_20261013_final.png)。每组直接在 output/B/[set]/step5.2 和 step5.3 中有自己的模型图与 evaluation.png，illegal 前缀保留，没有 HTML。

## 算法与执行范围

Step4 在原有限合法包覆材料中取一个真实材料连通分量，删除孤块后对实际剩余接触检查全部原始需求。原七维反力锥同时约束物体平衡和共享支撑合力向下；长脚参考不能补救接触主体。Step5.2 只删除，先尝试缩小 BBX，再删除其它冗余材料；Step5.3 保留原来的合法长脚。每个通过结果均已在保存后的实际实体上认证。

第一轮新连通搜索为 43/52。难组通过分开的 32／48 槽位冷启动重试，以及本轮已保存候选的额外精修／重新验收恢复到 52/52；这不是第一轮 52/52，也不是所有组相同的计算预算。所有初始槽位都是原位注册，不预先指定 Juxtapose 身份，不读取此前接受的历史优化布局；N/3 仍只影响抽样。最终选择中两个组使用 48 个初始槽位，其余使用 32 个，合计认证 1696 个初始化槽位、3344 个已保存槽位事件。

九个连续 pose 的最后恢复来自本轮 32 槽位重试第 7 轮保存的候选：pose6 剩 5 条实际需求失败，继续一次平移调整后通过；额外实际接触测量 31 次。另一个九 pose 组来自本轮 48 槽位第 7 轮已受力通过的保存候选，合法域裁剪后重新验收；额外测量与原搜索轨迹分开记录。捕获的过程 NPZ、完整轮次轨迹、运行参数和控制器均保存为只读输入，原父实验继续或停止不会改变已认证输入。

Oracle 边界修复只是实际删除再认证，提案删除上限为主体的 1%，通过容差仍是原来的 1e-10 m³。组件提取记录属于裁剪前的诊断；针对超过阈值的诊断，最终实际主体已和按原输入重新构造的有限包覆域求差，实际新增量通过原容差。

已经取得全部选定 Step4 证书后，24 个多余探索／等待进程停止；其部分轨迹、初始化和日志全部保留，未完成控制组不当成完整冷启动基准或物理失败。停止记录与最终选定来源分别保存在 [interrupted_controls.json](data/results/B/data/system_connected_core_postprocess_all52_20261013_final/interrupted_controls.json) 和 [resolved_batch.json](data/results/B/data/connected_core_postprocess_all52_20261013_final/resolved_batch.json)。按实际最终 BBX 优先选择完全匹配的 Step4、删减、地面需求与长脚结果，没有混用不同布局。

[算法细节](connected_core_algorithm.md)。实验别名：Step4 为 connected_core_all52_20261013_final，后处理为 connected_core_postprocess_all52_20261013_final。

## 验证

52/52 实际保存实体检查通过：接触主体、删减主体和成品均一个材料连通分量；删减是原主体子集；成品保留必要主体；合法空间和物体侵入保持原容差。全部 8,617,984 条原始需求的保存证书通过，最终每个 pose 都有 32,768 条需求证书。Evaluation 本身从保存模型读取 BBX，没有重新求解受力。

54 个相关单元测试已通过。9,259 个历史／原输入数值文件保持字节一致；5,380 张历史 PNG、1,882 张当前 PNG、2,891 个冻结源码文件以及 10,965 项阶段源码存档均已核对。每个所选 Step4 的物理输入哈希、注册初始化、实际事件、最终阶段依赖和图片哈希全部通过。

[实体几何检查](data/results/B/data/system_connected_core_postprocess_all52_20261013_final/actual_solid_geometry_audit.json) · [完整审计](data/results/B/data/system_connected_core_postprocess_all52_20261013_final/saved_artifact_audit.json)。不宣称全局最优、统一八轮计算量或所有组都接近 N/3。

## 与上一版对比

[全部52组旧→新BBX、复用／Juxtapose／安装pose对比](data/results/B/comparison_v8_connected_core_20261013.md) · [逐组比较图](output/B/comparison_v8_connected_core_20261013.png)。以最近V8为对照，共同49组成品BBX413.719519→386.941085L（−6.4726%），Step4 BBX324.423771→328.262339L（+1.1832%）；33组变小、14组变大、2组基本不变。相同49组支撑安装pose／复用129→116，Juxtapose109→122。新版本第一轮43/52、旧版50/52；最终新52/52、旧49/52，额外重试和精修计算不能忽略。若对照更早已满意的暖启动52/52版，成品433.193199→432.466717L（−0.1677%），基本持平；支撑安装112→123、复用108→123、Juxtapose155→140。数字从保存报告读取，原来源哈希核对保持不变，没有重跑物理检查。
