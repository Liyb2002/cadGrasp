# B：当前 whole Step4 结果

[全部组及当前状态](../output/B/step4_results.md) · [方向、sweep、过程及最终图片](../output/B/step4_index.html)

42个已有集合／180个pose实例的原模型可行性基线全部完成；13组初始承载失败已修复，四组使用单独记录的数值延续，不能宣称相同主预算42/42。之后对六组进一步做小体积双搜索对照，另加两份消融；六组都改善10.81–60.24%，逐组平均28.60%。详见 [真实材料对照](../output/B/compact_comparison.md)、[最小支撑图集](../output/B/compact_index.html) 和 [论文算法](paper_algorithm.md)。原42组基线保留；不是全部42组都重新跑了材料搜索。

当前结果从新版 whole Step3 初始化运行，全部 pose 共同参与 Direction、Juxtapose、Translation 搜索。真实最终支撑及原载荷、完整工作禁区、退出检查见每组的 data/report.json。未通过的候选只保留为诊断，不记作成功。

原七组结果及原报告保存在 output/B/_history/before_whole_step4_20261008；本文不再将旧分离保底或旧退出路径图与新结果混用。整件接地、连通和强度尚未验收。
