# 当前图片更新：base 最后考虑

Step4.1 的全部 30 个选组图片已按不含 Step3.3 接地环的 Step3.2 包裹支撑重画。算法、保存方向和力学结果未改；本轮无 base 图片不代表无 base 模型已通过受力。绘图进度见 [记录](data/base_deferred_render_batch.json)。

# B 当前算法与图片

算法以 [Co-optimize 当前说明](../../algorithm.md) 为准：Step4.1 求共同或相近合法方向，Step4.2 默认从该状态做 contact-recovery。本次文档更新没有重新运行算法或更新成功率。

示例 **pose1+2+4+6** 的最新图片：

- [exit direction](pose1+2+4+6/step4/step4.1/exit_directions.png)：横向四格，pose1、pose2、pose4、pose6。
- [sweep](pose1+2+4+6/step4/step4.1/exit_sweeps.png)：同样横向四格。
- [direction space](pose1+2+4+6/step4/step4.1/direction_space.png)：pose1 六方向，无文字；四个合法、两个向下非法；切除支撑留空。模型放大 50%，图片总尺寸不变。
- [direction coverage](pose1+2+4+6/step4/step4.1/direction_coverage.png)：兔子上四个淡色半球，无箭头，重叠颜色加深。
- [初始化支撑](pose1+2+4+6/step4/step4.1/final_result.png)。

Step3.3 接地环扩大后，30 个选组均完成支撑构造；物理诊断不作为该阶段否决条件，见 [接地环记录](data/ring_expansion_results.json)。下游各组的完成状态与检查数据须读取各自报告，不能把旧批次统计套用于新输入。图片整理和文档更新不构成新的力学或几何验收。

旧的选组表、17 组停在 Step3.3、旧支撑通过率和 shared 批次均属于历史记录，保存在 [原索引](../../data/documentation_history/before_current_version_20261006/Co-optimize/output/B/README.md)。旧索引中的相对链接按其原始位置解释。
