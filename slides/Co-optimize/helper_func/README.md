# 共用工具

当前Step4.2由 `step4.2/stable_pipeline.py` 组织，使用全组需求梯度、离散落座与可行材料下降。

| 模块 | 当前职责 |
| --- | --- |
| `co_common.py`、`current_task.py` | 原任务读取、坐标变换、反力与实体基础工具 |
| `whole_search/` | 共享接触／材料增删缓存、落座候选、保存初始化与材料择优 |
| [continuous_support](continuous_support/README.md) | 全组积分目标、方向梯度、局部下降与有界自身状态接续 |
| [direction_choice](direction_choice/README.md) | Direction操作说明 |
| [translation](translation/README.md) | 世界XYZ梯度、地面投影、独立／共同移动及材料下降 |
| [juxtapose](juxtapose/README.md) | 离散host／座位候选与全组修复 |

XYZ Translation允许工件airborne，按实际高度切换工件地面反力。原质心需求、重力与第七方程保持。全部原需求通过后直接保存mask／反力，名义mesh仅作为固定布局显示与材料体积。

公开运行入口位于各step目录，绘图位于 `vis_func/`，相关测试见 [tests](../tests/README.md)。[当前算法](../step4.2/fast_gradient_algorithm.md)
