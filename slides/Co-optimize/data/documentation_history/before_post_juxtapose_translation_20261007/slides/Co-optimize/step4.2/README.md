# Step4.2：Direction 与 Juxtapose

从 Step4.1 延续共享支撑设计，使用两个 operations：

- **Direction**：在当前相对布局上微调退出方向，释放有用接触，同时考虑新增锁定与其他 pose 的接触损失。
- **Juxtapose**：为停滞 pose 选择另一个 pose 的支撑摆放，直接提出重叠终点，并确定缺哪些接触、材料在哪里补、怎样连接、补多少。

Juxtapose 是以目标重叠布局确定位移的高端版 translation；微小 translation 不再是独立 operation。它修改最终设计布局，不打通重定位轨迹。两种候选都要检验全部配置，保护已通过的原始载荷，比较改善与材料／占地代价。见 [算法定义](algorithm.md)。

**实现状态：当前代码仍运行旧 direction／translation 搜索，Juxtapose 尚未实现。** 批量入口默认 `multi-step`，单组 `solver.py` 默认历史 `contact-recovery`；本次文档修改不改变数值入口。旧实现与运行命令见 [历史说明](../data/documentation_history/before_direction_juxtapose_20261007/slides/Co-optimize/step4.2/algorithm.md)。

独立的 [Juxtapose 几何演示](../operation_demo/juxtapose/README.md) 已提供：固定 pose1 支撑，尝试装入 pose2，删冲突材料，再沿用 Step3.2 规则补新 pose 所有不妨碍 sweep 的贴合包裹。演示检查物体、完整退出、工作面与全部可用包裹覆盖；尚未接入上述搜索或执行全部原始载荷验收。

[旧七组实验](../data/experiments/multistep_two_tool_v2_B_20261007/README.md)为 **2/7 通过、5/7 未通过，没有接受平移**，不是 Juxtapose 的结果。原始输入、模型及已发布图像保留。完整夹具连通、安装接地合法性与强度尚未验收。
