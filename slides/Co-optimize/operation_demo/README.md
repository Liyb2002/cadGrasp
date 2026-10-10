# Step4.2：Direction、Juxtapose、Translation

当前设计采用 **Direction、Juxtapose、Translation**。连续修改对全部pose需求到原反力锥的积分平方距离求数值梯度，同时计入收益和损失；Juxtapose是离散跳步，落座后再联合修复。可行后保持原载荷可行并减少实体材料。

- [Direction](direction/README.md)：保持当前相对布局，微调退出方向；唯一数学图展示全部需求积分、接触几何与球面数值差分更新。保留 `pose1+2+4+6` 既有视频。
- [Juxtapose](juxtapose/README.md)：用 Direction 同一个 B shape，固定支撑在 pose1 的摆放，展示 pose2 装入受阻、删冲突材料，再在所有不妨碍 sweep 的非工作面补同样的贴合支撑；含视频、一张最终 PNG 和真实几何记录。
- [Translation](translation/README.md)：在 Juxtapose 之后继续微调相对位置，比较实际支撑损失、可增加的接触及占地；视频演示 1/3 身位，6 秒。
- [Combined](combined/README.md)：8 秒展示 Juxtapose → Translation → Direction，三步使用同一个形状和连续的布局。

Juxtapose 提出重叠布局并补材料，代价可能包括扩大占地；Translation 继续改进这个布局，Direction 调整其装卸切除。一个支撑state可服务多个pose，没有两个的容量限制。详细搜索与验收定义见 [当前Step4.2算法](../step4.2/fast_gradient_algorithm.md)。

原模型的接触反力非负、无界。接触面积是几何量，不能直接当成承载容量；接触位置和法向决定反力锥。新增体积、代理改善都不能替代全部原始载荷验收。

**实现与示例状态：** 当前Step4.2已接入全组Direction／Translation梯度、离散Juxtapose及有界自身状态修复，七组8–10-pose新搜索全部满足原力／力矩需求，结果另有完整记录。本目录视频采用预设演示位移／角度，不是当前优化器的梯度轨迹。Translation支持世界XYZ和airborne；当前视频保留横移1/3身位的几何示例。
