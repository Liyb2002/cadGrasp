# Step4.2：两个 operations

当前设计为 **Direction、Juxtapose**。二者共用最不满足的力／力矩需求与反力锥距离，更新所有配置的可用接触，再用真实全载荷评价候选。

- [Direction](direction/README.md)：保持当前相对布局，微调退出方向；数学包括最差反力差、接触几何与方向的求导、球面更新。保留 `pose1+2+4+6` 既有视频及数值导数检查。
- [Juxtapose](juxtapose/README.md)：用 Direction 同一个 B shape，固定支撑在 pose1 的摆放，展示 pose2 装入受阻、删冲突材料，再在所有不妨碍 sweep 的非工作面补同样的贴合支撑；含视频、一张最终 PNG 和真实几何记录。

Juxtapose 可以理解为直接提出目标布局的高端版 translation；微小 translation 不再是独立 operation。详细搜索与验收定义见 [Step4.2 算法](../step4.2/algorithm.md)。

原模型的接触反力非负、无界。接触面积是几何量，不能直接当成承载容量；接触位置和法向决定反力锥。新增体积、代理改善都不能替代全部原始载荷验收。

**实现与示例状态：** 当前生产求解器仍为旧 direction／translation 搜索，Juxtapose 有独立的删补几何演示，尚未接入 Step4.2 搜索或执行全部原始载荷验收。Direction 视频是既有示意动画，不是逐帧验收的梯度轨迹。[Translation（none_applicable）](translation/README.md) 的最终布局图和 1/3 身位动画保留为历史材料变化参考，不作为第三个 operation，也不是 Juxtapose 的结果。`grasp/` 已按用户要求删除。
