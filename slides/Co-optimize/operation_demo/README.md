# Step4.2：Direction、Juxtapose、Translation

当前设计采用 **Direction、Juxtapose、Translation**。连续修改对全部pose需求到原反力锥的积分平方距离求数值梯度，同时计入收益和损失；Juxtapose是离散跳步，落座后再联合修复。可行后保持原载荷可行并减少实体材料。

![Step4.2 全组 operation 选择](step4_2_pipeline.png)

从保存的Step4.1开始，全组先做Direction；显式Juxtapose落座的pose还可做世界XYZ Translation。每个局部块固定全部需求的积分节点、权重和尺度，按中等、小、大步幅依次比较梯度候选组，组内按全组损失排序，找到可提交的改善就继续；梯度步无进展才访问有界接触事件sampling。连续修复仍未解决时，Juxtapose同时考虑失败pose与有用的阻挡pose，最多96个廉价候选、3个分支，分支使用同一父测度再做Direction／XYZ微调，选择净收益更好的分支。

全部原需求通过后，尝试恢复转动复用并做保持可行的材料下降。候选期间通过接触／材料增删和有效反力基底复用加速评分；原32768需求用于入选的变化布局，通过mask用于最终保存。固定布局导出mesh报告实体材料体积，最终可与已保存的可行答案择小。

图源：[draw_step42_pipeline.py](code/draw_step42_pipeline.py)。在仓库根目录重画：

~~~sh
MPLCONFIGDIR=/tmp/cadgrasp-step42-diagram-mpl PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/Co-optimize/operation_demo/code/draw_step42_pipeline.py
~~~

- [Direction](direction/README.md)：保持当前相对布局，微调退出方向；保留原数学图，新增 [R6 与实际梯度图](direction/vis/direction_gradient.png)，展示连续需求前向表达、投影基底复用和全组球面差分。保留 `pose1+2+4+6` 既有视频。
- [Juxtapose](juxtapose/README.md)：用 Direction 同一个 B shape，固定支撑在 pose1 的摆放，展示 pose2 装入受阻、删冲突材料，再在所有不妨碍 sweep 的非工作面补同样的贴合支撑；含视频、一张最终 PNG 和真实几何记录。
- [Translation](translation/README.md)：在 Juxtapose 之后继续微调相对位置，比较实际支撑损失、可增加的接触及占地；视频演示 1/3 身位，6 秒。
- [Combined](combined/README.md)：8 秒展示 Juxtapose → Translation → Direction，三步使用同一个形状和连续的布局。

Juxtapose 提出重叠布局并补材料，代价可能包括扩大占地；Translation 继续改进这个布局，Direction 调整其装卸切除。一个支撑state可服务多个pose，没有两个的容量限制。详细搜索与验收定义见 [当前Step4.2算法](../step4.2/fast_gradient_algorithm.md)。

原模型的接触反力非负、无界。接触面积是几何量，不能直接当成承载容量；接触位置和法向决定反力锥。新增体积、代理改善都不能替代全部原始载荷验收。

**实现与示例状态：** 当前Step4.2已接入全组Direction／Translation梯度、离散Juxtapose及有界自身状态修复，七组8–10-pose新搜索全部满足原力／力矩需求，结果另有完整记录。本目录视频采用预设演示位移／角度，不是当前优化器的梯度轨迹。Translation支持世界XYZ和airborne；当前视频保留横移1/3身位的几何示例。
