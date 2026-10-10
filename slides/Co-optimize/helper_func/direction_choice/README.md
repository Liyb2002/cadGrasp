# Direction choice

生产搜索固定物体位置和支撑state，对所有pose需求到当前原反力锥的加权平方距离求方向数值差分。改变一个方向也会影响其他pose，释放与新锁定接触均计入全组目标。

方向位于单位球面，用两个切坐标求差分并沿球面更新；普通探针角0.75°，比较多个角步幅，保持原合法半球。还提出全组联合梯度和共同转向，处理多个相近方向共同锁住接触的情况。接触平台允许少量独立／共同sampling，日志区分梯度与采样。

候选只更新接触锁和供力锥，不重建实体Boolean。局部下降块及其竞争Juxtapose分支共用需求点、权重和归一化尺度。全部原需求可行后，可继续保持可行的Direction材料下降。

实现见 [fast_gradient.py](../continuous_support/fast_gradient.py)、[stable_gradient.py](../continuous_support/stable_gradient.py)。保存直接复用通过mask；反力锥按实际布局是否airborne包含或移除工件地面反力。

[数学图与符号](../../operation_demo/direction/README.md) · [当前流程与公式](../../step4.2/fast_gradient_algorithm.md)
