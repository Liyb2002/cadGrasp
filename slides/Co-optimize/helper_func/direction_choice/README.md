# Direction choice

生产搜索固定物体位置和支撑state，对所有pose需求到当前原反力锥的加权平方距离求方向数值差分。改变一个方向也会影响其他pose，释放与新锁定接触均计入全组目标。

方向位于单位球面，用两个切坐标求差分并沿球面更新；普通探针角0.75°，比较多个角步幅，保持原合法半球。还提出全组联合梯度和共同转向，处理多个相近方向共同锁住接触的情况。接触平台允许少量独立／共同sampling，日志区分梯度与采样。

候选只更新接触锁和供力锥，不重建实体Boolean。局部下降块及其竞争Juxtapose分支共用需求点、权重和归一化尺度。全部原需求可行后，可继续保持可行的Direction材料下降。

实现见 [fast_gradient.py](../continuous_support/fast_gradient.py)、[stable_gradient.py](../continuous_support/stable_gradient.py)。保存直接复用通过mask；反力锥按实际布局是否airborne包含或移除工件地面反力。

梯度的内部损失评估现在使用 [r6_gradient.py](r6_gradient.py)。需求由
[objects 的精确前向 R6 公式](../../../../objects/r6_quadratic.md)生成；积分节点、权重和尺度沿用原算法。
同一个有效接触基底内，投影损失是解析二次式。微调后若基底接触仍可用，保留其原始可行系数和残差，只检查新增反力列的对偶条件；删除基底接触或违反新增列条件时才重新求解该节点。反力仍非负、无上界，保留第七方程。

R6 的载荷参数导数不是退出方向导数：绕 COM 的需求不依赖退出方向。
当前硬接触网格在接触事件之间有平台，因此方向操作继续使用原尺度的接触事件割线。
这个修改加速割线两端的全组收益／损失评估，保留搜索顺序、步幅、sampling 和全部原需求 PASS 判断。

[数学图与符号](../../operation_demo/direction/README.md) · [当前流程与公式](../../step4.2/fast_gradient_algorithm.md)
