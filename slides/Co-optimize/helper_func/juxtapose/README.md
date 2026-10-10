# Juxtapose：结构停滞时的离散跳步

连续Direction／XYZ Translation无法取得足够进展时，对失败pose或正在阻挡它的pose提出重叠落座候选。按退出方向接近程度考虑已有host／state，尝试原位置、重心对齐及有重叠的小偏移；方向种子包括guest、host和中间方向。新增位置生成贴合材料，统一扣除全部当前物体、工作锥和退出空间。

每轮共享96个廉价候选预算，最多3个完整分支，各再做最多2轮全组Direction／XYZ Translation修复。分支固定同一需求测度，比较修复后的整组收益与损失；必要时包含有界成对落座。一个state可承载多个pose，没有两个的容量限制。

额外需求修复还可保留当前支撑state，重新选择客体座位，再做世界XYZ梯度微调，允许airborne。原任务朝向、工作面和质心需求保持。候选覆盖与接触锁使用缓存，不做实体Boolean；选定布局后导出名义mesh。

可行后尝试恢复原注册／转动复用，减少新增材料。离散采样与连续梯度分别记录，有限预算不保证一次跳步解决或得到全局最小体积。

实现见 `whole_search/reuse_first.py`、`continuous_support/stable_gradient.py`、`continuous_support/fast_state_seat_gradient.py`。[完整流程](../../step4.2/fast_gradient_algorithm.md) · [几何演示](../../operation_demo/juxtapose/README.md)
