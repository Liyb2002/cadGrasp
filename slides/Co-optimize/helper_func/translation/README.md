# Translation：世界 XYZ，允许 airborne

当前生产搜索对已显式Juxtapose的pose使用三个世界坐标自由度。物体原任务朝向保持，位置可向任意XYZ方向调整，最低点不得穿地；离地后可下降回到地面。原注册复用pose先保持位置，需要换位置时通过显式落座操作进入可移动集合。

## 一个三维梯度

[layout_update.py](layout_update.py) 对全组需求积分平方距离求同一世界梯度：

\[
g_i=(\partial_xL,\partial_yL,\partial_zL),\qquad
 t_i'=\Pi_{h_i\ge0}\!\left(t_i-\alpha g_i/\|g_i\|\right).
\]

这里 $h_i$ 是工件最低点高度。三轴使用同一差分尺度、范数与步幅池；着地边界对Z使用可行单边差分并投影下降方向。世界位移乘各host旋转转置转换为支撑局部坐标；不同host的共同移动也对应同一世界向量。

[proposals.py](proposals.py) 提出独立／协同三维梯度步和正反XYZ轴样本。普通差分为物体尺度的1/100，独立步幅为1/128、1/64、1/32、1/16、1/8身位；细修用更小尺度。多步幅按全组损失与原需求比较，sampling与gradient分别记录。

修改一个pose时更新其自己的接触行、它对其他pose的阻挡列，以及材料覆盖增删。共同移动可同时释放多个pose锁住的接触。平移改变落座终点，横移轨迹不切成滑道。

## airborne 的反力变化

物体最低点高于1e-9m时，删除工件自身四个物理地面反力列；回到地面后恢复。第七方程非负slack保留，缓存区分实际接地状态。原重力及绕质心的32768需求数组不变，不单独计算airborne需求。

所有pose的需求收益与损失共同进入 $L$。最终位置对系统—地面的力矩变化由Step5处理。[物理公式](../../../obj_supp/airborne_equations.md)

## 可行减材料与保存

[volume_descent.py](volume_descent.py) 在固定体积积分框内提出Direction和XYZ减材料候选，只接受材料下降且全部原需求仍通过的步骤。

选定结果后直接复用已有mask／反力。mesh导出保持布局不变，仅用于显示与实体体积；导出失败不改变力／力矩PASS，估计体积不能替换有实体体积的旧答案。

七组8–10-pose新搜索7/7通过，3个新pose实例离地。[结果](../../output/B/stable_gradient_xyz_force_v3_results.md) · [完整搜索](../../step4.2/fast_gradient_algorithm.md)
