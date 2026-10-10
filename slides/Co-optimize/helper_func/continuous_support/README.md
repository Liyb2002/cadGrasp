# 全组需求梯度与共享几何缓存

当前搜索由 [stable_gradient.py](stable_gradient.py) 固定一个局部下降块及所有竞争Juxtapose分支的需求测度。所有pose的正权需求进入同一积分平方距离目标；[projection.py](projection.py) 将需求投影到原非负、无幅值上限的七维反力锥。

Direction使用独立、全组联合和共同转向梯度；Translation使用统一世界XYZ梯度，允许工件airborne。接地边界投影及物体地面反力切换见 [translation](../translation/README.md)。[progressive_gradient.py](progressive_gradient.py) 按需访问不同步幅与接触事件样本。

[geometry_cache.py](geometry_cache.py) 共享相对摆放的物体／sweep／工作锥查询，[strict_nominal_cache.py](strict_nominal_cache.py) 处理同位置近相切退出的接触锁。Direction更新阻挡列，Translation同时更新接触行和阻挡列；材料覆盖变化共同计入所有pose。

[releasable_blockers.py](releasable_blockers.py) 排除被owner自身禁区锁住的无效释放收益；失败guest、外部blocker和有界成对Juxtapose共享预算。state容量不受一次跳步参与pose数限制。

正常流水线先执行全组搜索，仅对需求仍未满足的组，从其本轮布局追加 [fast_state_seat_gradient.py](fast_state_seat_gradient.py) 的保持state落座和全组梯度。全部可行后，保持可行做Direction／XYZ材料下降。

[force_result.py](force_result.py) 直接保存选定布局已有的全部需求mask和反力，全部通过即PASS。固定布局名义mesh只用于显示／实体材料体积；没有几何微扰补救、末尾接触供给重建或重复需求求解。

七组8–10-pose新搜索7/7通过，其中6组初次、1组自身状态接续。[当前算法与预算](../../step4.2/fast_gradient_algorithm.md) · [结果](../../output/B/stable_gradient_xyz_force_v3_results.md)
