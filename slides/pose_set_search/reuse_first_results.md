# 转动复用与材料体积：新结果

目标明确为在全部原始力／力矩、退出和工作面约束满足后减少实体材料体积，支撑摆放次数不加惩罚。采用共同／相近物体相对退出方向初始化，优先 Direction 和转动复用；失败任务／尚未接纳的新增任务才尝试选择性 Juxtapose，并继续 Translation 与 Direction。

| 集合 | 顺序 | 材料 cm³ | 转动复用 | Juxtapose | 最终原载荷验收 |
|---|---|---:|---:|---:|---|
| 1–7 | joint | 109.031 | 5 | 2 | 229,376 / 229,376 |
| 1–10 | joint | 116.210 | 6 | 4 | 327,680 / 327,680 |
| 1–10 | incremental | 110.171 | 9 | 1 | 327,680 / 327,680 |

[最终图和各 pose 退出路径](../Co-optimize/output/B/step4.2/README.md) 使用 Step4.1 原有画法，每个面板中蓝色支撑形状完全一致，只用保存的刚体摆放改变其世界位置／朝向。实际物体、完整退出和工作禁区均在构造验收时检查。

增量 1–10 的前六个任务不用 Juxtapose；第七个任务的自身载荷通过但妨碍旧接触，因此“新增任务自身通过”不能把它排除出 Juxtapose 候选。将 pose7 放入 pose6 的支撑摆放后整组通过，后面三个任务保持转动复用。全部十个加入前缀通过。最终材料下降阶段只接受整组仍满足全部原载荷的更小材料候选。

可行后材料体积进一步下降：1–7 为 119.130 → 109.031 cm³，1–10 joint 为 125.390 → 116.210 cm³，incremental 为 136.986 → 110.171 cm³。本次增量结果比整体结果小约 5.2%。这是具体结果比较，不能推广为某个顺序对所有集合更优，也不能称全局最小。

原始输入及生产 Co-optimize 源码哈希保持一致。最终实体直接保留精确 worker 已验收的浮点边界，避免再次读入几何库造成简化。最终导出 OBJ 的体积与报告体积一致；另从导出实体重提接触，保存并回代总计 884,736 条原始载荷的非负反力，含第七维条件。所有反力回代残差远小于原 2e-9 容差。

上述结果分阶段延续原构造／可行检查点，并不是三次新的独立初始化运行。各报告保存来源与 SHA、构造历史、体积下降候选、原始输入和执行代码快照。对应最终数据：

- [1–7 joint](output/reuse_first_verified_v4/seven_chain/joint/report.json)，[全部反力](output/reuse_first_verified_v4/seven_chain/joint/pressure_audit.json)。
- [1–10 joint](output/reuse_first_joint_optimized_v5/pose1-10/joint/report.json)，[全部反力](output/reuse_first_joint_optimized_v5/pose1-10/joint/pressure_audit.json)。
- [1–10 incremental](output/reuse_first_incremental_optimized_v5/pose1-10/incremental/report.json)，[全部反力](output/reuse_first_incremental_optimized_v5/pose1-10/incremental/pressure_audit.json)。

历史结果保留在 [results.md](results.md)，不与新目标下的结果混淆。算法及运行入口见 [reuse_first_algorithm.md](reuse_first_algorithm.md)。数值失败／超时不作为物理不可行；中断、报告导出失败和临时前沿保留作诊断，不能替代最终验收报告。

验收仍沿用原阶段，不包括整件材料连通、安装接地、强度或机器人运动。19 项回归检查通过，覆盖体积优先级、全部载荷优先、转动复用注册关系、分支保护、新增任务选择、精确边界导出及原数值不变量。
