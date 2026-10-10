# 快速全组梯度与共享几何缓存

默认 `step4.2/stable_pipeline.py` 先执行固定积分的全组梯度搜索；未满足原需求时，从自己的布局调用 `fast_state_seat_gradient.py` 的保持state落座与联合梯度修复。后者继承 `state_seat_gradient.py`／`balanced_gradient.py`，保持支撑state和客体世界高度，梯度提出重叠座位，过滤已不满足必要接受条件的采样完整载荷检查。全部采样通过而实际网格未通过时，另作有界联合微扰。当前十组7–10-pose经8个冷启动成功和2个明确接续成功，累计10/10通过原真实检查；来源链预算和体积局限见算法文档。

当前快路径依次使用 `fast_gradient.py`、`adaptive_fast_gradient.py`、`cached_fast_gradient.py`、`descent_fast_gradient.py`、`efficient_fast_gradient.py`、`active_fast_gradient.py`；`progressive_gradient.py` 添加按需线搜索；`expanded_gradient.py` 展开所有连续坐标并扩大相对几何缓存，`loss_first_gradient.py` 以全组积分误差主导不可行下降。当前 `stable_gradient.py` 固定整次局部下降及所有竞争Juxtapose分支的评价点，`releasable_blockers.py` 排除自身仍锁住的接触，`blocker_gradient.py`／`paired_blocker_gradient.py` 用同一个预算考虑失败guest、外部blocker及有界两-pose跳步。每个支撑摆放state的pose容量不受跳步的参与pose数限制。公式、预算及局限见 [Step4.2算法](../../step4.2/fast_gradient_algorithm.md)。

所有 pose 的正权需求积分进入同一目标。`projection.py` 将耦合需求投影到原非负、无反力上限的7维锥；`geometry_cache.py` 共享相对摆放查询，`strict_nominal_cache.py` 补上相同摆放在几乎相切退出方向下被射线外移漏掉的 leading face。真实物理源码与容差不变。

局部方向和平移采用有限接触分辨率的数值割线；梯度线搜索优先，明确记录的有限 sampling 与离散 Juxtapose 用来逃离平台。原32768载荷检查保留为诊断；不可行连续步由同一积分损失下降决定提交，完全可行布局始终优先，可行减材料保持全部原载荷通过。真实接触网格、完整工作／退出、核心和净空检查决定发布；候选没有3维 Boolean。

以下完整接触多边形实现是前一条较慢的连续路径，保留作为历史及回归接口。

# 共享连续需求误差与接触边界

三个operation共用 `CoverageObjective`。读取原 `needs.json` 的作用点／方向／力大小耦合域、原工作面和30度工作角度；不修改保存的32768载荷。

当前修复目标是全组需求到原非负、无幅值上限反力锥的积分平方距离。`projection.py` 用小活动集批量求最近锥点，检查所有反力列的KKT条件，必要时回退到原7维投影求解；不引入面积压力或反力上限。求积位置和角度固定，力大小用3点Gauss节点，并检查零力与最大力边界。覆盖比例仅作诊断，旧 `coverage.py` 的力大小区间LP保留用于历史接口与回归，不在新梯度路径中调用。

`surface.py` 直接计算完整工作／全长退出棱柱与整个接触三角形的二维交集，避免只插值几个点而漏掉内部阴影。原位注册时使用已保存的Step3真实材料接触；重新落座后使用自身固定wrap的保守接触，减去其他物体的三角网格平面截面与全部工作／退出阴影。候选没有三维实体Boolean，不在已删掉的材料上虚构反力。边界顶点与法向生成原反力列，面积不当作容量。

两个连续操作对同一个全组损失做可行有限差分、方向导数核对、曲率步幅和Armijo回溯。原位pose不能平移；全部需求节点及边界满足后才减材料。求积和接触仍是搜索指导，不是整个连续域证书；最终沿用真实接触、核心／1%净空、全部工作／退出及原32768载荷检查。

公式和局限见 `../../step4.2/continuous_algorithm.md`。旧实验的成功率和体积不属于本次梯度修复结果。
