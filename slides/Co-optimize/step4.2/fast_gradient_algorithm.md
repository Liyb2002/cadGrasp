# 快速 whole：全组梯度、Juxtapose、XYZ Translation

入口为 `run.py` → `stable_pipeline.py`。从各组保存的Step4.1开始，所有pose同时参与。连续变量是退出方向和已Juxtapose位置，离散变量是落座关系／支撑state；目标是全组承载可行后减少共享实体材料。

**世界XYZ Translation与airborne已实现并完成七组8–10-pose实验。** 三轴共用梯度、归一化和步幅，地面只施加最低点高度非负的物理边界。

## 需求与反力锥

布局记为 $x=(d_i,t_i,h_i)_{i=1}^n$，其中 $d_i$ 为退出方向、$t_i$ 为世界平移、$h_i$ 为离散host标识。每个pose的原始需求 $b_i$ 绕工件质心表达，包含重力；整体平移时相对质心的施力臂不变，需求数组不变。此处host标识与下文最低点高度 $H_i$ 分开。

接触位置、法向和实际地面接触生成原七维非负反力锥 $C_i(x)$。反力无幅值上限，面积只是几何量；保留原第七方程与非负slack。工件最低点 $H_i>10^{-9}\,\mathrm m$ 时删除四个工件物理地面反力列，着地后恢复。完整需求与代理缓存都包含此接地状态。

数据只保存15°／30°／60°三份角度需求，grounded／airborne共享绕质心需求。系统绕地面的力矩由最终位置决定，Step5按最终布局处理。[物理公式](../../obj_supp/airborne_equations.md)

## 同一个全组目标

用原需求测度 $\mu_i$ 和优化前固定尺度 $s_i$，最小化所有pose需求到当前反力锥的积分平方距离：

\[
L(x)=\frac{1}{2n}\sum_i\frac{1}{s_i^2}
       \int \operatorname{dist}(b_i,C_i(x))^2\,d\mu_i(b_i).
\]

已满足需求距离为零，改变布局后若变差就重新贡献正误差。每次计算所有受影响pose，收益和损失一起进入目标。投影使用NNLS并检查全部反力列的KKT条件。

正权求积混合固定边界节点与原32768需求的分层节点，两类等权。分层同时保留已满足和未满足区域，按原样本占比赋权；小于等于256项的失败区域全部保留。一个局部下降块及其竞争Juxtapose分支共用节点、权重和尺度；下一次外层决策再刷新分层。这是梯度指导的积分近似，PASS仍取全部原始需求。

## Direction

每个单位方向选两个球面切坐标，对 $L$ 做正反角探针差分，再沿球面指数映射更新并保持原合法半球。普通差分角0.75°，步幅0.25°、0.5°、1°、2°、4°、8°；细修用更小尺度。

独立、全组联合和共同转向都针对同一 $L$。共同转向处理多个相近方向锁住同一接触的情况。平台处允许少量明确标记的方向sampling。[数学符号与图](../operation_demo/direction/README.md)

## XYZ Translation

对已Juxtapose的pose，用同一个世界梯度

\[
g_i=(\partial_xL,\partial_yL,\partial_zL),\qquad
 t_i'=\Pi_{H_i\ge0}\!\left(t_i-\alpha g_i/\|g_i\|\right).
\]

三轴共用差分尺度、范数和步幅池；普通差分为物体尺度1/100，独立步幅为1/128、1/64、1/32、1/16、1/8身位。着地时Z用向上可行单边差分，并去掉负梯度中穿地的分量；离地后可升可降，下降最多返回地面。

世界向量分别乘各host旋转转置，转换为支撑局部坐标。协同Translation对所有可移动pose施加相同世界向量；正反XYZ轴样本使用同一组尺度。原注册pose先保持位置，显式落座后进入可移动集合。

[layout_update.py](../helper_func/translation/layout_update.py) 处理坐标、地面投影与差分；[proposals.py](../helper_func/translation/proposals.py) 提出独立／共同候选。Translation只改变设计终点，横移轨迹不挖成滑道。

接触开关有平台和跳变，Direction／Translation的梯度是有限分辨率数值割线。不同步幅线搜索确认进展，有限sampling跨过平台，日志分别记录gradient和sample。

## Juxtapose 的离散搜索

连续修复未能解决时，选择失败guest或确实阻挡可用接触的blocker，考虑退出方向相近的最多四个host。候选含原位置、重心对齐、重叠小偏移，方向种子含guest、中间和host方向。也可保留支撑state重新选择客体座位，随后用世界XYZ连续修复，允许airborne。

blocker排序用当前投影残差衡量释放反力列的收益。只统计有材料覆盖、任务允许且未被owner自身禁区锁住的接触；移走其他pose无法释放自身仍禁止的区域。

每轮共享96个廉价候选预算，最多3个完整分支。必要时纳入至多4个成对Juxtapose样本，至少一个进入原分支预算。每个分支再做最多2轮全组Direction／XYZ Translation，使用父布局固定测度比较。一个state可承载多个pose，成对跳步不是state容量上限。

## 快路径与提交

复用接触锁和材料覆盖计数。Direction更新相应阻挡列；Translation更新自己的接触行、它对其他pose的阻挡列，以及共享材料增删。接触可用条件是有材料覆盖，且没有任何pose的物体、完整工作锥或退出sweep锁住。某一个pose让开，其他pose仍锁住时不能计为释放。

共享相对摆放缓存复用物体／sweep查询；工作锥只在必要点上补查。近相切退出的leading face由严格名义接触锁处理。反力列集合相同则复用需求结果。候选不执行三维Boolean；材料在共同Sobol空间比较，体积域变更时重算竞争基准。

尚未全组可行时以 $L$ 的净下降提交，允许某些pose暂时退步；全部原需求可行的布局优先。零积分误差的求积盲区用原始需求计数处理。明显不满足接受条件的采样候选无需完整需求检查。

## 执行顺序与预算

1. 从保存Step4.1初始化，最多先做3轮全组连续修复。
2. 仍不可行时，触发上述有界Juxtapose分支；接受后再做2轮全组连续修复。初次搜索最多10轮结构跳步。
3. 全部原需求可行后，尝试恢复原转动复用，并做2轮保持可行的Direction／XYZ材料下降。[volume_descent.py](../helper_func/translation/volume_descent.py) 固定共同体积积分框，只接受更小且原需求仍全通过的候选。
4. 仅对仍有未满足需求的组，从自己的本轮布局追加最多8轮保持state落座和全组梯度修复。已经通过的组不进入接续。
5. 选定布局后，[force_result.py](../helper_func/continuous_support/force_result.py) 复用已有32768需求mask／pose和供力列，全部通过即PASS。固定布局导出名义mesh与图片，不再求解同一结果。
6. 最终材料择优只允许通过且导出实体材料更小的新答案替换旧答案；旧答案不作冷启动。mesh导出失败保留力／力矩PASS，但估计值不能替换有实体体积的旧答案。

普通图片是无文字等轴测mesh，工作禁区仅在Step3.2单独显示。支撑state分组按实际世界变换判定，容量不限。base、整体连通和强度是后续阶段。

## 七组完成结果

七组8–10-pose新搜索7/7通过：6组初次、1组追加自身状态梯度。3个新pose实例离地，最终择优中1个。原Step3／4.1、需求、历史结果与执行快照保留。

新名义材料721.55cm³，对旧713.01cm³大1.20%；3组更小、4组更大。最终采用3个新答案、保留4个旧答案，655.74cm³（−8.03%）。三进程整批含出图21.38min，搜索中位403.6s／组，mesh导出中位4.5s。旧流程搜索中位271.0s／组、整批42.8min；总耗时下降主要来自取消末尾几何验收，搜索本身未加速。

49项相关检查与存档核对通过。各pose全部原始mask通过，两个阶段各65份源代码／快照一致，exact调用和末尾重复求解均为0；核对只读取记录和保存mesh体积。

[结果与新／旧来源](../output/B/stable_gradient_xyz_force_v3_results.md) · [图集](../output/B/stable_gradient_xyz_force_v3_index.html) · [核对记录](../output/B/data/stable_gradient_xyz_force_v3/verification.json)
