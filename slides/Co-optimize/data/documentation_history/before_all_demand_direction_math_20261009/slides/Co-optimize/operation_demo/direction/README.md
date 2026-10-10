# Direction operation：用最差力／力矩引导退出方向微调

这个 operation 固定物体位置，只改变某个或几个 pose 的退出方向。目的不是统一向 +z，而是释放对当前最差载荷有用的接触，同时计入其他接触被新锁住的损失。

[此前 pose1+2+4+6 的视频](vis/pose1+2+4+6_exit_direction_changes.mp4)已经复制到这里。它从该组保存的 Step4.1 方向开始，上排 pose1、pose2，下排 pose4、pose6，每次改变一个方向。**后续方向是预设球面插值，不是梯度优化轨迹，也没有对动画各帧做受力和净空验收。** 它只展示方向改变如何影响共同剩余蓝色支撑；使用 Step3.2 包裹、不含 base。原始记录见 [source_animation_record.json](vis/source_animation_record.json)。

![原视频起始帧](vis/video_poster.png)

## 1．最大力／力矩差怎么计算

所有力表示施加在物体上的力。沿用当前任务保存的反力需求 \(w_{k\ell}\)，不再次翻转已经转换好的载荷符号。设物体特征长度 \(L_k\) 是原 mesh 最大 extent，原尺度为

\[
S_k=\operatorname{diag}(1,1,1,L_k^{-1},L_k^{-1},L_k^{-1}),\qquad
\widehat w_{k\ell}=(S_k w_{k\ell},0)\in\mathbb R^7.
\]

接触点 \(p\)、物体质心 \(c\)、物体外法向 \(n\)，单位压力反力 \(f=-n\) 产生

\[
a(p,n)=\big(S_k[f,(p-c)\times f],\ f_z\big).
\]

地面反力列第七坐标为零；再加入 \(a_N=(0,0,0,0,0,0,-1)\)。第七平衡式是 \(\sum_{\mathrm{head}}\lambda_i f_{iz}-N=0\), \(N\ge0\)，即现有的必要不上抬条件。它不替代最终落地、摩擦或抗倾覆验收。

将地面列、当前可用接触列和 \(a_N\) 组成列矩阵 \(A_k(d)\)，投影到无界非负反力锥：

\[
\lambda^*_{k\ell}\in\arg\min_{\lambda\ge0}
 \frac12\|A_k(d)\lambda-\widehat w_{k\ell}\|_2^2,
\qquad
r_{k\ell}=A_k(d)\lambda^*_{k\ell}-\widehat w_{k\ell}.
\]

\[
\ell_{k\ell}=\frac12\|r_{k\ell}\|_2^2,\qquad
(k^*,\ell^*)\in\arg\max_{k,\ell}\ell_{k\ell},\qquad
\mathcal L(d)=\max_{k,\ell}\ell_{k\ell}.
\]

最大差是七维尺度化残差，不是只取最大力或最大力矩分量。第七坐标也会影响最差载荷。在实际 B 求解中必须检查全部原始载荷；既有 `farthest_load` 用可行锥点距离作安全上界剪枝，未把少数代表载荷冒充全量最大值。几条同时并列最差时，应检查这些活跃载荷；只对其中一条下降不能保证全局最大值下降。

投影系数可能不唯一，投影点和残差唯一。需检查投影 KKT 条件，求解未收敛不能作为下降证据。

## 2．力／力矩 → 接触区域 → direction：哪里能求导

### 接触区域是集合，不能只用一个面积数

设原允许接触面为 \(\Gamma_k\)，pose \(j\) 完整退出扫掠为 \(Q_j(d_j)\)，相应锁定接触集合为 \(B_{kj}(d_j)\)。可用区域是

\[
C_k(d)=\Gamma_k\setminus\bigcup_j B_{kj}(d_j).
\]

包括自身和所有其他 pose；初始物体占据、退出扫掠、合法接触朝向均需考虑。释放量与新增锁定量分别是

\[
\Delta A^+_k=|C_k(d')\setminus C_k(d)|,\qquad
\Delta A^-_k=|C_k(d)\setminus C_k(d')|.
\]

一个 blocker 不再覆盖，其他 blocker 仍覆盖时，不算释放。新方向锁住的旧接触必须扣除。

**面积相同的两个区域可以具有完全不同的力矩臂和法向，因而不同的反力锥。标量 \(|C|\) 不能决定 \(\ell\)，一般不存在一条精确的“\(\partial\ell/\partial|C|\) × \(\partial|C|/\partial d\)”链。** 当前反力无限幅，把非零面积乘在每根反力列上不会改变非负锥，因此不能用这种乘法伪造力学面积梯度。

如果接触拓扑固定，边界及决定反力锥的极值接触位置平滑变化，且局部投影活跃集合稳定，则可讨论对**区域形状**的导数。令 \(\theta\) 是方向的局部坐标，\(A(\theta)\) 包含随边界移动的反力列，在常规包络定理条件下：

\[
\frac{\partial\ell}{\partial\theta}
 =r^T\frac{\partial A}{\partial\theta}\lambda^*.
\]

这条式子包含接触位置和法向变化。它不是面积容量模型；面积增大但不改变锥的极值生成元时，锥距离可能完全不变。新接触出现、旧接触消失、遮挡并集拓扑改变时，平滑条件失效，可能出现平台或跳变。

### 可执行的连续引导链

先冻结最差载荷的残差，给潜在接触反力列评分：

\[
b_i=\frac{\max(0,-a_i^T r)}{\|a_i\|}.
\]

因为加入微小非负反力 \(\epsilon a_i\) 的损失一阶变化为 \(\epsilon r^Ta_i\)，\(b_i\) 表示反力方向的有用程度，**不是对面积的精确导数**。

令 \(s_{ij}(d_j)\) 是局部阻挡分数，正值表示被 pose \(j\) 锁定。实际模型可以用接触多边形对完整 leading-face sweep 棱柱的裁剪／距离；本目录小例子仅用初始朝向锁定 \(s_{ij}=n_i^Td_j\) 演示链式法则，不宣称检查完整 sweep。

\[
u_{ij}=\sigma(-s_{ij}/\tau),\qquad
\widetilde a_i=\Delta A_i\prod_j u_{ij},\qquad
J(d)=-\sum_i b_i\widetilde a_i(d).
\]

乘积是所有 blocker 并集的平滑补集；一个 blocker 明显锁定会使可用度很小。\(\Delta A_i\) 只作几何积分权重，没有进入反力容量约束。

\[
\frac{\partial J}{\partial\widetilde a_i}=-b_i,\qquad
\frac{\partial\widetilde a_i}{\partial d_j}
=-\frac{\widetilde a_i(1-u_{ij})}{\tau}
 \frac{\partial s_{ij}}{\partial d_j},
\]

\[
\nabla_{d_j}J
=\sum_i\frac{b_i\widetilde a_i(1-u_{ij})}{\tau}
 \nabla_{d_j}s_{ij}.
\]

这是 **接触代理目标对 direction 的梯度**，不是原始最坏锥距离的精确梯度。它能给出值得尝试的方向，是否改善仍由真实区域和原始载荷检验。若阻挡分数没有解析导数，可对局部接触裁剪／阻挡模型作中央差分；不能在每个导数探针里重建整个支撑。

只奖励新接触也可能破坏旧接触。更完整的引导可加入当前承载接触的保护权重，但最终仍必须检查所有新旧接触及全部载荷；本例即使代理下降，也同时发生释放和新增锁定，未声称真实力学改善。

## 3．球切平面更新与真实回溯

每个单位方向 \(d_j\) 选择正交切平面基 \(E_j\in\mathbb R^{3\times2}\)，\(E_j^Td_j=0\)。坐标梯度与回缩为

\[
g_j=E_j^T\nabla_{d_j}J,\qquad
z_j=-\eta g_j,\qquad
 d'_j=\frac{d_j+E_jz_j}{\|d_j+E_jz_j\|}.
\]

信赖步限制 \(\|z_j\|\le\tan(1^\circ)\)，实际角度为 \(\arctan\|z_j\|\)。若 pose 原世界竖直方向映射到共同坐标后为 \(h_j\)，还要求 \(h_j^Td'_j\ge0\)。非法方向缩步或重新求合法切平面步，不能默认改为 +z。

实际操作的候选回溯使用 \(\eta,\eta/2,\eta/4,\ldots\)：

1. 用局部接触模型检查释放、新锁定及代理改善；没有改善则拒绝。
2. 对有希望的候选才构造实际剩余支撑，检查完整名义退出、原接触核以外的 1% 净空、工作区和脱离终点。
3. 重算所有 pose 原始载荷的反力锥，选择实际全局最坏距离改善的候选；不能只检查冻结的最差载荷。
4. 最差载荷变化后重选目标与接触评分。全部原始载荷满足才算当前 force/exit 范围成功；完整夹具连通、落地和强度仍另验。

几何平台时，可尝试更合理的局部接触边界事件，或交给 [Juxtapose operation](../juxtapose/README.md)，尝试在另一个 pose 的支撑摆放上重叠放置并补材料；零梯度不能证明不存在解。

## 本目录可运行核对

在仓库根目录运行：

```bash
MPLCONFIGDIR=/tmp/cadgrasp-direction-mpl .venv/bin/python slides/Co-optimize/operation_demo/direction/code/demo.py
```

[代码](code/demo.py)复用现有七维 `cone_projection` 和 `missing_values`，用两个 blocker、四个潜在接触核对平滑并集梯度。例子是合成数据；不重跑 B，也不构造完整夹具。

![解析梯度、差分和释放／新增锁定](vis/derivative_check.png)

[数值记录](vis/derivative_check.json)：解析切平面导数与中央差分最大误差约 \(1.6\times10^{-15}\)；一度候选平滑释放约 2.146 mm²、新锁定约 1.615 mm²；代理目标下降。正数缩放反力列的投影损失变化约 \(1.1\times10^{-19}\)，核对面积权重不能成为无界反力锥的容量。**这些是导数核对结果，未证明该候选对真实 B 载荷有效。**
