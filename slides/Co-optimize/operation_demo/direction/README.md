# Direction operation：对全部需求的差距求导

Direction 固定本次布局中的物体位置、支撑state与原始需求，只改变退出方向。现在的梯度针对所有pose的需求到反力锥的加权平方距离，包含释放接触的收益和新锁定接触的损失。梯度目标是全部需求的连续积分差距。

vis里只保留一张数学图：[direction_math.png](vis/direction_math.png)。沿用原图的 dir、\(\mathcal G\)、\(\mathcal C\)、\(D\)、\(F\)、\(\tau\) 与步幅\(\eta\)，补充求积权重、固定尺度与球面切坐标来对应代码的实际计算。

![全部需求的方向梯度](vis/direction_math.png)

## 1．全部需求的积分目标

\((F,\tau)\) 表示任务已经保存的反力需求，包含原重力与力矩，不再反转载荷符号。\(S_k\) 是原力／力矩尺度变换：力不变，力矩除以物体原特征长度。\(\mathcal C_k(\mathcal G_k(\mathrm{dir}))\subset\mathbb R^7\) 是当前可用接触、按实际接地状态存在的工件地面列与原不上抬列生成的无界非负反力锥；第七坐标保留原不上抬平衡式。单个需求的原始差距为

\[
D_k(F,\tau;\mathrm{dir})=
\operatorname{dist}^2\!\left(
  (S_k(F,\tau),0),\,
  \mathcal C_k(\mathcal G_k(\mathrm{dir}))
\right).
\]

图中原来的 \((F_{\mathrm{near}},\tau_{\mathrm{near}})\) 对应七维最近锥点的前六个力／尺度化力矩分量。距离还包括第七维残差，不能只拿前六维计算新的目标。反力无限幅；接触面积不是反力容量。

所有pose、所有需求共同进入积分平均：

\[
\boxed{
D_{\mathrm{all}}(\mathrm{dir})=
\frac{1}{2K}\sum_{k=1}^{K}\frac{1}{s_k^2}
\int D_k(F,\tau;\mathrm{dir})\,d\mu_k(F,\tau)
}
\]

\(K\) 是本组pose数；\(s_k\) 是优化开始前确定的原需求数值归一化尺度，过程中固定；\(\mu_k\) 是由原工作面积、加工方向角和力大小诱导的需求概率测度。系数\(1/2\)与代码的投影平方损失一致。\(s_k\) 和测度只用于评分，不改变原承载判定、反力模型或最终容差。

这里“全部覆盖”准确指降低**所有需求的总差距**：已满足需求的距离为零；原来满足的需求若被新方向破坏，会变为正误差并参与目标。未满足需求可以同时改善。它不是直接对离散覆盖率求导，也不是只奖励新增接触面积。

代码用正权求积近似这个积分：

\[
\widehat D_{\mathrm{all}}(\mathrm{dir})=
\frac{1}{2K}\sum_{k=1}^{K}\frac{1}{s_k^2}
\sum_\ell\omega_{k\ell}
D_k(F_{k\ell},\tau_{k\ell};\mathrm{dir}),\qquad
\omega_{k\ell}>0,\quad\sum_\ell\omega_{k\ell}=1.
\]

当前实现各占一半的固定边界求积和原32768需求的分层求积。分层同时保留当前已满足、未满足的区域，并按区域原始需求占比赋权；少量失败需求可全部保留。一个局部梯度块及其竞争Juxtapose分支使用同一组需求点、同一组权重。只在下一次外层决策刷新分层。求积仍是近似，不是整个连续需求域的证明。

## 2．方向如何影响所有需求

\[
\mathrm{dir}_j
\longrightarrow
\{\mathcal G_k(\mathrm{dir})\}_{k=1}^{K}
\longrightarrow
\{\mathcal C_k(\mathcal G_k(\mathrm{dir}))\}_{k=1}^{K}
\longrightarrow
D_{\mathrm{all}}(\mathrm{dir}).
\]

\(\mathcal G_k\) 仍是带接触位置和法向的可用区域／生成元集合，沿用原图的\((p_i,n_i)\)。接触必须有支撑材料覆盖，且不被任何pose的物体、完整工作锥或退出sweep锁住。一个pose让开后，另一个还锁住的接触不算释放；新方向锁住旧接触也计入损失。

所以调整\(\mathrm{dir}_j\)时，必须重算所有受影响的\(\mathcal G_k\)与全部pose目标。不能只看第\(j\)个pose，也不能把标量面积变化当作力／力矩差的完整导数。代码只更新缓存的接触锁列、材料覆盖及反力基，不在差分探针里重建实体Boolean。

## 3．实际怎么算梯度、怎么更新

每个单位方向选两个正交切平面坐标，写成\(E_j\in\mathbb R^{3\times2}\)，满足\(E_j^T\mathrm{dir}_j=0\)、\(E_j^TE_j=I\)。沿第\(a\)个切向列做两个小角度探针：

\[
\mathrm{dir}_j^\pm=
\operatorname{legal}\!\left(
\cos h\,\mathrm{dir}_j\pm\sin h\,E_{j,a}
\right),\qquad a=1,2.
\]

\(\operatorname{legal}\)表示沿用原地面合法半球约束。\(\mathrm{dir}^{j,\pm}\)指整组方向向量，只有第\(j\)个方向被替换。其余布局参数、需求点、权重与归一化尺度保持一致。实际坐标梯度是数值差分：

\[
\boxed{
g_{j,a}\simeq
\frac{\widehat D_{\mathrm{all}}(\mathrm{dir}^{j,+})-
      \widehat D_{\mathrm{all}}(\mathrm{dir}^{j,-})}{2h}
}
\]

这相当于对全部需求的差距变化求加权和；每个探针都重新投影到变化后的锥，不冻结某个最远需求或其最近点。普通差分角为0.75°，细修用更小的角度；合法边界上的裁切与接触开关使它成为给定分辨率的割线，不能称为解析shape gradient。

沿\(-E_jg_j\)更新，采用球面exponential与原合法方向投影：

\[
\mathrm{dir}_{j,\mathrm{new}}=
\operatorname{legal}\!\left(
\cos(\eta\|g_j\|)\,\mathrm{dir}_j
-\sin(\eta\|g_j\|)\frac{E_jg_j}{\|g_j\|}
\right).
\]

零梯度保持原方向。\(\eta\)通过多个角步幅候选选择，不固定每步走一样远。代码还对共同转向坐标求同一个全组目标的差分，使多个共同遮挡者一起移动；平台处另允许有限接触事件sampling。二者在日志中分别记录，不把sampling写成梯度步。

不可行时按全组积分误差净下降选择连续步，所有原需求通过的候选优先；已经可行后保持全部原需求可行，再减材料。最终复用已算出的每pose全部32768原始需求mask和反力，通过即PASS；固定布局mesh用于显示和材料体积。airborne时工件地面列移除，质心需求与重力保持。

实际实现：[方向探针和球面更新](../../helper_func/continuous_support/fast_gradient.py)、[全部需求分层](../../helper_func/continuous_support/adaptive_fast_gradient.py)、[固定局部积分尺度](../../helper_func/continuous_support/stable_gradient.py)。完整搜索流程及预算见 [Step4.2](../../step4.2/fast_gradient_algorithm.md)。

## 图与演示视频

在仓库根目录只重画这一张数学图：

~~~sh
MPLCONFIGDIR=/tmp/cadgrasp-direction-mpl PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/Co-optimize/operation_demo/direction/code/draw_math.py
~~~

[原pose1+2+4+6视频](vis/pose1+2+4+6_exit_direction_changes.mp4)保留不变。它采用预设球面插值，展示方向改变对支撑几何的影响，不是新梯度算法的优化轨迹。来源见 [video_provenance.json](vis/video_provenance.json)。
