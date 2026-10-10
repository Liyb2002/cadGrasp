---
status: none_applicable
---

# Translation（none_applicable）：历史布局示意

**历史示意。** 当前 Step4.2 已改为 Direction＋[Juxtapose](../juxtapose/README.md) 两个 operations，微小 translation 不再是独立工具。下面保留此前的最终布局／材料变化图与动画；其预设位移、数学图和视频不是 Juxtapose 的实现或验收结果。

[数学示意图](vis/translation_math.png) · [视频](vis/pose1+2+4+6_translation.mp4) · [最终 pose1 四个等轴测视角（唯一 PNG）](vis/pose1_final_four_isometric.png)

使用 B/pose1+2+4+6 的原始物体、Step3.2 紧贴包裹、Step4.1 保存的退出方向与逐 pose 切除。只移动 pose1，其他三个 pose 的位置及四个退出方向固定。当前视频演示距离为原生 x 身位的 **1/3，约 48.4 mm**；不是优化求出的最小距离。四视角静态图现读取视频记录的实际终点，已同步为 1/3 身位。

## 当前布局的支撑

设物体为 O，原非工作面包裹为 W，各 pose 相对共同 fixture 的平移为 t_p。每一帧只使用**当前布局**：

\[
S_0(t)=\bigcup_p(W+t_p),\quad
Q_p(t_p)=\bigcup_{0\le s\le L}(O+t_p+s d_p),\quad
S(t)=S_0(t)\setminus\bigcup_p\big(Q_p(t_p)\cup(K_p+t_p)\big).
\]

初始帧直接复用 Step4.1 保存的 `initialization_final_support.obj` 与原包裹的交集，避免重复构造完全重合布局。

K_p 是保存的 Step4.1 `data/pose_*_own_removed.obj`，保留原先实际挖出的逐 pose 退出空间；Q_p 是原保存方向的完整 500 mm 名义退出扫掠，防止新增包裹侵入其他 pose 的完整路径。固定 pose 的切除留在原位，pose1 的切除随其最终位置平移。完整扫掠包含物体初始占据。

**不扣除 pose1 从旧位置到新位置的平移轨迹。** 视频是设计布局参数变化，不是物体在已经制造好的同一支撑里滑动。没有滑动通道，没有凸包填充；起点与移动后位置的包裹共同构成候选支撑。

四格显示同一共同支撑在四个原生 pose 的视图：灰色物体，蓝色保留支撑，绿色当前新增支撑，橙框标记 pose1。实体不透明；最后绿色并入蓝色。没有退出箭头或 sweep 叠加。最终示意图使用 output/B 同一深度渲染器，不透明蓝色实体、灰色 pose1，四个等轴测相机，仅输出一张 PNG。

## 与受力算法的关系

direction/vis/direction_math.png 第一部分的最大反力锥距离可继续共用。旧 Translation 方案曾尝试依据最差反力残差寻找有用接触、其阻挡者及最小相对错位。当前改为 Juxtapose：直接提出目标重叠布局，并决定在哪里、怎样、补多少材料；仍需重新计算所有 pose 的新增／失去接触及原始载荷平衡。

**当前视频展示布局几何，尚未执行上述受力选步。** 新增蓝色或绿色体积不意味着获得有用接触，未证明其他 pose 保持可行。重用保存的逐 pose 切除与完整名义退出，并不构成新增区域的完整 1% 接触核、交叉工作区、连接或强度验收；动画的 Boolean 重叠量仅作数值记录。不能把这一版标成已经解决 B 的力学失败。

## 代码

- [layout_geometry.py](code/layout_geometry.py)：视频和最终图共用的当前布局构造。
- [render.py](code/render.py)：四格视频。
- [final_views.py](code/final_views.py)：唯一的最终四视角 PNG。
- [animation.json](vis/animation.json)：位置、方向、新增/失去材料、名义退出重叠及来源记录。

从仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/Co-optimize/operation_demo/translation/code/render.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/Co-optimize/operation_demo/translation/code/final_views.py
```

显示网格用米，Manifold 内核用 `米 / S.SCALE`；所有实体位移传入 `t_p / S.SCALE`。

## 最终物体空间修正

旧图未同步视频终点，且名义退出扫掠的 Boolean 结果对最终物体有少量侵入。现已额外直接扣除每个 pose 的最终物体实体；1/3 身位的四个原生 pose 物体交叠数值均近零（最大约 6.1e-10 mm³）。这仅验证最终物体占据，完整反向放入路径、制造净空及原始受力仍待验证。当前静态图已经应用此修正，保存的视频生成于此修正之前，不能作为修正后的几何验收证据。

## 数学示意图

[translation_math.png](vis/translation_math.png) 沿用 direction_math.png 的三段结构：最差需求、平移到接触到反力锥的形式链式关系、单分量导数及地面内更新。使用原尺度化七维 wrench；平滑条件不成立时只能用几何引导并做真实全载荷验收。移动一个 pose 也必须更新其他 pose 的接触，保住原已可行的载荷。生成代码：[draw_math.py](code/draw_math.py)。仅生成一张 PNG。

数学图记号修正：第一部分从 direction_math.png 逐像素原样复制；后两部分沿用 `(F, tau), C, G, D`，仅将 `dir` 换成 `trans`，移除新增的 pose/load/wrench 索引符号。平移更新不做单位向量归一化，梯度限制在地面内。
