# Step5.1：最终摆放的地面撒点

复用objects中每个pose的32,768个原配对力／力矩，按Step4.2最终保存的布局重算系统—地面的压力中心。计算是数组叉乘和除法，没有新sampling、梯度、反力求解或几何Boolean。

## 公式

沿用原符号：绕工件质心的需求为$(F_D,\tau_D)$，最终世界质心为$c$。系统绕地面世界原点$O$的需求为

\[
\tau_O=\tau_D+c\times F_D.
\]

地面压力中心撒点为

\[
p_O=\left(-\frac{\tau_{O,y}}{F_{D,z}},\;
             \frac{\tau_{O,x}}{F_{D,z}},\;0\right).
\]

直接复用[codes/precompute_objects/floor_points.py](../../../codes/precompute_objects/floor_points.py)的`pressure_centers`公式。每个点与原样本逐行对应，保留原顺序和权重，不重新生成载荷。

原朝向保持不变，所有施力点和质心平移$\Delta c$时，绕质心需求不变，而撒点变化为

\[
\Delta p_x=\Delta c_x-\Delta c_z\frac{F_{D,x}}{F_{D,z}},\qquad
\Delta p_y=\Delta c_y-\Delta c_z\frac{F_{D,y}}{F_{D,z}}.
\]

因此XY平移会等量移动整片点云，Z移动会按每个样本的水平力分别改变撒点。airborne无需另外一份需求；不能把旧点云升高或只作XY整体平移来代替重新投影。

## 系统边界与支撑自重

撒点描述物体加支撑的**总地面反力需求**。工件—支撑反力是内部力；工件自身地面反力是总地面反力的一部分，不能预先从未知承载分配中扣掉。airborne时只剩支撑与地面接触。

默认支撑质量／工件质量$\eta=0$，与已有objects需求模型一致。如给定$\eta$，支撑世界质心为$c_s$，加入

\[
F_O=F_D+\eta\hat z,\qquad
\tau_O=\tau_D+c\times F_D+c_s\times(\eta\hat z),
\]

再用$(F_O,\tau_O)$代入压力中心公式。全部力仍以工件重量归一化，全部力矩单位仍为工件重量乘米。模型没有材料密度输入，不由支撑体积推断$\eta$。

压力中心确定地面的roll／pitch需求；数组仍保留完整六维地面需求，包括水平力和yaw力矩，供后续base算法使用。点云和有限样本凸包本身不宣称底座已满足完整地面承载。

## 最终布局与图

与现有最终结果渲染一致：

```python
T_world_fixture = native_world[hosts[k]]
T_world_object = T_world_fixture @ placements[k]
```

`native_world` 是包含支撑安装XYZ位移的实际最终变换；`native_world_original` 保留原任务变换。最终平移由 `T_world_object[:3,3] - native_world_original[k,:3,3]` 得到，包含座位与支撑安装两部分。使用实际object变换得到最终$c$，不会把Juxtapose或Translation后的pose重新居中；旧布局没有原任务字段时沿用其原始 `native_world`。

可视化在`output/B/{pose_set}/step5/step5.1/{experiment}/`，最终数值在同层级`data/results/B/`：

- `pose_<i>.png`：每个pose的实际世界摆放，一张等轴测图。蓝色是原最终支撑，灰色是物体，彩色点是该pose全部32,768个地面撒点，深色闭合折线是全部点的最小凸包边界。
- `floor_demands.png`：逐pose图的无文字汇总，顺序见report中的poses。
- `common_fixture_demands.png`：固定同一支撑，将每个pose的世界撒点及凸包边界通过其实际支撑变换的逆变换映射到共同支撑坐标，再用首个参考state显示。各pose的地面仍是不同平面，不合并、不投影到一个共同平面。
- `data/pose_<i>.npz`：全部撒点、绕世界原点需求、竖直反力、原生点云、凸包、实际变换和质心位置。
- `data/report.json`：逐pose高度、来源、颜色、点数和计算记录；`index.html`在图片外标注pose与颜色。

图片没有文字、退出方向、sweep或工作禁区。全部样本进入绘图，不抽点；实体支撑保留真实遮挡。Step4支撑、布局和objects文件均不改动。

## 运行

默认读取动态 base Oracle 已通过的全部 52 个 B 集合，包含原 `illegal/` 的 10 组：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5.1/run.py B --experiment my_final_base --batch-name step51_my_final_base
```

可用`--selection-summary`指定另一份最终方案清单，`--sets`选择其中的组，`--fixture-weight-ratio`给出支撑／物体质量比。新版整套流程默认沿用 Step4 Oracle 的质量比零。改变质量比应使用匹配的布局与 Oracle，不应沿用零质量比的可行性结论。已有Step5.1输出受到保护，不会被默默覆盖。

凸包由每个pose全部32,768点确定。绘图扩展位于本目录[hull_render.py](hull_render.py)，不修改前序步骤。当前直接画出全部凸包。

入口[run.py](run.py)调用[system_floor.py](../helper_func/system_floor.py)计算和[step51_render.py](../vis_func/step51_render.py)绘图；后者复用现有透明物体／实体支撑CPU绘图轮子。当前最终计算记录在`data/results/B/data/step51_connected_base_v1/`，保留上游指纹和源码哈希；临时执行副本校验后删除。接下来由 [Step5.2](../step5.2/README.md) 在 Step4 合法空间中构造实体。

[物理测试](../tests/test_system_floor.py)检验重力、XYZ平移、水平力高度力臂、支撑自重和共同支撑坐标的往返变换。

## 已完成结果

全部52组完成：263个pose实例，共8,617,984个原始撒点和逐pose凸包；全批含绘图85.26s。Step4受力记录不重新求解，保存的上游和执行源码指纹核对通过。

[全部图片](../output/B/step5.1_connected_base_v1_index.html) · [逐组记录](../data/results/B/data/step51_connected_base_v1/summary.json) · [文件核对](../data/results/B/data/step51_connected_base_v1/verification.json)
# 最新运行记录（2026-10-12）

最新V8的50个通过布局已完成全部244个pose的7995392个原始地面需求点。输出批次`dynamic_postprocess_all52_20261012_01_floor02`；输入选择为`data/results/B/data/system_dynamic_postprocess_all52_20261012_01/selection.json`，每个case保留自己的Step4实验来源。直接PNG在各组`output/B/[set]/step5.1/`，无HTML。[后处理与Evaluation](../dynamic_postprocess_results_20261012.md)。下文保留历史批次记录。

