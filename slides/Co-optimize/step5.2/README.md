# Step5.2：冻结当前 process 的占用 BBX

读取当前七组最终选择和 Step5.1 保存的实际摆放，保持物体、支撑及全部 state 不变。所有长度用米计算。

对任务 $i$，物体和支撑在任务世界中的变换分别为 $U_i,T_i$。当前工作站占用是

\[
\mathcal P_0=\bigcup_i\bigl(U_i\mathcal O\cup T_i\mathcal S_4\bigr),\qquad
B_0=\operatorname{AABB}(\mathcal P_0).
\]

直接取全部实体网格顶点的逐轴最小值和最大值，不加余量，也不把不同 state 的局部坐标混在一起。体积为

\[
V_0=(h_x-l_x)(h_y-l_y)(h_z-l_z).
\]

这个 BBX 是所有已保存**工作状态**中物体和支撑的占用盒。工具／机器人轨迹和装卸运动尚未给出，不计入这个有限盒；完整工作禁区和退出通道仍作为 Step5.3 的几何排除条件。占用盒体积与支撑材料体积分别记录。

在保留现有材料和摆放的前提下，只增加 base 不会减少 $B_0$，所以 Step5.3 始终用这一份固定参考盒。对新增材料 $A$，真实增量是

\[
\Delta V_B=\operatorname{Vol}\!\left(\operatorname{AABB}\left(B_0\cup\bigcup_iT_iA\right)\right)-V_0.
\]

只在某一个 pose 留在盒里并不足够；新增实体在**每个支撑 state** 下都必须检查。

运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5.2/run.py B
```

每组输出 `step5/step5.2/occupied_volume.png` 和 `data/report.json`。图片叠加全部物体工作摆放和所有不同支撑 state，深色直线给出同一个三维占用盒；PNG 内不加文字。总图集为 `output/B/step5.2_index.html`。

前序代码、Step4／5.1 输出和 objects 均只读。执行源码和输入指纹记录在本次 batch 下。
