# Juxtapose：在所有不妨碍 sweep 的地方补贴合支撑

配合 [Direction](../direction/README.md) 和 [Translation](../translation/README.md)，构成 [Step4.2](../../step4.2/algorithm.md) 的三个互补 operations。Juxtapose 后的位置和退出方向仍需优化；连续演示见 [Combined](../combined/README.md)。这里是独立几何演示，尚未接入生产搜索。

[视频](vis/pose1_pose2_juxtapose.mp4) · [最终示意图（唯一 PNG）](vis/pose1_pose2_final.png)

## 这次操作怎么做

使用隔壁 Direction 同一个 B/pose1+2+4+6 shape：起点是原 Step3.2 包裹减去四个保存方向的 sweep，固定在 pose1 的摆放。让物体以 pose2 的原生朝向尝试装入这个位置；视频只显示原来的右侧 pose2，支撑保持不动。最终 PNG 保留两个配置分别使用同一支撑的对照。

新增支撑沿用 Step3.2 的贴合包裹构造，最大顶点外移量同样为 5 mm。在新 pose 的所有非工作面生成包裹，再减去所有当前 pose 的物体、工作禁区和完整退出 sweep，保留所有不冲突的贴合区域。

**最终支撑 =（原支撑 ∪ 新 pose 的贴合包裹）−（所有物体占据 ∪ 所有退出 sweep ∪ 工作禁区）。**

删掉的是旧支撑与新配置冲突的材料；增加的是新包裹中可用而原支撑没有的材料。这里直接填入全部可用贴合区域，不预选少量承载接触或柱子。包裹围绕最终物体生成，不包裹 sweep，不切除设计重定位轨迹。原 pose2 配置由新配置替换，其废弃退出不作为新的约束。

## 视频顺序

视频加速为 6 秒，保留全部 193 帧。只有一个画面，不显示文字或图例，仅保留物体、支撑和红／绿色材料变化。最终 PNG 保留原说明。

1. 原支撑固定在 pose1，pose2 首次尝试装入，在真实首次碰撞附近停止。
2. 物体继续沿真实插入路径前进。每一帧计算前后位置之间的连续实体 sweep，只删除该段 sweep 与尚存旧支撑的交集；红色仅表示刚被扫到的材料，不预先标出或删除后续路径的材料。
3. 到达最终位置后，绿色贴合包裹在所有可用表面同时加厚，保持其形状贴合物体，并避开全部当前 sweep。
4. 保持物体和支撑的位置，最后新增材料变为蓝色；几何不切换。

每个切除帧检查累计已删除材料都位于已经走过的实体 sweep 中、剩余支撑不与当前位置的物体相交、已删除材料不会重新出现。

视频严格只按真实物体 sweep 切除，不额外删除物体尚未扫到的工作面数值余量，也不在最后一帧偷偷切换几何。最终 PNG 仍采用包含工作面余量的参考几何；两者的微小体积差单独记录在 `animation.json` 的 `canonical_work_relief_not_animated_cm3`，下文的工作面验收针对参考几何。

## 已检查什么

起点中 pose2 物体与支撑重叠约 47.0 cm³。本例删除约 **51.9 cm³**，新增约 **101.0 cm³**，新增材料在 pose2 上形成约 **15,362 mm²** 的真实接触。主体材料为一个连通分量。

已检查两个物体占据、四个配置完整 500 mm 退出路径和工作禁区的交集，均小于 `1e-10 m³`；四个原始工作面检查通过。另有两个直接覆盖检查：所有可用新包裹均在结果中，新增材料全部位于新 pose 的贴合包裹内。工作面切除包含 1 µm 向内数值余量，避免 Boolean 边界点接触工作面内部。

这份示例尚未执行全部原始载荷、1% 净空、安装接地或强度验收。贴合面积是几何量，不直接等于承载容量。后续对不同 host、重叠位置和退出方向产生的真实接触进行联合反力评价。

生成代码：[geometry.py](code/geometry.py)、[render.py](code/render.py)。几何与来源记录：[geometry.json](vis/data/geometry.json)，动画时间表：[animation.json](vis/data/animation.json)。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cadgrasp_juxtapose_mpl .venv/bin/python slides/Co-optimize/operation_demo/juxtapose/code/render.py --rebuild
```

[Translation](../translation/README.md) 已恢复为 active，6 秒视频展示本例 Juxtapose 后横移 1/3 身位；此前 Juxtapose 的柱子版本已由本贴合包裹版本替换。
