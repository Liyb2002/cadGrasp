# 两个 pose 的接触区域与地面撒点

[合并面板用 Pose 1](contact_areas.png) · [合并面板用 Pose 3](contact_areas_pose3.png) · [绘图代码](contact_areas.py)

两张图的前两个面板相同，只有最后的合并面板采用不同 pose。单独面板：[Pose 1](contact_areas_data/pose_1.png)、[Pose 3](contact_areas_data/pose_3.png)、[合并到 Pose 1](contact_areas_data/combined_pose1.png)、[合并到 Pose 3](contact_areas_data/combined_pose3.png)。

- 前两栏各画当前 pose 的三个物体接触面及地面撒点。
- 最后一栏分别把两个**物体**对齐到 Pose 1 或 Pose 3，并对接触面、地面和撒点使用相同刚体变换。两套地面仍有各自的方向，没有压到同一个平面。两个版本都沿用原 pose 与原相机方向，不再额外旋转 90°。
- 接触面直接来自 B/pose1+3 保存的 Step3 曲面三角形。共享接触面的对齐误差小于 `1e-12 m`，合并后只画一次，共五个区域。
- 地面统一灰色，橙色与蓝色撒点及虚线分别表示 Pose 1、Pose 3 的地面需求。每套原始 32,768 点全部参与绘制；虚线凸包用完整点集计算。透明工件帮助显示背面的接触区域。

这是物体坐标下的两任务需求合并图，使用 `T_world_mesh` 做对齐；不使用 Step5 的支撑配准变换。原始几何和样本均未改动，输入哈希、变换矩阵与对齐误差保存在 `contact_areas_data/metadata.json`。

在仓库根目录重新生成：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/private/tmp/cadgrasp-slides-mpl \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/sys_floor/contact_areas.py
```

绘图复用同目录 `steps.py` 的 CPU 光栅化和字体工具，不需要运行 Step5 构造。
