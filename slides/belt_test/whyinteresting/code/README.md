# One contact module, one fixed base, three tasks

2026-09-22 新增 [单 dock 图](../overview_single_dock.png)：参考的是 `overview` 第一格的旧 B / pose 2，而不是当前 baseline 已更新的同名姿态。当前 baseline 姿态及支撑都已变化，用它重绘不能复现原图。最终图以 `overview_no_labels.png` 为参考，由内置 imagegen 编辑：保留第一格的直立工件、侧面绿色工作区、蓝块轮廓、斜向矩形接口及相机视角，去掉两个闲置插座和对应立柱，仅留下已占用的一个 dock。

这是用于 slides 的栅格示意图编辑，不是恢复旧版 CAD 网格，也不附加实体或承载验证结论。[编辑提示词](single_dock_edit.md) 记录参考图和生成要求。已移除会读入错误版本的单 dock 重绘入口及其不适用于本图的几何报告；以下 `draw.py` 命令仅针对原三 dock 图。

2026-09-21 更新。[overview.png](../overview.png) 带支撑方向箭头，[overview_no_labels.png](../overview_no_labels.png) 不带箭头；历史文件名 no_labels 不表示去掉所有文字。

两版图都展示三个 task：pose_2、pose_2 倾斜 25° 的示意姿态、pose_4。它们与 [three_poses.png](../../three_poses.png) 共用 `shared_base.build()` 生成的同一橙色地面框架、三个固定插座、同一蓝色模块及工作区。[新版流程视频](../../code/shared_workflow.md) 已改用分开装卸的另一套几何。闲置插座不隐藏；每个面板是同一工件在不同工位的状态。

第二个 task 的绿色工作区位于头部上方，与原工作面及蓝色接触区域分离；对最终公共底座和蓝色结构的面中心视线筛查未发现遮挡。该筛查不代替工具扫掠或载荷验证。

静态图保留矩形接口分离／落座的放大细节，非接口几何淡化并轻微模糊；视频已去掉细节窗与右侧步骤，只显示全幅主画面。矩形插入行程 30 mm，接口没有防退出锁。

本目录只保留绘图入口；源几何、工作区、公共底座、渲染与验证共用 `../../code/`。尺寸、来源与边界见 [统一说明](../../code/README.md)。[geometry_report.json](geometry_report.json) 保存三个 task 的变换、工作面、相同底座标记及几何采样结果。

红色 F_supp1、F_supp2 是原接触区域按面积平均的内法向，随模块旋转，只表示支撑方向，不是新任务下已求解的反力。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/belt_test/whyinteresting/code/draw.py --both
```

无参数生成带箭头版；`--no-forces`（兼容 `--no-force-labels`）生成不带箭头版。
