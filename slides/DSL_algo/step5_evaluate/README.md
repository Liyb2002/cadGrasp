# Step5：整组 pose 的额外占地

用两个 **XY 轴对齐 bounding box** 评价同一个工位的静态占地：

1. `B_object = bbox(union_i O_i)`：所有 pose 的物体。
2. `B_supported = bbox(union_i (O_i ∪ S_i))`：所有 pose 的物体与实际摆放的支撑。

面积均为包围盒的 `width × depth`。主结果记录两个面积及其差值；额外占地比例为
`(A_supported - A_object) / A_object`，越小表示支撑额外占地越少。
不是各 pose 面积之和或最大单 pose 面积，也不是材料体积、表面积或脚垫接地面积。

沿用保存的 task-world 工位坐标轴、原点和各 pose 摆放，不进行独立对齐、重新居中或最小面积矩形搜索。
Step4 的行向量约定为 `fixture = task_world @ basis + offset`，因此每个 pose 的支撑顶点为
`task_world = (fixture - offset) @ basis.T`。先变换实际网格顶点，再取包围盒。
不使用过程图中为显示退出方向而对齐到同一个物体的坐标。

本阶段只评价静态工作姿态，不计退出/翻转的运动扫掠，不增加尺寸限制，不改 Step3/Step4 的验收状态。
保留各 pose 的诊断结果和三维盒范围，但面积评分只使用 XY 两维。

```sh
PYTHONPATH=slides/baseline_algo python slides/baseline_algo/step5_evaluate/evaluate.py --object B --groups pose1+3
PYTHONPATH=slides/baseline_algo python slides/baseline_algo/step5_evaluate/run_all.py --object B
PYTHONPATH=slides/baseline_algo python -m unittest step5_evaluate.test_evaluate
```

输出在 `output/B/pose1+3/step5_evaluate/`：`report.json`、`bbox.png`、`README.md`。
只读取已保存网格、姿态与来源信息，不重新搜索 contact、不重建支撑。

存在 `step4/data/growing_support/report.json` 时，自动使用该报告和同目录的实际模型、摆放；否则使用原公共 Step4 模型。这样当前八组与 `pose1+3copied` 的评价与新 Step4 图一致，不会误用保留的旧 `step4/shape.obj`。自动发现组名支持 `copied` 后缀。

`bbox.png` 使用三维 CAD 视图：半透明蓝色物体、灰色支撑、完整 XYZ 包围盒，后侧盒边用虚线表示。
左右使用同一个正交相机和比例，显示长 × 宽 × 高；评价指标仍然是 XY 占地面积，不改成体积。
绘图实现在 `render_bbox.py`，直接使用计算 BBX 的实际顶点与三维最小/最大坐标。

`run_all.py` 自动发现全部现有 Step4 组合，在每组输出各自的 BBX 对比图，并将总览
`all_groups.png`、数值表 `all_groups.md` 与来源记录 `all_groups.json` 放在
`pose1+3/step5_evaluate/`。不在 B 根目录创建批次文件。`pose1+3copied` 使用历史姿态及新支撑；保留历史参考组合与当前姿态组的来源区别。
