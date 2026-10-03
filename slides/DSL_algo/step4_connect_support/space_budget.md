# Step4.1：使用时的空间预算

只考虑使用时的空间，不优化支撑自身的长宽比例，不要求方形或立方体。
沿用所有任务保存的实际工位坐标轴和原点，不逐 pose 重新居中或配准。

`space_budget.py` 读取组内 Step0 report 所引用的原始 Step1 物体顶点，以及
Step0 已计算的全部地面需求点；验证物理输入和需求数组哈希，不重新采样。
每个需求点在所属 pose 的世界坐标表达为 `(x,y,0)`。

1. 计算所有物体 pose 合起来的一个总 3D 轴对齐盒。
2. 按全部原始样本检查每个 pose 的需求点是否落在该盒内，记录各边越界数量和距离。
3. 计算所有物体 pose 加所有需求点的最小总 3D 轴对齐盒，作为理想空间目标。
4. 从物体盒开始，只向确有越界的边扩建，默认每次最多扩 2 mm；到达目标边界就停止该边扩建。
   保存每次扩建后的盒与全部需求点检查结果，不用绘图抽样计算边界。

在固定工位轴和原点下，该并集盒同时给出包含这些输入的最小盒体积，数值记录为 `box_volume_cm3`。
不搜索盒旋转，不把包围盒体积当作实体材料体积。3D 盒用于限定空间，Step5 评价仍是 XY 使用占地。
不会为了长宽相等增加材料或放大盒子。
需求点是压力中心，不是必须覆盖的实体材料；沿用当前实际接地凸包覆盖需求的政策，
物体与需求的并集盒是必要的空间目标，不是已经可以实现的最小支撑。
有限厚度、接触根部、连通、物体避碰和退出可能要求进一步扩建。
同一支撑翻转时其他 pose 的脚也会移动；本阶段未检查这些跨 pose 的脚位影像，
其兼容性属于后续共同允许材料空间与实体构造检查。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python slides/baseline_algo/step4_connect_support/space_budget.py
python slides/baseline_algo/step4_connect_support/space_budget.py --groups pose3+6 --expansion-step-mm 2
PYTHONPATH=slides/baseline_algo python -m unittest step4_connect_support.test_space_budget
```

默认处理有正式 Step0 report 的全部现存组合，排除没有 Step0 的历史参考 pose1+3，避免混用历史和当前姿态。
每组输出位于 `step4/data/space_budget/`：`report.json`、同尺度两盒对比 `bbox.png`。
不会改变原 Step0–3、Step4 实体或 Step5 评价，也不会重建支撑。
这是新构造算法的第一步。后续 [Step4.2](boxed_support.md) 以此为理想下界，
确定性收紧安装摆放，必要时扩大使用空间后构造；新入口为 `deterministic_space.py`。
