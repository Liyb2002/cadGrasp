# Step4.2：确定性的紧凑使用空间构造

当前入口为 `deterministic_space.py`。目标是全部工作姿态下，**物体＋安装支撑的总 XYZ 轴对齐包围盒体积**。
世界坐标轴和物体原点沿用保存的任务，既不独立居中，也不要求细长物体变方。
材料体积、接触面积、表面积均不是优化目标。Step5 的 XY 占地另外记录。

Step4.1 的“全部物体＋原始 Step0 需求点”盒是理想下界，不是保证可构造的盒。
允许 Step4.2 超出它；优化始终比较实际使用空间，而不是减少材料。

## 快速、直观的算法

1. 冻结 Step3 的接触面、全部载荷、退出方向和力学结论，保留每个接触的小根部。
   从已通过几何检查的历史安装摆放开始，使搜索有合法起点。
2. 把全部根部和各 pose 的 Step0 需求凸包变换到共同支撑坐标，再映射到每个安装姿态。
   与所有物体合起来计算必要空间包络。需求点只要求由实际接地凸包覆盖，不要求每一点有材料。
3. 固定朝向，最多三轮小型线性规划联合移动全部安装位置，最小化包围盒体积的一阶近似。
   保留地面、2 mm 接触面高度和已有退出分离平面；每次须让真实包络体积下降并通过完整根部扫掠检查。
   固定第一姿态作为坐标基准，固定回溯序列和候选排序。
4. 用固定 12、6、3、1.5 mm 的平移和对应角度的水平旋转做少量局部改进。
   每档最多十二次改进；只接受体积下降且通过完整根部碰撞筛查的候选。无随机采样。
5. 从最小候选包络开始，依次尝试 1、2、4、8、16、32 mm 的边界余量。
   将预算盒按全部安装变换映射到共同支撑坐标，取所有盒及地面半空间的交集。
6. 扣掉所有姿态的完整 500 mm 连续物体退出扫掠及 0.4 mm 构造间隙，补回精确根部。
   保留包含所有根部的整块自由空间；不搜索细腿、大环、局部脚位或最小材料。
   提取连通体时保留内部空腔，不能误把空腔边界丢掉并填实。
7. 对最小通过余量再二分三次。末尾检查完整退出、全部原根部和接触、工作面、地面、
   实际接地凸包覆盖全部原需求，以及各安装下的预算盒约束。仅通过后导出模型。

排序、步长、线性规划、回溯、扩盒和终止条件固定，因此算法可重复。
它是以小体积为目标的有限局部搜索，不宣称全局最小；必要包络允许比理想下界大。
原始头在不同姿态下的互相避让，可能使这个差距明显，并非只多出最后的 1 mm 余量。

## 连续扫掠与数值处理

扫掠由原始封闭物体和**每个正法向投影三角形的完整平移棱柱**做 Boolean 并集得到；
不是若干运动帧的采样。不删前向面，不改变接触，不放宽原 `8e-14 m³` 碰撞容差。
固定按 64、8、16、2 的 Boolean 批次顺序重试；数学输入完全相同，只改变拼接顺序。
原姿态预计算的扩张排除体必须包含未加间隙的连续扫掠，否则拒绝这一数值结果。
构造排除体可以保守填充扫掠中的封闭空腔，最终验收仍用未经扩张/填充的原始连续扫掠。
每次重试的包含误差和最终批次都保存在报告中。

缓存键包含原始顶点、三角形、方向、退出长度、构造源码、Boolean 批次和 Manifold 版本。
安装时先变换原始根部顶点和扫掠顶点，再做布尔运算，避免变换已经拼接的边界造成共面残差。
同一安装摆放的排除体缓存复用，扩盒和二分不重复计算。OBJ 使用 Python 浮点数的最短精确往返表示，并重读确认顶点和面逐位相同。
默认八位小数会破坏接地检查；固定十七位小数也曾引入 7e-18 m 的坐标变化，导致共面布尔重放失败。
本机已验证 `manifold3d==3.5.4`；其余依赖为 NumPy、SciPy、trimesh、Shapely、rtree、embreex 和 Matplotlib。

## 运行和结果

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/run_boxed_batch.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/deterministic_space.py --groups pose3+6

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/replay_compact.py

PYTHONPATH=slides/baseline_algo .venv/bin/python -m unittest \
  step4_connect_support.test_space_budget step4_connect_support.test_boxed_support \
  step4_connect_support.test_deterministic_space
```

每组新结果在 `step4/data/boxed_support/`：

- `shape.obj`、`overview.png`：新构造的支撑与安装姿态。
- `report.json`、`geometry_certificate.npz`：实际盒体积、XY 面积、摆放、搜索过程和完整几何证据。
- `replay.json`、`footprint.png`：独立重读导出模型、重新生成连续扫掠后的验收，以及同尺度三维占地对比。
- `pipeline.log`：该组完整运行日志。

八组汇总和运行前后保护哈希在 `pose2+9+13+15+17/step4/data/boxed_support/batch_report.json` / `.md`。
旧公共 Step4 模型保留，便于比较；新模型必须搭配新报告中的安装变换使用。
不重跑或修改 Step3，也不碰 `pose1+3`；不写 B 根目录 JSON，不生成 HTML 或视频。
`passed` 仅指 Step4 几何。Step3 未通过的原始力学状态继续保留，不能把新支撑称为完整力学通过。

`boxed_support.py` 保留最初固定理想盒的原型和通用几何函数；它不是当前搜索入口。

## 已完成的八组结果

除 pose1+3 外，八组全部生成并通过新 Step4 几何验收；导出 OBJ 的独立重读也全部通过，覆盖 28 个安装姿态。
12 个回归测试及三个连续扫掠解析参考例通过。完整搜索的两姿态重跑，安装变换、实际使用盒和精确 OBJ 字节均相同。
核对 3222 个受保护文件，Step3 和排除的参考组运行前后相同。

与保留的旧公共模型比较，总 XYZ 使用盒体积减少 30.2–72.3%，XY 使用占地减少 29.0–57.2%。
本机最终批次（复用连续扫掠缓存）的搜索时间为每组 7.4–31.0 秒，不含渲染和导出。
部分多姿态组仍明显大于 Step4.1 理想盒；固定原头的搜索未证明接近全局最优。

[逐组尺寸、缩减率及时间](../output/B/pose2+9+13+15+17/step4/data/boxed_support/batch_report.md) ·
[八组同尺度三维占地对比](../output/B/pose2+9+13+15+17/step4/data/boxed_support/all_groups.png) ·
[导出模型独立验收](../output/B/pose2+9+13+15+17/step4/data/boxed_support/independent_review.json)

## 最新图片发布

用户要求所有展示图覆盖到公共 Step4 路径。`render_compact_images.py` 使用新模型和新安装变换，
重绘八组共 44 张 PNG：`step4/overview.png`、`step4/support.png`、全部 `step4/pose_N.png`。
旧图备份在 `data/history/before_compact_images/`，新渲染来源在 `data/compact_visualization.json`。
公共 shape.obj 仍为旧比较模型；图片使用 data/boxed_support/shape.obj 的新支撑。pose1+3 不动。
