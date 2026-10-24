# Working-task figures

根目录只保留两张总图：
- `hard_working_tasks.png`：translation = 100 mm，14 个已验证可行的 poses。
- `hard_working_tasks_unlimited.png`：translation = unlimited，5 个已验证可行的 poses。

两者均为 permitted angle = 25°、direction = 45° downward；全部 5,000 个原始面均有任务归属。5 和 14 是当前得到的可行数量，未证明连续姿态空间下的全局最少数量。

图中仅保留 `w_i` 编号和粉色区域的表面积占比。100 mm 图画固定工作立方体；unlimited 图画物体自身的世界 AABB，仅作展示，不作为可达范围约束。

从仓库根目录运行，使用保存的场景数据重新生成两张图：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B whole_task_2/idea/easy_hard_task/code/generate.py
```

只重画无限 translation 的图：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B whole_task_2/idea/easy_hard_task/code/generate.py --case unlimited
```

追加 `--recompute` 可从头搜索所选案例。搜索通过候选物体旋转和平移、集合覆盖、逐面工具与机械臂检查以及位置微调求解；没有指定目标 pose 数。限时搜索重新运行时，得到的可行姿态和数量可能变化。

代码：`generate.py` 为入口，`planner.py` 搜索和检查，`cover.py` 选择覆盖组合，`render.py` 画图。`scene_100mm.npz` 和 `scene_unlimited.npz` 只保存重画图需要的 poses、表面归属、工具轴和 setup；不包含逐面关节证书、搜索缓存或检查报告。所有检查在内存中完成，不生成 certificates、progress 或 validation 文件。

这里只处理工作区域和物体姿态，未计算支撑和任务间运动路径。所有输出都在本目录中。
