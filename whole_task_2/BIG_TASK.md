# Big task：候选池 + Step4 认领（2026-10-25 晚）

输入是 `objects/B/big_tasks/<name>.json` 的一大块工作区（先跑 whole_shape）和一个工位（默认 `robots_config/random/07_kuka.json`）。Step1 不再单独跑，它的检查并进了 Step3。所有计算只在 big task 的面上进行。

## 一个 task 是什么

- 支撑带着物体放到桌上就不再动。针能做多少就做多少：一个 task 只有一个固定位置，不在桌上滑。
- 出口逆着针，也就是物体沿针来的方向退出。出口会把物体朝向它的整个正面（法向与出口夹角小于 90° 的面）从外壳上挖开。
- task 的工作区 = 这个正面里，在这一个位置上落进工作立方体、并且通过 Step1 精确检查（针、针头、KUKA）的面。
- 支撑 = 5 mm 外壳 − 物体 − 工具禁区 − 出口通道，也就是背面半壳。

## Step3：候选池（`step3/big_init.py`，只算覆盖，几分钟）

1. 先撒 20 个均匀随机的姿态。每个姿态平移到让工作立方体装下最多正面的位置，再精确认证。
2. 之后每轮补 2 个瞄准的候选，直到池子有 50 个：
   - 按"面积 ÷（1 + 已覆盖次数）²"抽一个面；
   - 出口沿一根能够到它的针轴，绕针随机转一个角度；
   - 立方体必须装下这个面，并尽量装下覆盖少的面。
3. 工作区不到 big task 2% 的候选扔掉。
4. 做不到的面：没有任何针轴能够到的，或者被瞄准了 3 次仍然没人能认证的。这些面以后不再瞄准，最后并给相邻的 task，标记为未认证。
5. 支撑不在 Step3 建，留到 Step4。

whole_shape 的结果：50 个候选，181 秒，每个面至少被 1 个候选覆盖，98.6% 至少被 2 个覆盖，没有做不到的面。

## Step4（`step4/pool.py` + `step4/combine.py` 的候选池模式）

- **准备**（`step4/pool.py`）：为每个候选建禁区（针、针头加 KUKA）和 Step3 的初始膜（几何体，不单独审核），存到 Step4 和 Step5 读取的位置。`seats.context()` 遇到候选池格式的 run 就用 `PoolContext`，其余流程照旧。
- **搜索**沿用原来的随机合并：32 个样本，Rotational reuse 和 Juxtaposition 两种操作，包围盒门槛，完整审核。改动有三处：
  - **抽取**：只抽能新增至少 0.5% 目标面积的候选，按 value 加权随机抽，失败过的权重降低。value = 新增面积 − 代价；代价 = 一个典型候选的面积 ×（1 − 它和组内最近成员的重力方向夹角余弦），也就是朝向越接近，越可能 Rotational reuse、不加材料。32 个样本同时抽，相当于每一步都抽了好几个，高 value 的更容易被抽中。
  - **完成**：剩下没盖住的目标面积不超过 1%，就算完成；用不上的候选直接删掉。
  - **挑结果**：先比座位数，再比包围盒、task 数、体积。
- **保守做法**：搜索时每个 task 用它完整的认证区域来算载荷和禁区。最后按"先到先得"（加入顺序）把面分给各个 task，写进 report 的 `assignment`。一个 task 最终做的面更少，只会更容易扛住，所以完整区域通过的支撑，最终区域也一定通过。
- **还没做**："按认证结果缩"，以及按最终区域重算禁区来省材料。先看现在的结果再决定。

## 运行

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
.venv/bin/python -B -m whole_task_2.step3.big_init whole_shape --workers 16
.venv/bin/python -B -m whole_task_2.step4.pool whole_shape --workers 16
.venv/bin/python -B -m whole_task_2.step4.combine --object whole_shape --workers 22
.venv/bin/python -B -m whole_task_2.step5.combined --object whole_shape
.venv/bin/python -B -m whole_task_2.evaluation.combine --object whole_shape --top 40
```

Step3 的图在 `output/<run>/step3/pool.png`：每个候选一格，粉色是它能做的面。
