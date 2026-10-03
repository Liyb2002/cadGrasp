## 五组数据的组合泛化实验（最新）

完整结果见 [transfer/README.md](transfer/README.md) 和 [结果图](transfer/results.png)。原网络冻结后在十个新组合上通过 1/10。仅用其中五组从随机权重重新训练，训练组 5/5 达到记录解代价；新留出组通过 4/5，但 0/5 的代价接近最好记录解；在 baseline 原有十组上通过 4/10，仅 1/10 的代价接近记录解。接近定义为头数＋退出方向代价差 ≤0.05，沿用训练前的停止标准。

这表明“能找到可行解”的能力有一些迁移，“少头且退出方向接近”的质量泛化仍不足。初始数据是五组 ×20 state；仅在训练五组补采后，最终 checkpoint 实际使用 1,892 个不同 state、195,606 个有限 value。只测试已有 15 个 pose 的新组合；没有测试未见过的 pose 几何。新 checkpoint 在 `transfer/fit_21/best.pt`；原 `current/best.pt` 不变。

# 当前版本：固定十组 pose，完整 state 编码与真实 Step3

已把 baseline 完整复制到 [`../baseline_algo/`](../baseline_algo/README.md)，替换副本的 Step3 选头入口。运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/baseline_algo/run_step3.py
.venv/bin/python slides/value_network/train/predict.py --state slides/value_network/data/B/pose1+3/pilot20/state_000.json
```

当前权重在 `current/best.pt`，真实搜索结果在 `../baseline_algo/output/B/value_network_step3/report.json`。

之前把已选头 embedding 求平均，无法保留每个头的完整身份。现在直接输入 3000 维已选头 mask、15 维任务 mask、15 维每 pose 头数，经过两层 512 宽 MLP，一次输出全部头的 Q 和记录补全覆盖 logits，共 4,892,528 参数。Q 拟合 `N_remaining + D_final`；辅助覆盖分类学习“数据里有成功补全证据”，不把未知当成物理不可行。运行时先做精确候选、非重复和共同路径过滤，再在预测有补全证据的动作里选最小 Q。没有此类动作时使用最高覆盖预测，并记录 fallback。力学与整体不上抬只决定终局是否接受。

只记住初始 20 state/组，还不足以逐步搜索：第二步就可能进入未采样的 state。训练因此反复执行真实 rollout，给遇到的新 state 补充同一已验证解库的条件标签，再训练；`improve_fixed.py` 保存了 11 轮记录。终止条件为十组全通过且每组实际终局代价与最好记录解之差 ≤0.05，实际最后差值均为 0。

最后训练包括初始数据和已采集的纠正 state，其中 518 个 canonical state 有有限标签，共 79,874 个去重有限值。最后一轮 MAE 0.01712，已有有限标签集合内的最低值选头命中率 100%；学到的覆盖门控在有标签 state 上选到差值 ≤0.01 的头为 99.807%。这些是训练拟合指标，不是泛化指标。最终网络只针对固定十组任务，所有曾经的 validation/test state 都可以用于此版本。

复制后的 baseline 实际 Step3：10/10 组通过，各 pose 对所有 32,768 条原始载荷满足联合受力和整体不上抬；头数依次为 4、4、4、4、7、8、9、9、11、11，与记录的最好补全相同。终局物体系退出方向代价也相同。候选及共同方向／连通分量是硬过滤；中间受力失败不会杀死搜索。运行时不读取最终解库或动作标签。

`experiments/feedback_progress.json` 记录每轮训练和 rollout；`current/report.json` 保存最终训练数据哈希与指标；`fixed_model.py`、`fixed_data.py`、`fit_fixed.py`、`rollout_fixed.py` 是当前实现。`seed_optimal_prefixes.py` 是可选额外采样工具，本次达到终止条件未依赖它。还没有用新接触构造 Step4 支撑实体；退出方向代价是紧凑性的代理量，不是实体体积。这里的最好解只指有限预算已找到的解。

---

## 历史：原来的小网络与 state 留出实验

# 固定十组 pose 的小型 value network

输入是所有 pose 已选头的 3000 维 mask、15 维 pose 组合 mask、每 pose 已选数量，以及当前候选头的编号。每个头有 16 维 embedding；已选头的 embedding 求平均得到 state 表示，和动作 embedding、pose mask、头数拼接，进入 128 → 64 → 1 的 MLP。模型约 6.4 万参数，只针对当前固定候选集合，不使用三维形状编码。

输出 = 头数基线 + 神经网络修正，并截断到非负。头数基线用训练集对 pose mask 和每 pose 头数做线性拟合后固定；网络学习接触组合和动作造成的偏差。输出拟合 `N_remaining + D_final`，越低越好。

仅使用 30,868 条有已验证补全的有限标签。未知、非法、无路径动作不作为数值回归标签。每组约 12 个训练 state、4 个验证 state、4 个测试 state；相同 pose 组合和已选头集合必须处于同一划分，包括两个重复组的空状态。当前网络没有接收数据集的最终解、remaining_heads 或 dispersion，避免把答案作为输入。

训练用 AdamW、value 的 Huber loss，以及同一 state 两个动作分差的 Huber loss。默认最多 400 epoch；验证 MAE 与平均选头 regret 之和连续 50 epoch 不改善则停止，保留验证最好的模型。regret 是所选头的标签与该 state 最低标签之差。测试集不参与模型选择。报告同时对比“每个头一个固定 value”和“pose 组＋已选头数”的简单基线。选头指标只比较测试 state 中已有有限标签的候选，不代表对未知动作的可行性判断，也不是实际 Step3 rollout 成功率。

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/train/train.py
.venv/bin/python slides/value_network/train/predict.py \
  --state slides/value_network/data/B/pose1+3/pilot20/state_000.json
```

`predict.py` 先排除已选头、非法候选，以及加入后共同退出方向／连通分量交集为空的动作，再给剩余动作预测 value。受力和整体不上抬仍由原来的终局检查决定。

输出放在 `output/`：`best.pt` 权重，`report.json` 指标与输入哈希，`history.json` 训练曲线数据，`split.json` state 划分，`predictions.npz` 逐条预测，以及 `fit.png` 拟合图。训练日志在 `training.log`。

第一版直接编码 state mask 的对比结果保留在 `output_initial/`；仅用头数先验和回归损失的对比结果在 `output_count_prior/`。最终模型和完整验证指标在 `output/`。

## 本次训练结果

RTX 4090，64,415 参数；训练 285 epoch，保留第 235 epoch，训练与评估阶段约 12.75 秒。有效标签按 state 划分为训练 118 state / 18,197 条、验证 39 state / 6,359 条、测试 40 state / 6,312 条。两个没有有限标签的独立 state 不参与拟合。

| 测试指标 | 网络 | 固定头 value | 组＋头数基线 |
|---|---:|---:|---:|
| MAE | 0.847 | 2.009 | 0.751 |
| 平均选头 regret | 0.350 | 0.690 | 0.545 |
| 最低已知 value 命中率 | 47.5% | 27.5% | 35.0% |

网络选头排序优于这两个基线；测试集数值误差仍高于简单头数基线，拟合尚不精确。以上是有有限标签候选上的离线结果；实际搜索成功率需另做闭环实验。模型修改与 checkpoint 选择依据验证集指标，测试指标用于报告。

![拟合曲线](output/fit.png)
