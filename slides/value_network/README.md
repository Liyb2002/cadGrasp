# Value network

- [方法与公式图](method/method.md)。
- [标签生成代码](data_producer/README.md)。
- [十组 × 20 个 state 的数据总览](data/B/pilot20_shared/README.md)。
- [试验配置和重跑方式](data_producer/PILOT20.md)。
- [最初 pose1+3 的空状态实验](data/B/pose1+3/README.md)。

当前已完成固定十组的条件 value 数据：200 个不同组内 state，126,919 条未选头动作记录；30,868 条有已验证成功补全，5,084 条预算内未知，其他由候选合法性或共同路径直接排除。每条有限 value 都保留最终接触组合、剩余头数和方向代价。所有成功终局每个 pose 的 32,768 条原始载荷都通过联合受力与整体不上抬检查；记录与训练 JSONL 的一致性审查通过，未决数值错误为零。

分数为已找到的最好成功补全代价，不是全局最优保证；未知值保留 null。多 pose 方向联合选择使用有限预算搜索。当前已训练固定任务网络并接入复制后的 baseline Step3，十组全部通过；尚未构造这些新接触的 Step4 支撑实体。所有任务使用当前 independent_poses 输入；pose1+3copied 与 pose1+3 共用相同任务身份，但采样了不同 state。baseline 输入和结果未修改。

- [小型网络与训练结果](train/README.md)：已在 RTX 4090 上训练；当前权重位于 `train/current/`，历史小网络在 `train/output/`。

- [复制后的 baseline 与网络 Step3](baseline_algo/README.md)：运行 `baseline_algo/run_step3.py`，结果在 `baseline_algo/output/B/value_network_step3/`。

- [五组训练与组合泛化实验](train/transfer/README.md)：留出任务只用于最后评估；可行性有部分迁移，头数与退出方向代价的质量泛化仍不足。
