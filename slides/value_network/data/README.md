# Value network 数据

已生成 [B / pose1+3](B/pose1+3/README.md) 首轮离线标签与可视化。

两个 pose 各 200 个候选；对合法候选逐一固定先选，进行 32 次补全尝试，并记录跨 pose 的剩余补全头数与最终退出方向分散度。全部原始载荷验收后才给成功标签；非法和预算内未完成分别保留状态。

数据生成及复核入口见 [data_producer](../data_producer/README.md)。图片、JSON、NPZ、CSV 和 JSONL 是本地生成数据；本目录保留它们供检查与训练，不改写 baseline 输入。
