# Step4：只计算地面需求点

2026-09-26 最新入口：顺序式 `run_sequential.py` 为两个任务分别输出原始 32,768 个载荷对应的地面需求，位于 `sequential_3plus2/from_<先求的pose>/terminal_expansion/`，配图为 `sequential_overview.png`。它不将未被采用的头自动变成地脚。此前共同五头的末尾补全结果位于 `fixed_area_1pct/terminal_expansion/heads_5/`，使用 v5 报告；下方 v3/v4 路径均为历史版本。

2026-09-26 固定面积版：Step3 每头固定为总表面积 1%，不执行尺寸优化。新输出使用各 pair 阶段下的 `fixed_area_1pct/`，显式头数预算再进入 `heads_<n>/`；当前 Step4 指向 v4 `schedule.json`。下述原 pair 根目录／`sample_result.json` 对应历史尺寸优化实验。

2026-09-26 当前双 pose 入口 [run_pairs.py](../step3_scheculer/run_pairs.py) 对每个任务恰好读取 Step1 的 32,768 个固定采样载荷，分别计算地面散点。每个载荷已经包含加工力和重力；不再追加纯重力行、连续域反例或连续需求外包点。

每个任务使用自己的载荷、质心和原始接地点，不把两个姿态的散点合成一个需求。压力中心公式及完整地面力矩关系保持不变。输出为 `floor_contact_<pose>.npz`，包含 `load_wrenches`、`floor_demands_xy_m`、`total_floor_normal_mg`、`original_pivot_m` 和 `moment_origin_m`。

结果放在 `output/<object>/pose<i>+<j>/step4_floor_contact/`；五头预算对照使用其 `heads_5/` 子目录。既有实验按新规则重新判定后，Step4 `pair_result.json` 指向 Step3 的 `sample_result.json`；新运行指向采用 v3 schema 的 `schedule.json`。

即使 Step3 尚未通过，也输出两套需求散点。Step4 不生成实际地脚或连接实体；后续连接与实体插入另行处理。

旧 `whole_assembly.py` 及相关连续外包实现保留为历史参考，不属于当前入口，不按旧说明自动运行。
