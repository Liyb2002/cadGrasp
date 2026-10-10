# Step4.2 接触恢复梯度

默认 `contact-recovery`：从 Step4.1 方向出发，联合选择补足受力的缺失接触，沿梯度减少它们的扫掠阻挡。允许有界且不损失任何已通过载荷的平台步；目标走不通时确定性切换接触目标。真实几何、完整退出／1% 净空和全部原始载荷决定接受。详见 [algorithm.md](algorithm.md)。

下方为此前版本。

# Step4.2 当前入口

默认 `local-descent`：从 Step4.1 退出方向直接做联合局部优化，无随机 propose。真实材料重建、完整退出／1% 净空和全部原始载荷决定是否接受；连通和完整夹具验收仍未完成。运行方式与限制见 [algorithm.md](algorithm.md)。

下方为历史对照。

# Step4.2 候选局部力学梯度优化

默认 `--search force-descent`：一条链，每轮 32 个单 pose direction_choice 候选；每个候选独立建立局部力学松弛目标并 gradient_descent。全部候选的 descent 前后都检查原始载荷；真实变差则保留 descent 前状态。按最差 pose 通过比例、总通过比例选优，当前状态也参与比较。

旧梯度优化的是冻结几何评分，其导数正确但目标不等于真实承载。新梯度每次更新时重新优化反力平衡，不冻结反力；导数有限差分验证通过。真实接触开关不连续，因此新目标仍为连续松弛，不能保证真实局部最优或全部通过。

B/pose2+3+4+7 的 3 轮 / 96 候选实验：约 40.52 秒，descent 改善 88 个、变差 8 个（均回退）；最后通过数为 `[3204,32766,31489,32751]`，仍未全部通过。候选 descent 平均约 245 ms，其中局部模型/关键载荷准备约 199 ms，优化余下约 46 ms。比旧几何近似慢，但确实检查并记录真实效果。

实验：`data/experiments/force_descent_chain/B/pose2+3+4+7/`；旧/新梯度同起点对照：`data/experiments/descent_review/B/pose2+3+4+7/`。80 个测试通过，包括力学包络导数和全部 32 候选的真实回退。细节见 [algorithm.md](algorithm.md)。接触点模型不代表完整支撑/净空验收，原 published 结果保留。

无 gradient_descent 对照入口：`--search proposals-only`，其余候选数、随机种子、载荷检查和选优规则保持一致。B 同种子 3 轮 / 96 候选耗时 8.62 秒，最终 `[1482,30792,1068,31297]`，比有 descent 的同候选预算结果差，但约快 4.7 倍；尚未比较同时间预算。记录在 `data/experiments/proposals_only/B/pose2+3+4+7/`。
