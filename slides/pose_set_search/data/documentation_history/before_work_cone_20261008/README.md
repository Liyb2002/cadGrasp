# 大 pose set 的共享支撑搜索

当前正在补齐完整工作禁区后重跑：B 原始设定为 **30° 半角（60° 总开角）**，工作面积沿这个角度范围向外形成的半无限区域不可被支撑占据。接触、材料占据和实体构造使用同一保守区域；不再只切掉工作面附近5mm。旧十组结果违反此约束，历史载荷通过不能代表当前可行。具体实现与范围见 [reuse_first_algorithm.md](reuse_first_algorithm.md)。

本轮按用户要求直接比较 **10 个 7–10 pose 的 B 集合 × whole / incremental**，只做快速采样搜索，不做最终实体验收。结果入口在 [output/B/README.md](output/B/README.md) 和 [可点击浏览](output/B/index.html)；各组的过程图、每一步候选选择以及新支撑放在 `output/B/{pose_set}/{whole,incremental}/`。数值源码和以前验收过的结果保持独立。

展示使用保存布局直接重建的三角 mesh，保存在每个方法的 `support.obj`。`process.png` 无文字；全部候选和步幅仍在 README/JSON。目录按实际编号命名，例如 `pose1+2+3+4+5+6+7+8+9+10/`。每个 set 正好两个视频：`process.mp4` 展示选中操作及材料增减，`result.mp4` 展示逐 pose 装入、放稳、取出。固定单个等轴测视角，先 whole、后 incremental，白底无文字。所有几何图和有物体的画面显示半透明琥珀色工作禁区；图只截取有限长度，算法排除半无限区域。放稳停顿1秒时显示向内的可能施力小箭头。

当前目标是：**满足全部原始力／力矩、退出和工作面约束后，尽可能减少实体支撑材料体积**。支撑摆放次数不作为代价；优先转动同一支撑复用，Direction 无法解决的任务才尝试选择性 Juxtapose、Translation 和 Direction。新实现见 [reuse_first_algorithm.md](reuse_first_algorithm.md)、[code/reuse_first.py](code/reuse_first.py)；新结果及 Step4.1 风格图发布到 [Co-optimize/output/B/step4.2](../Co-optimize/output/B/step4.2/)。

搜索加速版本现在默认使用 [fast_search.py](code/fast_search.py) 和 [delta_guidance.py](code/delta_guidance.py)：保留每个采样点的包裹覆盖／切除锁定计数，只更新改变配置的贡献；Direction 更新遮挡列，Translation 还更新移动任务自己的接触行。候选和搜索提交不重建实体 Boolean，反力基仅在所用接触仍存在时复用。过程中的体积、接触及加入前缀属于采样评估；最终导出仍独立构造实体、检查真实接触和全部原载荷。历史验收结果不受这一修改影响。

`run_reuse_first.py --evaluation sampled` 是当前默认；`--evaluation exact` 保留旧的逐候选实体评估。`--search-only` 用于计时或快速比较算法，输出 `search_report.json` 和采样布局，明确 `final_acceptance_run=false`，不导出已验收支撑。每轮默认最多筛选 96 个候选（`--screen-budget`），Juxtapose 保留不同 guest／host，再采样偏移和方向；未试过的候选不是不可行证明。

[加速实现和实测时间](speed_search.md)：此前单集合测速中，pose1–10 从头纯搜索 whole 174.5 秒、incremental 44.5 秒；其末尾实体验收结果保留在测速目录。本轮十组比较通过 `code/run_fast_batch.py` 独立初始化，明确跳过最终验收；过程图通过 `code/render_fast_process.py` 的快速材料占用网格绘制。

[新验收结果](reuse_first_results.md)：1–7 joint 109.031 cm³；1–10 joint 116.210 cm³、incremental 110.171 cm³。增量结果 9 个 pose 转动复用，只有 pose7 Juxtapose 到 pose6；十个加入前缀和最终全部原载荷均通过。

下文和 [results.md](results.md) 保存前一轮实验，含成组 Juxtapose 的历史结果；它们不代表最新体积目标下的最优方案。

这里独立实现两种顺序，不修改 `../Co-optimize/` 的生产代码：

- **joint**：全部 pose 先用共同／相近物体相对退出方向注册到同一位置，尝试联合 Direction；停滞后为困难 pose 选择 host 做 Juxtapose，继续 Translation 与 Direction。
- **incremental**：从一个 pose 开始，每次加入一个；先试当前注册布局及相近方向，不能容纳时尝试 Juxtapose＋局部调整。新增任务的候选必须重新检查所有已有任务。

配置保存物体到支撑的完整刚体变换、退出方向与 host。Juxtapose 保持任务的世界朝向，在 host 的支撑摆放下表达新的物体；Translation 只沿该摆放的地面切平面调整设计终点。允许工位中的水平位置改变，载荷、工作面及原始地面接触随物体一起平移，关于质心的原始力／力矩不变。没有横移滑动通道。

共享材料从各配置的 5 mm 贴合包裹并集构造，统一切掉所有当前物体、工作禁区和完整退出；保留实际接触核心，在核心外沿用原有 1% 净空规则。搜索初筛使用接触几何与最差载荷，入选候选才重建支撑、提取真实接触、检查每个 pose 保存的全部 32,768 个载荷和第七维不上抬约束。面积不作为无界接触反力的容量。

未验收材料连通、完整安装接地和强度时，不称为完整夹具成功。远距离分离不作为 Juxtapose 成功；报告真实重叠、占地与最终材料量。

实验至少包括三个新建的 7-pose set，以及用户指定的 `pose_1`–`pose_10`。原数据只有最多 6 个 pose 的已选集合；新集合只在此目录登记，不修改 `objects/B/`。

实现保存在 `code/`，实验保存在 `output/`；实验启动时冻结代码哈希，保存全部原始输入哈希、布局、反力覆盖、共享支撑与候选轨迹。

算法细节见 [algorithm.md](algorithm.md)，汇总及验证范围见 [results.md](results.md)。1–10 的整体路线和增量路线均已通过原承载、退出、工作面验收及导出实体的全部反力回代；增量运行的十个前缀也全部通过。

两种 10-pose 实体的同尺度对照见 [PNG](vis/ten_pose_supports.png)；整体优化的十个物体 pose 逐一装入同一固定摆放的支撑见 [pose1–10 图](vis/pose1-10_joint.png)。均使用原演示的实体深度渲染器，没有修改已有视频。

从仓库根目录运行，`--out` 必须是新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u slides/pose_set_search/code/run.py \
  --case pose1-10 --mode both --out slides/pose_set_search/output/new_ten_run \
  --iterations 3 --finalists 3 --branch-rounds 1

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/pose_set_search/code/certify.py \
  --result slides/pose_set_search/output/new_ten_run/pose1-10/joint
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/pose_set_search/code/certify.py \
  --result slides/pose_set_search/output/new_ten_run/pose1-10/incremental
```

`--case` 还支持 `seven_chain`、`seven_hard`、`seven_spread`、`eight_chain` 和 `all`；`--mode` 支持 `joint`、`incremental`、`both`。候选与步幅用确定性采样枚举，当前 `--seed` 不改变候选顺序。历史结果应对照各运行目录的源代码快照复现，而不是假设后续版本会给出同一布局。

数值超时或 Boolean / LP 未解决的候选不能被当作物理不可行。每次成功重建后保存检查点；完整可行检查点单独保存为 `exact_states/feasible_*.npz`。`accepted_layout` 也可能来自未完成的分支，最终通过状态应读 `report.json` 和反力审计。
