# Value network 离线标签生成

已实现并运行当前 B / pose1+3 的首轮数据实验。入口复用 baseline 的当前 Step1 载荷、Step2 接触面、退出检查、路径路线图及联合反力求解，不训练网络、不修改 baseline 输出。

## 重跑

在仓库根目录使用已安装项目依赖的 Python：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/value_network/data_producer/search.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/value_network/data_producer/report.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s slides/value_network/data_producer -p 'test_*.py' -v
```

默认物体 B，两个 pose 各使用原来的 200 个候选。`--attempts 32` 为每个合法 head 的条件补全次数，`--max-heads 6` 为每 pose 头数预算，`--direction-weight 1` 为本次试验 lambda；这些是首轮实验配置，不是最优设置。当前状态为两 pose 都未选头，动作是为一个 pose 固定先选某个 head。

输出为 [data/B/pose1+3](../data/B/pose1+3/README.md)。改变输入、生成代码或搜索参数会拒绝复用原运行配置；原配置不变可恢复已有候选结果。检查记录在本次数据目录，数值错误不当作无解。

## 复用与搜索过程

- `prepare.py`：读取当前 `baseline_algo/output/B/independent_poses/pose_1`、`pose_3` 的原始载荷和候选；核对任务与文件哈希。重放原有限厚度头的方向与根部通路，把合法方向和连通分量缓存到数据目录。没有重新拟合接触面，没有使用历史 pose1+3 展示的旧任务。
- `search.py`：先重验既有独立搜索的成功集合，再对每个 pose 搜索 32 次随机换头／合法扩展，建立成功补全集合。对每个合法候选强制先选它，尝试 32 次种子增补或随机合法扩展，并在成功终局上随机删除冗余头，始终保留强制头。
- 每次加头保留共同退出方向和连接通路，交集为空则分支死亡；中间前缀不要求受力或整体不上抬通过。
- 完整候选首先用 96 条真实原始载荷作快速否决：失败载荷足以否决“全部原始载荷通过”，但通过这 96 条不能算成功。只有全部 32,768 条的联合受力与整体不上抬通过才接受终局。
- 终局验证复用 `joint_samples.classify`、原地面摩擦列和 `passive_support` 的共享不上抬方程。数值错误使用现有 `strict_lp_retry.recover` 重试原方程，不放宽容差；仍未决的组合单独记录。
- `report.py`：对动作自己的已找到成功补全，与另一 pose 在整轮搜索发现的全部成功补全配对，准确计算有限方向集合的最小夹角及总代价。两 pose 的局部受力／路径独立，方向代价在配对时耦合，不强制共享 head。
- 报告重验所有被选为标签的局部终局：全部原始载荷重新分类，另用独立非负 LP 抽查反力；核对每个 action 的强制头、方向、头数及分数公式。

## 标签

根状态下固定先选 h，最终跨两 pose 的成功总头数为 n，则 `N_remaining=n-1`。两 pose 的合法方向统一转换到物体坐标系，枚举方向对取最小夹角 theta，`D_final=(theta/pi)^2`；头数与方向项来自同一个完成方案。`value=N_remaining+lambda*D_final`，越低越好。

每 pose 的 200 个候选全部保留：Step2 非法为 `step2_rejected`；预算内未找到补全为 `budget_unresolved`、value 为 null；找到补全为 `success_found`。未找到不表示不存在补全，null 不替代为任意有限分数。已找到的 value 是本次搜索范围内的最好代价，不是数学上的全局最少头数或已训练网络的预测。

从验证终局额外抽取 32 个不同部分状态／动作，保存已知补全代价。每个部分状态只有一个已知动作标签，不宣称已经搜索它的全部 200 个动作。全部记录合并到 `training_records.jsonl`。

搜索以少量既有成功集合为起点，存在种子与探索预算偏好，后续训练集需要增加独立探索与更多 shape。最终实体、实际体积由 Step4 检查；本次没有生成支撑实体。方法定义见 [method.md](../method/method.md)。

## 十组 × 20 个 state 的试验

入口为 `collect_states.py`，记录当前 state 下所有未选候选的条件 value。配置、搜索预算、输出位置和复用策略见 [PILOT20.md](PILOT20.md)。记录审查入口为 `audit_pilot.py`。

## 五组组合泛化数据

`collect_transfer.py` 只为预定五组生成各20个state，数据保存在 `../data/B/transfer10/`。复用已验证单pose解库，重新计算组合的条件Q，保持无见证动作为null；不生成留出五组的训练数据。后续训练组补采与完整实验见 [泛化报告](../train/transfer/README.md)。
