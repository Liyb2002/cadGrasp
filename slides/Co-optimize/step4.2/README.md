# Step4.2：全组梯度、离散跳步与动态 base Oracle

当前搜索在[动态换座的32候选V8](../population_soft_algorithm.md)及连通主体检查上，只将连续位置提案换为PSO；入口[run_population_soft.py](run_population_soft.py)，默认`--population-continuous-policy pso`，显式`differential`保留旧差分对照。原位初始化不分配Juxtapose身份；每槽位每轮一次离散或PSO变异，再一次调整。身份在优化中选择，N/3只影响新增/取消抽样。旧close/up、translation及floor高度修复保持；raw与调整结果均接受完整检查后保留较好者。默认32活动槽位，不读取历史优化布局。下方V8的50/52统计属于旧差分批次；PSO全量结果另存，不能沿用这些数字。

中断的ALL52实验可由[run_pso_campaign.py](../helper_func/run_pso_campaign.py)续跑到Step5与Evaluation：保留本批已经完成的组，单独归档中断组并从注册原位重跑。数值依赖冻结，原始物理输入及已接受结果保持；中断成本与重跑来源另列。

V8全部52组／263个object pose已完成冷启动、最多8轮，50组完整通过、2组原受力失败，无运行错误。2–8pose全部48组通过，9pose1/2，10pose1/2；通过组244个object pose／131个实际支撑安装pose。[逐组Evaluation与PNG](../data/results/B/step4_population_soft_all52_20261012_final.md) · [结果说明](../population_soft_results_20261012.md)。共469张直接PNG；1664个初始化、2784个槽位事件、8617984个保存mask条目、5625个保护文件与412个冻结源码已核对。图中失败布局只用于诊断。本轮只跑Step4。[旧32候选冷启动33/52存档](../data/results/B/step4_population32_all52_20261012_combined.md)。

后处理现已拆为Step5.2材料删减与Step5.3完整构造。原搜索内Builder仍采用同一确定性内核，源码位置移到step5.3；删减暂不回灌Step4搜索。新流程会重新检查删减后的真实接触，不能把旧成品的mask直接用于已删接触。

入口[run.py](run.py)调用[run_oracle.py](run_oracle.py)，搜索实现在[oracle_search.py](oracle_search.py)。所有pose从保存的Step4.1布局共同参与，原任务朝向与需求不变。完整算法与连续损失公式见[根README](../../../README.md#step4联合布局优化与动态-base-oracle)，Oracle细节见[base_oracle.md](base_oracle.md)。

[共享材料表示算法设计](../representation_algorithm.txt)定义了完整研究方案。[run_material.py](run_material.py)已接入第一版混合布局搜索：最多8个sample、原接触端口匹配、原位复用／换座组合、原地面需求的高度提案、完整连接与搜索内成品成本。[material_search.py](material_search.py)、[reuse_proposals.py](reuse_proposals.py)、[material_evaluator.py](material_evaluator.py)复用原物理和法向Oracle，并把已构造的成品交给Step5.2直接保存。一般的三维稀疏材料选择规划器尚未实现。

新实验可用 `run_material.py B --sets all --min-poses 5 --max-poses 8 --jobs 6 --iterations 8 --initialization-budget 256 --output-name <新名字>`。当前组合目标为2–4个原位复用pose及2–4个显式Juxtapose pose；成品通过和组合目标分别记录。初始化预览、局部割线探针和每轮正式提案分别计数，不把少量正式候选误报为全部评价次数。下文描述保留的旧 `run.py` 搜索入口。

混合搜索同时提出全部pose及重叠复用组的联合方向割线／接触事件。独立改变一个退出方向可能仍被重叠组的其他退出扫掠遮挡。Juxtapose的复用角色交换轮流探索失败的复用pose与其他复用pose；恢复原位时同步转换退出方向，保持世界方向。八个sample共用一轮需求测度，每轮最多8个新布局检查全部原需求；连接与成品构造分别限3个和2个新候选。

当前入口已合并确定性组件包含判断、座位／安装高度的共同线性提案、失败guest与阻挡guest的轮换，以及接触端口在全部安装state下的必要占用盒。种群保留一个物体着地的复用分支，避免所有sample都被抬进同一个受力死路。原始任务旋转、接触法则和32768需求保持。

成品规划试验少量世界占用盒，将它们拉回共同材料坐标后与原合法见证相交，再调用原Builder。全部合法名义材料必须保留，实体仍通过原地脚包围、连通、闭合和空间检查。稀疏三维材料选择仍是后续工作。点求积少量未满足的候选可在真实成品接触三角形上重新生成原反力锥；只有全部原始需求通过才可接受，不把撒点失败直接改成PASS。`material_contact_certifications`单独记录这些实体接触检查。

`--warm-checkpoint-experiment`冻结一轮的过程布局，若过程已清理则读取其最终布局。`--initialization-seeds`读取显式的按case排列的布局文件清单；所有种子重新接受当前完整检查，来源与副本哈希均保存。

全部26个5–8pose set的当前真实成品、组合数量、实际支撑state成员与完整BBX见[运行评估](../data/results/B/mixed_material_campaign_20261011_01_review.md)。各次重试单独保留批次、warm来源、执行代码和失败记录，不能把多次重试的最佳结果当成一次冷启动成功率。

默认由[layout_frontier.py](layout_frontier.py)持续保留4个完整布局。父布局与新候选同时比较，分别保留受力好的、Oracle包围好的和地脚位置／布局不同的分支，防止退出方向改善受力却挤掉关键base空间。同一轮全部分支共用需求求积点、权重和尺度；96个正式候选快筛额度由各父布局分摊，最多8个短名单候选执行全部原始需求检查，最多3个新finalist查共同连接。已找到的整体可行布局优先保留，连接见证随正确布局保存。

Direction在单位球面的两个切方向求接触事件割线；Translation允许物体座位与最终支撑state的世界XYZ，使用多步幅并保持不穿地。sampling和周期性Juxtapose扩展离散host／重叠结构，再在后续轮次连续精修。每步都考虑全部pose接触的新增与损失，一个state可服务任意多个pose。Oracle未找到的中间布局可以探索，地脚面积不能代替承载包围。

Oracle已经默认使用每个原工作三角形沿外法向的单向棱柱并集：没有角度扩张、全局凸包或安全余量。快速地脚包围与完整连接查询使用同一法向工作空间；原加工力角度和接触／初始化模型保留。

每个pose原始32768需求全部通过即工件受力PASS；整体PASS还要求合法地脚包围与共同连接Oracle通过。XYZ位置改变时需求凸包动态更新。机器人拿起空支撑后的搬运转动不参与约束。可行后继续保持Oracle约束的材料下降及转动复用恢复。

最终mask／反力直接复用，不做末尾重复需求求解或几何微扰。名义实体用于材料比较与显示；连接空间见证不是成品base。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B --sets all --jobs 3 --output-name my_base_oracle
```

`--frontier-width 4`为默认多布局搜索；`--frontier-width 1`是旧单布局对照。`--oracle-work-mode normal`为当前默认，`angular`只用于显式历史模式对照。新实验须使用新输出名字。

最终数值在`data/results/B/{pose_set}/step4/step4.2/{experiment}/`：`layout.npz`、`*_force.npz`、`support.obj`、`base_oracle.json`、`oracle_connected_witness.obj`与报告。`process.png`、`final_result.png`自动发布到相同层级的`output/B/`。候选布局、过程帧和日志在出图完成后删除；批次执行副本在哈希校验后删除，保留哈希与最终统计。

历史`base_oracle_v3`在其原有5mm／固定安装条件下52/52通过；该结果不代表当前法向工作空间、可移动安装及混合数量目标的成功率。[历史图集](../output/B/base_oracle_results_index.html) · [历史数值](../data/results/B/base_oracle_results.md)。
