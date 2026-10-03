# Step0：随机选择 n 个 pose，并检查地面兼容性

2026-09-29 输入更新：B 现有 **20 个新 pose**，通过完整载荷地面检查的组合有 206 组三姿态、100 组四姿态、5 组五姿态。pose1+2+3+4+5 是[已保留的正式 Step0 示例](../output/B/pose1+2+3+4+5/step0_pose_selection/report.json)，见[姿态生成说明](../../../codes/setup/README.md)。同一批 n=2、3、4、5 各两组已从 Step3 重跑，加入逐次继承的整组头退出检查：pose3+6 全覆盖并构造连接实体（后续 Step4 完整退出通过，但联合载荷验收失败），其余七组在十条链内未全覆盖；[结果与头部图](../output/B/pose2+9+13+15+17/step4/data/batch_report.md)。新机器人轨迹仍未生成。旧输出已删除。共享 Step0 缓存也已删除，其输入引用经哈希核对后指向各组原有 Step1 文件；Step0 数值及图片未重算。

输入物体、pose 数量 `n` 和随机种子。随机选一个 n-pose 组合，做 0.1 与 0.2；任一需求穿地就重新选不同组合，第一组全部通过后才进入 Step1。所有组合试完仍失败则明确退出，不重复抽样死循环，也不自动减少 n。当前多 pose baseline 支持 `2 <= n <= 可用 pose 数`。

## 0.1 地面需求

每 pose 使用相同的原始 32,768 个载荷。已有样本原样复用，缺失任务的输入在 Step0 缓存中准备。关于参考点 `o` 的需求力、力矩为 `F, M_o`：

```text
M_world = M_o + o × F
p_source = (-M_world_y/F_z, M_world_x/F_z, 0)
```

要求正地面法向力，不增加载荷。完整需求面仅用所有原始撒点的凸包显示，不设计实体脚位。Step0 通过后，Step1 直接发布同一批输入文件，不重新采样。

## 0.2 跨 pose 检查

```text
p_target = T_target @ inverse(T_source) @ [p_source, 1]
穿地 = p_target.z < -1e-9 m
```

对所有来源与所有其他目标 pose 检查全部样本。地面横向无限延伸，不构造合法区域或泡泡。总冲突数按原始样本计一次，逐对记录可分别查看。当前固定摆放、忽略支撑自重、单侧地面反力模型下，穿地是必要条件失败；通过不等于头、连接、厚度、装卸、摩擦或强度已经可行。

## 入口

在仓库根目录，用 cadgrasp 环境运行：

```bash
# 完整流程：选一组合法的三个 pose，再运行 Step1–4
python slides/baseline_algo/run_sequential_batch.py B --n 3 --seed 20260929

# 批处理：2、3、4 pose 各寻找两组不同的合法组合，接受者运行到 Step4
python slides/baseline_algo/run_sequential_batch.py B --n 2 3 4 --groups-per-n 2 --jobs 2

# 只验证 Step0，不生成候选头或实体
python slides/baseline_algo/run_sequential_batch.py B --n 2 --seed 20260929 --through-step 0

# 独立 Step0 入口
python slides/baseline_algo/step0_pose_selection/select_poses.py B --n 2 --seed 20260929
```

默认每任务 200 个候选、至多十条顺序选头链；可用 `--candidates`、`--particles` 调整。原来固定随机六组的 `--existing-groups` 批处理已替换。直接进入顺序 Step3 或实体 Step4 也要求已有通过的 Step0 记录。

批处理先记录全部选择，再用独立进程求解接受组。某个 n 穷尽后不足两组会明确记录，已有合法组照常求解。Step3 不完整时 Step4 只输出失败诊断和头预览，不造实体。选择／批处理记录位于物体级 `step0_pose_selection/`，每组运行日志为 `step4/data/pipeline.log`。独立复核入口 `audit_selection.py` 从原始加工力、位置与 setup 直接计算世界原点力矩，再用物体坐标系中的地面半空间检查，与选择记录逐项比较。

## 阶段与目录

| 阶段 | 作用 | 代码／输出 |
| --- | --- | --- |
| Step0 | 随机选 n 个 pose；需求面检查；失败重选 | `step0_pose_selection/` |
| Step1 | 发布同一批载荷输入 | `step1/`／`step_1_needs/` |
| Step2 | 固定 1% 面积候选头及几何筛选 | `step2_local_support/` |
| Step3 | 每 pose 3–4 头，后续继承一个最佳已有头 | `step3_scheculer/` |
| Step4 | 将选中头连接为实体；原 Step5 | `step4_connect_support/`／`step4/` |

物体级选择记录在 `output/<object>/step0_pose_selection/selection_n<n>_seed<seed>.json`，包含所有尝试和最终选择。拒绝组合不创建独立输出组；仅通过者建立 `output/<object>/pose<i>+.../step0_pose_selection/`。`--through-step 0` 不发布 Step1。

每组选中／既存检查结果保留一张 `floor_point_conflicts.png`：蓝色自身需求面，浅绿色其他 pose 未穿地部分，红色穿地部分及来源 pose。完整填色、透视显示，无坐标轴；地面绘图范围不参与计算。`report.json` 为 `step0_1` / `step0_2`；`data/floor_contact_<pose>.npz` 保留原始五字段，实体 Step4 直接从此读取。`floor_check_<pose>.npz` 保存高度与掩码。

只重绘：`python slides/baseline_algo/step0_pose_selection/draw_floor_points.py --object B --group pose3+4`。只复查已有 Step1 的组合：`python slides/baseline_algo/step0_pose_selection/rerun_step0.py B --group pose4+5+6+8 --replace`。

## 原十姿态输入的历史验证

B 原十姿态清空输出后批处理（seed=20260929）：n=2 在第 7、12 次选中 **3+4、5+10**；n=3 穷尽 120 组、n=4 穷尽 210 组，没有合法组。独立原始载荷复核与全部 342 次筛选一致。两组 Step3 分别用 6、5 个独立头，所有 pose 的原始 32,768 载荷均通过。两组 Step4 都因整组头沿当前候选方向退出时碰撞物体而失败，没有连接实体。详见[历史报告](../output/B/history/before_compatible_poses_860a4233e74b/step0_pose_selection/batch_report.md)。

原五组 Step4 检查移到 Step0 后结果不变：1+3 为 1713/0；1+9+10 为 1366/159/13173；4+5+6+8 为 0/7947/32765/551；5+7+8+9 为 7947/26/551/20200；6+9+10 为 788/32510/0。原 Step5 图片、模型迁到 `step4/`，没有重新计算这些历史构造。迁移记录保留原始代码哈希；旧实验不冒充新代码下的认证。
