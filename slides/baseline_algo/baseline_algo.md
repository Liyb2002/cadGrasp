# Baseline algorithm

当前目录是 `slides/baseline_algo/`。根目录只放 `step*`、`output/` 和本文件；运行、批量调度、重试与检查脚本放在所属阶段。目录规则见上一级的 `slides/AGENTS.md`。

## 坐标与图像

整个数据链使用 **Y-up**：世界坐标为 `(x, y, z)`，地面为 `y=0`，水平投影按 `(x, z)` 保存，重力为 `(0, -1, 0)`，长度单位为米。带 `_mm` 的 STL 使用毫米。

从旧坐标迁移时，位置、力、法向和相机向量映射为 `(x, z, y)`；力矩及旋转轴映射为 `(-x, -z, -y)`。后者需要变号，因为 Y/Z 互换的行列式为 −1。刚体姿态采用 `P T P`，三角面反转绕序，保持实体朝向和正体积。二维切平面参数、重心坐标、颜色、索引以及固定的数值求解器回归基底不属于世界坐标。

所有图像画布为纯白 `RGB(255,255,255)`。Step4 不显示接触块编号；Step5 图片不显示编号，视频只有完整轨迹的一个视图，没有文字。

## 各阶段

| 阶段 | 职责 |
|---|---|
| [Step1](step1/README.md) | 读取既定姿态与工作面，生成 32,768 个实际可达的加工载荷样本，载荷幅度为 0–0.5mg。 |
| [Step2](step2_local_support/README.md) | 按真实表面积分配 200 个中心，构造圆形接触域，检查包裹角、物体和地面，并保存各头的三维退出方向。 |
| [Step3](step3_scheculer/README.md) | 在共同退出方向非空的约束下贪心选块、优化尺寸，检查联合受力与连续载荷覆盖。 |
| [Step4](step4_floor_contact/README.md) | 将外力与重力形成的完整合力、合力矩映射到地面压力中心，输出实际需求点和连续需求的保守外包。 |
| [Step5](step5_connect_support/README.md) | 沿 Step3 剩余方向构造有间隙的框架、开口底座与连接，验证同一实体的连续承载和整件插入轨迹。 |

保留已有阶段目录名，包括 `step3_scheculer` 和输出中的 `step_1_needs`。Step3 的子阶段分别位于 `step3.1_score_candidate`、`step3.2_select_contact`、`step3.3_optimize_contact`。

接触头仅提供法向力，不按上下半球筛选。Step3 原始工件落地点使用四射线单边摩擦锥，系数 64，与 Step5 最大测试系数一致；这是充分摩擦的模型假设。加工射线禁区当前关闭，实际工作面仍须避开。支撑是一个刚体，承载由其共享刚体方程判断。

完整通过需要 Step3 连续工件载荷覆盖，以及 Step5 同一实际实体的连续承载、物体/地面避碰和整件插入全部通过。Step4 的需求点不能独立认证底座。有限方向或形状搜索失败不代表全局无解；失败输出保留实际尝试的头、构件和原因。模型不包含支撑自重、结构强度、变形或跨姿态复用。

## 运行

在仓库根目录、安装了本项目依赖的 Python 环境中执行：

```sh
# 只完整重跑 B 的两个姿态
python slides/baseline_algo/step3_scheculer/rerun_common_directions.py \
  --from-step 1 --workers 2 --case B:pose_1 --case B:pose_2

# 单个姿态，从指定阶段继续
python slides/baseline_algo/step3_scheculer/run_all.py B --pose pose_2 --from-step 4

# 只更新 Step3 最终接触块与共同方向图
python slides/baseline_algo/step3_scheculer/draw_result.py B --pose pose_2

# 重放换轴前后 B 设计的独立对照（参考数据本身也使用 Y-up）
python slides/baseline_algo/step5_connect_support/check_coordinate_equivalence.py

# 各目录的独立回归检查
python slides/baseline_algo/step3_scheculer/check_tests.py
```

批量入口的 `--resume` 会先检查来源和审计，再复用有效检查点。不指定 `--case` 时运行 A1-f、B、C5 的全部四个姿态。旧参数名 `--connection-edge-budget` 在当前 Step5 表示方向候选数，上限 24；每个方向试固定的框架外扩和底座菜单。不同姿态可以并行，同一刚体的连接搜索串行进行。

入口退出码 0 表示模型通过，2 表示流程完成但设计未全部通过，其他非零表示执行或审计错误。

## 输出与查看

所有结果位于 `output/<物体>/pose_<编号>/<阶段目录>/`。批量日志、台账和索引写入最后请求阶段的 `batch_run.log`、`batch_run.json`、`results.json`、`results.md`；总表打印到终端。不创建额外运行目录、自动归档或源码快照。

- Step3：`selected_contacts_directions.png` 展示最终接触块和共同退出方向；`schedule.png`、`withdrawal_directions.png` 展示逐轮结果。
- Step4：`floor_contact.png` 为地面撒点图，`floor_contact_object.png` 为物体视图，`floor_diagnostic.png` 带需求解释。
- Step5：`connection.png`、`geometry.npz`、`support.stl`、`support_mm.stl` 保存形状；`trajectory.json`、`audit.json` 保存验证；`insertion.mp4`、`insertion.gif` 为无文字视频。视频是否同时通过承载检查记录在 `video_metadata.json`。

本次完整重跑了 [B/pose_1](output/B/pose_1/) 与 [B/pose_2](output/B/pose_2/)。接触块分别仍为 C089/C153 和 C139/C024/C011，共同退出方向分别仍有 60 和 48 个。换轴前后的支撑表面比较记录在各自 Step5 的 `coordinate_equivalence.json`，其参考数据为同目录的 `coordinate_reference.npz`。表面距离同时检查双方顶点和三角形中心，不将离散比较冒充连续 Hausdorff 距离证明。

其余 10 组也已按当前 Y-up 代码从 Step1 重跑到 Step5：A1-f、C5 的 pose_1–4，以及 B 的 pose_3–4。它们均完成执行和结果审计，但没有新增完整设计通过：7 组构造出支撑并通过整件插入检查，连续承载未通过；A1-f/pose_1 未在当前方向与形状菜单中找到完整框架；B/pose_4、C5/pose_4 未选出支撑头。各自 Step5 的 `results.md`、失败原因图和 `batch_run.json` 保存具体结果及补跑记录。

A1-f/pose_3 使用 `step3_scheculer/strict_lp_retry.py` 补跑：重试支持包含第七维约束的原始方程，反力解仍检查原始残差；数值不可行结论要求至少两个重试形式明确返回不可行，未知状态不当作不可行。该姿态的 Step3 连续覆盖通过，Step5 整件插入通过，连续承载未通过。两个未选出头的 pose_4 现在也能生成 Step3 结果图，且不把空集合的形式共同方向画成有效设计方向。

上述当前阶段报告使用本次代码与输入指纹。其他保留的历史辅助产物若标有 `coordinate_migration.algorithms_reexecuted=false`，仍只表示坐标和画布迁移，不作为本次重新验证的证据。

研究问题与模型约定见 [problem_statement.md](../problem_statement.md)。
