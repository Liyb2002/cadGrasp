# 共享头配准纠错：B / pose1+3

[查看配准与失败说明](steps.png) · [绘图代码](steps.py)

2026-09-28 用户纠正：共享头必须是同一个物理头，两 pose 的完整接触面和头实体在支撑坐标系中重合。原四步图使用六块物理接触面、五个 ID，把黄色共享头复制成两份，违反原 3+2 定义，已撤回。

当前图仅展示**正确配准的五个物理头**，黄色共享头只保留一个。按原始 `T_world_mesh` 建立相对变换，再同时变换头、物体和两张地面；共享面的顶点集合误差约 5.8e-17 m。浅灰物体是真实 B / pose1 网格，以 24% 不透明度显示。

两套原始 32,768 个地面需求全部参与绘图。橙色对应 pose1，蓝色对应 pose3；虚线仅为需求凸包标记，不是实体底座。

身体生长图的配色统一为灰白色（`#dce2e2`），不再用浅蓝色强调本步新增材料；接触头保留各自颜色。这一规则适用于后续恢复的构造步骤图，当前诊断图没有支撑身体。

**没有生成新的支撑身体。** 在固定原始完整接触对应关系、原任务和无自重支撑的模型下，pose1 的 1713/32768 个地面需求越过另一地面所允许的落脚半平面，最坏越界约 30.713 mm；pose3 越界数为 0。通过原任务坐标间直接变换独立复算，得到相同结果。因此不能只把黄色头挪到一起，再保留旧身体或宣称构造成功。

该必要条件失败只针对以上固定对应关系，不证明所有接触设计都无解。本次未追加载荷，未运行最终受力或退出审计。

## 来源与状态

读取原 Step3 接触、头实体、Step1–4 任务数据与历史 Step5 来源记录，所有 baseline 输入只读。

- [当前失败记录](../baseline_algo/output/B/pose1+3/step5/data/report.json)：`registration_checks` 包含旧配准拒绝原因、严格配准矩阵、物理头计数和地面需求检查。
- [图像元数据](steps_data/metadata.json)：源文件哈希、共享头检查和图像哈希。
- 旧六面支撑及报告保留在 `../baseline_algo/output/B/pose1+3/step5/data/history/before_fast_construction/`，当前不导出有效 `shape.obj`。

绘图脚本在读取旧构造阶段前强制检查共享头；拒绝时输出上述诊断图，不再画错误的四阶段成功序列。

## 重新绘制

在仓库根目录执行：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/private/tmp/cadgrasp-slides-mpl \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/sys_floor/steps.py
```

主图输出到 `steps.png`，无文字的正确五头场景为 `steps_data/registered_heads.png`。旧 `step_01.png` 至 `step_04.png` 已移除，避免继续作为有效构造引用。绘图使用 CPU 三角形光栅化，不依赖浏览器。
