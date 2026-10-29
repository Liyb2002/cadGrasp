# Whole task 2：正方体工作窗口与共享支撑

2026-10-22 按 `whole_task/` 的已确认版本重置了 `step1/` 和 `idea/`。Step1 仅适配了包名、读写路径，以及下游所需的 `object_of` 辅助函数。`whole_task/` 已于 2026-10-24 删除，本目录不再依赖它。B、B+01、B+02 的运行数据也已删除（只留 `data/B/step3_mold_plate/` 的共用外壳），下文指向它们的链接仅作记录。

## 当前输入与 Step1

- 六个固定落桌姿态：+X、+Y、+Z、−X、−Y、−Z。最终任务数量尚未优化。
- 工作窗口为边长 77.393 mm 的正方体，取 +X 参考物体世界 AABB 最长边的一半，从靠机械臂的一面向物体内部延伸。
- 可加工面为角度可达表面与正方体内表面的交集。位置限制作用于加工点／针尖 TCP；针根和头部可以在框外。
- 实际任务读取互斥的 `face_task`，不填入未认证面，不删除可加工孤岛。
- 重跑 Step1 得到 6 个 task、2,456 个面、288.57 cm²（51.56%），重复分配与未认证补入面均为 0；3,293 组面片／姿态通过磁盘复验。

| task | 实际面数 | 表面积占比 |
|---|---:|---:|
| +X | 384 | 8.28% |
| +Y | 277 | 5.14% |
| +Z | 490 | 10.00% |
| −X | 354 | 7.92% |
| −Y | 631 | 13.12% |
| −Z | 320 | 7.09% |

[工位](output/B/step1/step1.1/station.png) · [分区](output/B/step1/step1.2/areas.png) · [working poses](output/B/step1/step1.3/working_poses.png) · [计划](data/B/step1_reach/plan.json) · [磁盘复验](data/B/step1_reach/validation.json) · [候选区](idea/find_tasks.png) · [边界描绘视频](idea/find_tasks.mp4)。

## 本次清理与重跑

已清空 B 的旧 `data/B/` 和 `output/B/`，共 25,613 个文件、9,522,290,534 字节，并清除 Python 字节码缓存。旧 B 后台试算已停止。其他工位实验及输入资产保留。复制与清理记录见 [manifest](copy_manifest_20261022.json)。

Step2 导出代码从旧数据目录移至 [code/export_loads.py](code/export_loads.py)。后续采用现有 `run_station.py` 使用的流程：Step3 初始化 → Step2 载荷导出 → Step4 禁区与 `combine` → Step5 `combined` → evaluation。所有支撑从新 Step1 冷启动，不复用旧 B 结果。

Step3 已通过独立复验：六件薄壳合计 977.682 cm³，共用 base 为 2,278.471 cm³。Step2 已导出并复验 196,608 对载荷及 13,110 个连续外包目标。57 项接入测试通过。Step4 禁区已通过磁盘复验，合并与后续阶段正在重跑。以各阶段新生成的 `report.json`、`validation.json` 和 [运行状态](data/B/rerun_status.json) 为准；旧版五任务或十三任务数值已撤下。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
.venv/bin/python -B -m whole_task_2.step1.run
.venv/bin/python -B -m whole_task_2.step1.find_tasks_figure
.venv/bin/python -B -m whole_task_2.step1.find_tasks_video
.venv/bin/python -B -m whole_task_2.step3.run --workers 6
.venv/bin/python -B -m whole_task_2.code.export_loads
.venv/bin/python -B -m whole_task_2.run_station B --start step4_keepout
```

具体流程见 [PROCESS.md](PROCESS.md)。工位、工作臂与搬运臂的当前状态（实机 demo 交接）见 [idea/demo/README.md](idea/demo/README.md)。
