# Active joint shared-head experiment (V4)

The active implementation is described in [the experiment README](../README.md).
Step3 uses one canonical physical-head program for the whole group, global
head deletes/merges, all-pose GPU numerical descent and joint exit-space selection.
Active results are in `dsl_shared/` and `dsl_shared_support/`; old `dsl/` results
optimized pose heads independently and must not be called shared-head results.
Step3 retains the heavy full-construction witness contract. Construction uses
one in-memory geometry acceptance; Step4 copies the same mesh.

---

The following notes describe earlier copied/independent experiments.

> Active V3: `try_dsl_growth.py` materializes the Step3 construction witness without a new connector search. It replays the saved selected paths and exports an identical mesh. Ground contacts in the witness grow adaptively; old foot targets are not mandatory. See [the experiment guide](../README.md). The text below is copied baseline history.

最新算法：[确定性体积导向生长](direct_head_growth.md)，入口 `run_direct_growth.py`。
先从实际接触头生长，候选连接按新增工位 XYZ 盒体积、杆长和固定编号排序；避开完整退出 sweep，必要时惰性网格绕行。所有目标连接完成后，取全部物体和安装后支撑的 min/max，计算实际占用盒子。不再扩盒寻找可行空间。当前保留原安装摆放与地面目标，不声称这些变量已共同优化。下方为历史算法记录。

# Step4：紧凑使用空间与支撑构造

最新材料构造：[从根部长出稀疏支撑](growing_support.md)，入口为 `growing_support.py`。
保持上一版紧凑安装摆放和使用盒，长局部脚垫、渐缩身体和共用短主干。八组均通过独立几何重放，
材料减少 96.2–99.0%，使用空间没有扩大。全部公共 overview、support 与 pose PNG 已换成生长实体；
新模型在各组 `step4/data/growing_support/`，上一版 boxed_support 保留作为合法空间和比较基准。

2026-10-01 新方向：[Step4.1 使用空间预算](space_budget.md)。先计算所有物体 pose 的总 3D 盒，
检查 Step0 需求点是否在盒内，再按越界侧逐步扩建至“物体 pose＋全部需求点”的最小盒。
只考虑使用空间，不优化支撑自身的方正程度。

[Step4.2 确定性紧凑空间搜索](boxed_support.md) 是新入口：`deterministic_space.py`。
用联合平移和固定步长的局部调整收紧全部安装姿态的使用包络，再小幅扩建、裁去完整连续退出空间，
保留包含全部原根部的连通自由空间。以物体＋支撑的总 XYZ 盒体积为目标，取消材料量和接触面积目标。
新模型和独立重放在各组 `step4/data/boxed_support/`，旧公共模型保留作比较；Step3 和 pose1+3 不变。
下方记录原外侧脚垫构造及其历史验收。

**工作面规则与 Step3 一致。** 支撑不能占用工作面；相邻非工作面与工作面共边、共顶点允许。不排除加工方向射线或圆锥，也不增加工作区域边缘的避让带。上一轮“八组原头均非法、必须重新选头”的结论来自错误加入的约束，现已撤销。

**力和力矩以 Step3 为准，Step4 不再重复验算或据此否决。** Step4 的 `passed` 仅指几何验收；不设尺寸上限；Step3 原始通过/未通过状态单独保留，不改写上游结论。

八组现有实体、共 28 个安装姿态全部通过工作面、地面、实际接地凸包覆盖、接触/根部保留、连通与退出检查。**Step4 共 8/8 通过**。原代码的紧凑尺寸上限不是用户要求，已移除；跨度仅作记录。[逐组结果](../output/B/pose5+7/step4/data/access_rerun.md)。104 个原头通过工作面掩码检查。原头、载荷、实体及图片未修改；重查工作面、凸包包含和尺寸，沿用相同输入/实体的原几何证据，没有重新求解力或生成 shape。

## 工作面检查

Step3 的 `eligible_polygons` / `PairGeometry` 直接剔除工作三角面，`SequentialGeometry.view` 检查 `source_faces` 与 `work_ids` 是否相交。它没有剔除非工作面和工作面共有的边/顶点；全局 `ENFORCE_PROCESS_ACCESS=False` 也明确不要求加工射线体积避让。

`process_access.py` 保留兼容文件名，实际执行工作面检查。原头入口复用 Step3 面编号规则；异 pose 头、生成根部和整个安装实体由 [working_surface.py](working_surface.py) 检查。完整三角面求交最大化交点到工作三角形三条边的距离，以区分内部接触与只共边/顶点；闭实体另查整片工作面被包在实体内的情况。使用 1 nm 几何容差，并重放交点的重心坐标。没有以几个顶点、面中心或射线采样代替完整面相交检查。

生成身体时，现有带间隙的物体/退出扫掠已经排除工作面；检查另外覆盖后续补回的根部、闲置材料、连接和掏空后的完整实体。保留物体碰撞、地面、接触、连通及退出检查。力和力矩的通过状态由 Step3 提供。

每组数值记录：

- `data/working_surface_check.json`：全部安装姿态的完整实体检查。
- `data/process_access_preflight.json`：原头工作面掩码检查。
- `data/access_rerun.json`：本次复查状态。
- `data/geometry_check.json`：几何验收汇总及尺寸记录（不设上限）。
- `data/report.json` / `data/current_build/report.json`：Step4 几何结果、Step3 原状态及历史承载诊断分别记录；历史承载诊断不参与判定，保留原生成来源。

旧 `work_access_check.json/.npz` 的加工射线计数以及 `work_access_witnesses.png` 属于额外约束诊断，**不参与当前 baseline 验收**。不能据此宣称现有支撑覆盖了工作面。

## 构造方法

1. 保留 Step3 原头、组内接触关系、独立摆放和退出方向。目前不共享头；同一个刚性支撑在各 pose 安装，活动和闲置材料都须满足该 pose 的约束。
2. 每个地面搜索 3 或 4 个稍向外的脚垫，使实际接地凸包包含原始地面需求。搜索三角形/矩形、四种角度、6/18 mm 外扩量；局部脚垫为 20×14 mm、3 mm 厚、2 mm 圆角。不填满凸包内部。
3. 按 co-design 方法从头根部长出渐缩体，裁去地面和带间隙的物体/退出扫掠。完整脚垫与原根部须在同一块实材中；同一头对同一地面最多一条连接。
4. 分配脚垫与局部身体，再用少量短圆杆连接各部分。连接须真实相交并减少连通分量，没有大地面环或填充整个包络体。
5. 检查实际接地凸包覆盖、接触/根部、地面、完整退出、封闭连通和工作面；不在 Step4 重新求解反力/力矩。
6. 最后考虑适度掏空。当前两个参考设计的小脚垫放不下 4 mm 外边和 6 mm 中筋，故保持实心。

实现为 [outer_feet.py](outer_feet.py)、[run_outer_feet.py](run_outer_feet.py)。对成本排序前 12 个脚垫/身体分配进行有限搜索；失败不证明一般无解。其余六组保留原局部脚片合并/末尾掏空构造。

## 运行与图片

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo OPENBLAS_NUM_THREADS=1
/Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step4_connect_support/run_access_batch.py
```

默认对现有实体重新检查工作面、凸包包含和尺寸，并汇总保留的几何证据，模型和图片不变。`--rebuild --group pose5+7` 可重新构造；构造结果必须通过上述几何验收。当前构造器显式关闭重复力学求解，输出 `geometry_certificate.npz`，不生成虚构的反力证书。旧力学审计脚本仅供历史诊断，当前验收入口为 `run_access_batch.py`。

结果直接在各组 `step4/`：`overview.png`、`pose_N.png`、`support.png`、`shape.obj` 与 `data/`。不生成 HTML、视频、文字/箭头/坐标轴或额外几何文件夹，不在 B 根目录写 JSON。参考渲染保持灰白支撑、不同颜色的头和半透明工件。

[八组统一物体坐标撒点图](../output/B/pose5+7/step4/data/floor_demands_all_groups.png) 保持每组只有一个固定半透明物体，每 pose 全部 32,768 个地面需求逆变换到物体坐标系。独立图与来源在每组 `data/floor_demands_common_frame.*`。用户参考图和冻结的 `slides/co_design_algo/` 未改。
