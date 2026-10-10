# Step3.1：whole 注册与工作禁区初始化

合并原 Step3.1 的注册和原 Step3.2 的贴合包裹。每组所有 pose 对齐到保存顺序的第一个 pose，生成同一件蓝色贴合支撑，再扣除全部工作面向外的完整禁区。B 沿用 `output/B/` 的 42 个已有集合，包括当前选组清单中的 30 组；不生成新 pose set。

设原始物体到世界的变换为 $T_i$，参考 pose 为 $r$，世界注册变换为 $T_rT_i^{-1}$。保存原始变换、注册变换及逆变换；内部物体和支撑仍用原 mesh 坐标（米）。同一支撑可随空支撑的摆放转回各原始任务。

工作禁区包含每个原始工作三角形的全部点，以及法向向外 30° 半角范围内的全部无界射线（B 原始设定，总开角 60°）。采用 pose_set_search 的 32 边外接锥，最大径向保守量 0.484%。5 mm 贴合包裹扣除这些禁区，Boolean 远端盖面始终超过整个未切种子。不是只在工作面挖一个薄孔。

输出在 `output/B/<set>/step3/step3.1/`：

- `overview.png`：沿用原支撑绘图算法；蓝色真实三角网格支撑、灰色物体，按各原始 pose 展示。
- `registered_object.obj`、`wrapped_support.obj`：实际注册物体和工作禁区切除后的支撑。
- `data/contacts.npz`：从最终内壁提取的真实接触三角形及原始面来源。
- `data/report.json`：注册、几何与原始载荷诊断；各 pose 另存 `force.json` 和 `coverage.npz`。

所有原始 32,768 载荷/pose 都沿用，不改原反力、接地或第七维不上抬模型。当前接触受力失败仅针对这一固定重合布局，不丢弃整组；后续 whole 搜索可调整方向或布局。`initialized` 表示几何初始化完成，退出路径、接地环及最终共享支撑尚未求解。

几何和实际 float64 网格回读都检查物体／完整禁区交叠，原体积容差 $10^{-10}\,\mathrm m^3$ 不变。必要时用相同禁区处理近共面 Boolean 边界，并重新提取接触；不放宽禁区或载荷。

在项目根目录运行，默认同时生成新 Step3.1 和 Step3.2 图片：

```sh
.venv/bin/python slides/Co-optimize/step3.1/run.py B --jobs 2
.venv/bin/python slides/Co-optimize/step3.1/run.py B --sets pose1+2+4+6
```

`--scope selected` 仅跑当前选组清单；默认 `--scope existing` 沿用已有输出集合。`--resume` 检查当前报告及依赖后复用完成记录。旧 Step3 和旧代码保存在 `output/B/_history/before_whole_initialize_20261008/` 与 `data/code_history/before_whole_initialize_20261008/`；原 Step4 文件是历史结果，须从新初始化重新生成。
