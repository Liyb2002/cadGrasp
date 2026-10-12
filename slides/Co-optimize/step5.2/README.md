# Step5.2：确定性连接为实体支撑

入口 [run.py](run.py)，构造算法 [deterministic_base.py](deterministic_base.py)。输入是 Step5.1 的各需求凸包、Step4 的固定布局与名义支撑、保存的合法连通空间 `oracle_connected_witness.obj`。空间见证只用于裁剪，绝不把整块最大合法空间作为成品输出。

1. 将名义支撑限制在已经保存的共同合法空间中。保留其中的原材料，删除名义包裹中原本处于禁区／地下的部分；删除量单独记录，不改任何pose或退出方向。已经合法时直接保留，避免沿重合边界重复Boolean裁切产生碎片；材料保留检查复用实际起始合法实体。
2. 从支撑与需求凸包的局部范围取各最终地面之上的 5 mm 薄层；进一步按该state在自己的地面XY轴上的实际需求凸包限定选脚范围，初始向外余量5mm。检查其实际地面接触能否包围所有需求，不足时才按固定倍数扩展。避免共同坐标盒转到任务地面时产生远处选脚角点。
3. 在地面接触凸包的顶点中按固定次序删除冗余点，保留满足全部任务的少量点位。在这些点位做 5 mm 小地脚，并裁进共同合法空间；5mm 是模板尺寸，裁后允许更细的部位，没有额外安全余量或最小宽度要求。不填充虚拟凸包内部。若有限小地脚不能覆盖需求，则保留合法薄层构造。
4. 对当前各实体组件计算最近点，依次尝试短连接体。连接体同样裁进合法空间，只有真正减少组件数才加入。直连受阻时扩大局部连接盒，使其绕开被扣除的禁区；优先选择这一层尝试中新增材料最少的连接。
5. 输出一件连通、封闭的实体，以及地脚包围、合法空间包含、原合法材料保留的构造记录。

这是固定布局的确定性几何构造，有限的连接宽度序列不是 Direction／Translation sampling。它复用 Step4 的障碍几何，不重新造工作锥或退出 sweep，不做工件需求求解或 pose 微扰补救。

所有最终安装地面的半空间、物体、完整工作锥和退出路径都已包含在空间见证中，因此新增材料的裁剪同时约束所有任务。机器人拿起空支撑后的搬运转动不参与约束。这里完成的是原模型的法向地面压力包围和实体连通，不是新增一套地面摩擦／yaw／强度求解。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5.2/run.py B --experiment my_final_base --batch-name step52_my_final_base --jobs 3
```

每组最终数值与模型在`data/results/B/{pose_set}/step5/step5.2/{experiment}/`，包括`support_with_base.obj`和`data/report.json`；`overview.png`、`all_poses.png`、逐pose图与`index.html`自动发布到`output/B/`同层级。图中蓝色为实体、灰色为物体、彩色轮廓为需求凸包，无文字和工作禁区叠加。

静态空间评估在同级 [evaluation/](../evaluation/README.md)。临时执行副本与进度在校验完成后删除。

当前完成版为`clearance_compact_v1`，复用同布局同名Step5.1和`base_oracle_compact_clearance_v1`的合法空间。全部52/52通过，包含全部10个≥7-pose集合；成品连通、封闭，地脚接触凸包覆盖各任务需求。OBJ用16位小数导出，不改变布局，不重跑工件受力。[成品图集](../output/B/step5.2_clearance_compact_v1_index.html) · [构造完成清单](../data/results/B/data/step52_clearance_compact_v1_completed/summary.json)。

完整材料总量20817.49cm³，相对Step4净增3.71%。全批新增746.32cm³，裁去已非法名义材料0.78cm³，新增量中位11.63cm³／组；全部使用小地脚，没有退回整块薄层。第一次批次41.57s完成51件，pose3+15因重复裁切产生数值碎片；通用修正后仅续算此组1.17s。完成清单保留两个执行批次的来源，其余51件不重算。静态空间代价见Evaluation，不能用材料增幅替代。

旧`connected_base_v2`从较大的合法薄层外缘挑选地脚，pose1–10的4个脚离整组需求凸包约9–25cm；静态空间由Step4的54.09L增至118.51L。新构造先限制到各最终地面上的需求范围，仍按固定顺序选脚与连接，不修改物体布局或重跑工件受力。彩色环仅显示需求，不是实体底环；有限构造不是全局占地最优证明。
