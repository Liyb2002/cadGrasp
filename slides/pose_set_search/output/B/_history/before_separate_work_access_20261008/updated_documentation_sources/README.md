# 大 pose set 的共享支撑搜索

已完成 **10个7–10 pose的B集合 × whole / incremental**，20/20通过全部原始载荷的采样受力检查，所有导出mesh通过完整工作禁区几何检查。结果在 [output/B/README.md](output/B/README.md) 和 [视频浏览](output/B/index.html)。每个方法的过程图、选择记录和真实支撑mesh保存在 `output/B/{pose1+2+…}/{whole,incremental}/`。只在采样接触上检查全部原始载荷，不运行完整压力证书验收。

B原始 `needs.json` 的 `load.cone_half_deg` 为 **30°半角、60°总开角**。从完整工作面的每一点沿这个范围向外发出的射线形成半无限禁区；支撑不能触碰。旧版本只挖5mm工作带，20个旧结果全部存在实际射线阻挡，已失效；旧输入和历史结果保留，不沿用其通过结论。

[work_access.py](code/work_access.py) 用完整三角面与32边外接锥的Minkowski和保守表示禁区，径向最多额外排除0.484%。候选接触和材料占据查询同一无界半空间区域。最终三角支撑扣除相同禁区，仅在整个潜在支撑以外截断，并另查连续区域交叠。Direction改变退出路径，不改变工作法向；Juxtapose和Translation同时变换物体及其工作禁区。

目标是在满足原力／力矩及几何要求后减少支撑材料体积；支撑摆放次数不作为代价。优先让同一支撑转动复用，Direction无法解决的任务才尝试选择性Juxtapose及局部Translation／Direction。不挖横移通道，不用远距离分开所有pose作为保底。

- **whole**：全部pose以相同物体相对位置初始化，先调相近退出方向，再逐个处理困难pose，最后尝试减少材料。
- **incremental**：从一个pose开始逐个加入；旧任务和新任务全部原始载荷通过，才接纳该前缀。

不可行的临时状态可以暂时损失部分载荷，以整组未满足载荷数改善为优先。成功始终要求全部原始32,768条载荷/pose通过，保留原非负法向反力、地面摩擦及第七维不上抬条件。可行后，减体积候选必须保持全部载荷通过。详见 [reuse_first_algorithm.md](reuse_first_algorithm.md)。

搜索通过 [fast_search.py](code/fast_search.py) 和 [delta_guidance.py](code/delta_guidance.py) 维护覆盖／锁定计数，更新改变的行列；候选不重建Boolean。Direction、Translation都有有限差分候选和多种步幅，Juxtapose同时采样host、方向和偏移。相同基础预算为8轮、3个finalist、96个初筛候选；仍未通过时最多3轮额外小步修复，数值重试及后续继续搜索单独记录。LP数值问题用正列缩放、原方程的高精度非负反力回代处理，不改容差或载荷。

最终mesh直接从所选布局的5mm贴合壳并集，扣除所有活动物体、完整工作禁区和名义退出扫掠构造；不平滑voxel。展示用的真实mesh不是完整压力证书。原输入和生产 `Co-optimize/`、`idea/` 的代码保持不变。

每组两个视频：`process.mp4` 展示所选操作和实际材料增减，`result.mp4` 展示每个pose依次装入、停稳、取出。固定一个等轴测视角，先whole、后incremental，白底无文字。橙色为原工作面；停稳1秒时，工作面上显示朝内的可能施力小箭头。视频和普通过程、结果图不叠加禁区。过程图每步一个视角、无文字。

禁区单独放在一张 [环绕等轴测图](output/B/work_access_isometric.png) 中：同一pose1，每次绕45°，共8个视角，按行阅读。图中的琥珀色是原禁区与一个球形显示截断面的交集，外侧为圆弧；保持原工作三角面、法向及30°半角。这个球面只控制有限展示范围，算法排除的半无限禁区与保存的支撑均未改动。生成入口为 [render_work_access_figure.py](code/render_work_access_figure.py)。

从仓库根目录运行一个新批次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u slides/pose_set_search/code/run_fast_batch.py \
  --out slides/pose_set_search/output/new_work_cone_run --jobs 2
```

数值未解决的单次运行可独立重试；已保存搜索可通过 [repair_fast_search.py](code/repair_fast_search.py) 回代采样载荷及继续小步优化，保存旧尝试、额外时间和源代码哈希。图与视频入口为 [mesh_media.py](code/mesh_media.py) 和 [render_mesh_media.py](code/render_mesh_media.py)，支持各集合alias；禁区检查见 [audit_work_access.py](code/audit_work_access.py)。

历史完整实体验收流程、此前体积和旧图保留在 [原README](data/documentation_history/before_work_cone_20261008/README.md)、[results.md](results.md)、[reuse_first_results.md](reuse_first_results.md)。它们在完整30°工作禁区下尚未重新验收，不能作为本轮可行性证据。当前结果以 `output/B` 的新搜索及工作禁区检查为准。

完整禁区检查见 [audit](output/B/work_access_audit.md)：166次pose使用，11,267,256条射线全部无遮挡，并检查连续禁区交叠；35个回归检查通过。旧错误结果见 [历史目录](output/_history/B_before_work_cone_20261008/README.md)。
