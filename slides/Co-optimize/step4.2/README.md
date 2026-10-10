# Step4.2：全组梯度与离散 Juxtapose

从保存的Step4.1开始，所有pose共同优化。Direction调整退出方向，Translation调整已Juxtapose位置，Juxtapose在连续下降停滞时离散改变落座结构。可行后继续减少实体材料、恢复转动支撑复用。每个state可服务多个pose。

**Translation支持世界XYZ，允许airborne。** 三轴共用梯度、范数与步幅；只有地面边界施加最低点高度非负约束。离地后移除工件地面反力，保留重力、原质心需求和第七方程slack。[Translation](../helper_func/translation/README.md) · [物理模型](../../obj_supp/airborne_equations.md)

## 流程与预算

入口为 `run.py` → `stable_pipeline.py`。初次全组搜索最多10轮结构跳步，每轮96个廉价落座候选、3个完整分支，各分支最多2轮Direction／XYZ Translation修复。局部下降和竞争分支固定同一需求求积点和权重。

仅对仍有原需求未满足的组，从其本轮布局追加最多8轮保持state落座和全组梯度修复。全部可行后运行两轮保持可行的Direction／XYZ材料下降。少量sampling跨过接触平台，接受日志区分gradient与sample。

PASS使用搜索接触模型上每pose全部32768原始需求。最终保存复用已有mask／供力列，不进入几何微扰或末尾重复需求求解；固定布局名义mesh导出失败也不撤销力／力矩PASS。无实体mesh的估计体积不能替换旧实体答案。最终材料择优分别记录新尝试与旧答案来源。

[详细公式与实现](fast_gradient_algorithm.md) · [论文算法](paper_algorithm.md)

## 已完成结果

七组8–10-pose新搜索7/7通过：6组初次、1组自身状态梯度接续；3个新pose实例离地。七项新名义mesh721.55cm³，对旧713.01cm³大1.20%。最终采用3个更小的新方案、保留4个旧方案，655.74cm³（−8.03%）。三进程整批含出图21.38min，搜索中位403.6s／组，名义mesh导出中位4.5s。

[新搜索与最终结果表](../output/B/stable_gradient_xyz_force_v3_results.md) · [图片浏览](../output/B/stable_gradient_xyz_force_v3_index.html) · [保存记录核对](../output/B/data/stable_gradient_xyz_force_v3/verification.json)

49项相关检查通过；两阶段各65份执行源码／快照一致，七组冷启动与原Step4.1数组一致，原始需求mask通过。存档核对只读取文件、哈希和mesh体积。

## 使用与输出

在仓库根目录运行，选择已有set和新的输出名：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B \
  --sets pose1+2+3+4+5+6+7+8+9+10 --jobs 1 --output-name my_xyz_gradient \
  --incumbent-summary slides/Co-optimize/output/B/data/stable_gradient_xyz_force_v3/pipeline.json \
  slides/Co-optimize/output/B/data/stable_gradient_results.json
```

每组输出在 `output/B/{pose_set}/step4/step4.2/{output_name}/`：

- `layout.npz`、`*_force.npz`、`data/report.json`：选定布局、完整需求mask和供力列。
- `support.obj`、`process.png`、`final_result.png`：固定布局名义mesh与无文字等轴测图片，工作禁区只在Step3.2画。
- `process.json`、`process_states/`、`mesh_states/`：已接受操作和布局过程。
- `material_selection.json`：最终所选来源，可能保留旧答案。

批次数据在 `output/B/data/{output_name}/`，保存预算、执行源码、阶段通过数、原输入保护和新／旧材料比较。Step5按最终位置处理系统—地面与base；整件连通和强度仍为后续工作。
