# 当前算法的相关检查

统一XYZ／airborne与力／力矩通过即PASS的实现，在重跑前完成49项相关unittest检查。当前检查覆盖：

- 旋转host下的世界XYZ差分、独立／共同移动、同一范数和步幅。
- 不穿地投影、着地／airborne反力切换、第七方程slack及接地状态缓存。
- 全组固定需求测度、共享阻挡与接触／材料增删。
- 可行材料下降和新／旧材料选择。
- 选定布局保存时复用mask／反力；mesh导出失败不撤销PASS；正常流水线不进入几何补救。

在仓库根目录运行同一组相关检查：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -q \
  slides/Co-optimize/tests/test_xyz_translation.py \
  slides/Co-optimize/tests/test_xyz_monotone.py \
  slides/Co-optimize/tests/test_force_result.py \
  slides/Co-optimize/tests/test_stable_pipeline.py \
  slides/Co-optimize/tests/test_joint_gradient_repair.py \
  slides/Co-optimize/tests/test_stable_gradient.py \
  slides/Co-optimize/tests/test_fast_gradient_measure.py \
  slides/Co-optimize/tests/test_fast_state_seat_gradient.py \
  slides/Co-optimize/tests/test_state_seat_gradient.py \
  slides/Co-optimize/tests/test_progressive_gradient.py \
  slides/Co-optimize/tests/test_paired_blockers.py \
  slides/Co-optimize/tests/test_whole_delta.py \
  slides/Co-optimize/tests/test_strict_nominal_cache.py
```

原初始化、包裹与完整工作锥检查仍由 `test_whole_initialize.py`、`test_work_access.py`、`test_wrap.py` 等覆盖。目录中的旧模型专属检查仅对应其原接口。

七组重跑存档核对确认原输入、Step4.1冷启动数组、执行源码快照与保存mask一致，并读取导出mesh体积；没有新需求求解或Boolean。记录在 [verification.json](../output/B/data/stable_gradient_xyz_force_v3/verification.json)。

## Step5.1地面撒点

[test_system_floor.py](test_system_floor.py)的7项检查已通过：纯重力压力中心、XYZ平移与Z高度的水平力力臂、独立外力／重力平衡、支撑自重、grounded与airborne复用同一需求、正确host坐标变换，以及不合法竖直需求的显式错误。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s slides/Co-optimize/tests -p test_system_floor.py -v
```

七组输出核对保留全部62×32,768点；所有世界撒点z=0，原生到最终撒点的XYZ公式残差最多2.78e−17m。114个上游输入及100个执行源码／快照指纹保持一致。这是新地面需求的确定性投影和文件核对，没有重跑Step4反力验收。[Step5.1记录](../output/B/data/step51_xyz_floor_v1/summary.json)
