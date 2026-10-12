# 当前算法的相关检查

## 按pose区分的Oracle

[test_base_oracle.py](test_base_oracle.py)的13项检查包含：只显示当前任务自己的工作锥、各自世界坐标与共同材料坐标判定一致、其他最终安装地面的限制及不约束搬运中间姿态，以及原地脚余量／凸包／退出范围检查。新增检查区分物体在固定支撑内Z移动与整件支撑安装Z移动，核对前者的物体地面更新／共同地面不变，以及后者对其他任务材料的穿地限制变化。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s slides/Co-optimize/tests -p test_base_oracle.py -v
```

[test_work_access_shadow.py](test_work_access_shadow.py)的3项检查覆盖：朝外半射线拒绝工作面背面、删掉被物体遮挡的角度区域并保留可达射线、裁剪后的偏轴角度分量与实体几何判定一致。独立射线核对pose4的2,146条可达射线全部仍被禁区覆盖；少量不能连续证明的自身遮挡射线仍保守保留。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s slides/Co-optimize/tests -p test_work_access_shadow.py -v
```

## Step5.2 与静态空间评估

[test_final_base_and_evaluation.py](test_final_base_and_evaluation.py) 的14项检查覆盖静态物体与host旋转、退出方向不影响占用、无支撑叠放盒达到自由平移下界、独立pose空间与共享state空间的区别、相同变换去重与不同state计入支撑、Juxtapose与共享成员，以及小地脚、避障连接、合法材料保留和封闭单件导出。支撑pose分组检查使用不同host但相同安装变换、同一安装变换内不同落座位置及2／1／2的不同组大小，核对数量与物体pose成员。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s slides/Co-optimize/tests -p test_final_base_and_evaluation.py -v
```

当前全52组的 [只读核对](../data/results/B/data/system_connected_base_v2/verification.json) 检查实际导出模型的连通与封闭、具有正面积的地面三角形形成的包围凸包、所有地面半空间、源文件指纹及静态空间与state成员。原受力 mask 只读复用，没有新受力求解或几何 Boolean。

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


## Step5.1地面撒点

[test_system_floor.py](test_system_floor.py)的7项检查已通过：纯重力压力中心、XYZ平移与Z高度的水平力力臂、独立外力／重力平衡、支撑自重、grounded与airborne复用同一需求、正确host坐标变换，以及不合法竖直需求的显式错误。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest discover -s slides/Co-optimize/tests -p test_system_floor.py -v
```


## 输出生命周期

[test_artifact_store.py](test_artifact_store.py)验证旧路径读取、数值／可视化分离、HTML链接及中间产物清理，保证最终模型、布局和需求保留。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s slides/Co-optimize/tests -p test_artifact_store.py -v
```
