# Step4：只计算地面需求点

当前入口为 `whole_assembly.py`，数据 schema 为 `whole_assembly_floor_demand_points_v3`。Step4 不再生成支撑多边形、闭合圈或实际底座。

对每个加工载荷加重力，保留完整合力与合力矩，计算 y=0 的压力中心。输出包括物理载荷的落点 `floor_demands_xz_m`、连续载荷保守外包映射得到的点 `continuous_floor_enclosure_xz_m`、对应完整载荷以及工件原接地点。青色外包点不冒充实际加工样本。

连续外包用于 Step5 验收，不能只包住有限的橙色样本就声称覆盖连续载荷。正的总法向力使线性分式映射保持凸包含；审计独立用外力作用位置与重力重算落点，再检查完整力矩关系。

Step5 负责把这些点取需求凸包、设计真实接地材料与开口、连接全部头、加粗连接、找轨迹及验证整件承载。Step4 的需求点计算通过，不代表任何支撑几何已经通过承载。

```sh
python slides/baseline_algo/step4_floor_contact/rerun_step4.py --workers 3
```

当前撒点结果在 `output/{object_name}/{pose}/step4_floor_contact/`。不创建汇总或备份目录。
