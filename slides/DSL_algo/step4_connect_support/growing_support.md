# Step4：在紧凑使用空间内，从接触根部长出稀疏支撑

沿用已经通过的紧凑安装摆放和 XYZ 使用空间预算。Step3 的头、方向、载荷与力学结论不变，pose1+3 不动。
在这个空间目标之内，再减少材料：从原根部、少量局部脚垫和共用短主干构造实体。

## 生长顺序

1. 保留每一个精确接触根部，作为初始身体。
2. 在每个实际可接地平面上，删去接地凸包的冗余角点，保留能够覆盖全部原 Step0 地面需求的少量脚位。
   每个脚位只长局部的 4 mm 厚脚垫，不填充地面凸包内部。
3. 根部和脚垫各自向附近合法位置长一个局部渐缩身体。原根部必须完整保留，身体必须真实连到锚点。
4. 从第一根部开始，每次将最近的未接入根部/脚垫接入现有身体。
   已有主干可以共用，路径内部不会重复生成材料。连接半径为 4 mm。
5. 合并连续的直线段，只生成局部身体和连接杆。局部裁切后，无需承担根部或脚垫的孤立碎片被丢弃。
6. 完整实体通过接触、地面、真实接地凸包、工作面、连通和 500 mm 连续退出验收后导出。
   精确 Float64 OBJ 重读再次验收；独立重放另行重新生成连续扫掠。

上一版大实体仅作为已知合法的寻路/裁切空间；**不会作为初始材料加入输出**。
生长输出仅由根部、局部脚垫、渐缩身体和连接段的并集产生，不对大实体做挖空或薄壁化。
固定旧安装摆放是为了保留已实现的小使用空间，不重新增加宽大脚位或展开支撑。

寻路采用固定 8 mm 空间网格、确定性最短路径和较大间隙的偏好，直观且快速。
网格与沿边探测只用于提出路径；连续运动安全由最终完整实体 Boolean 验证，不能以采样代替验收。
路径代价、终点排序、步长、脚位删减顺序都固定，无随机搜索。
这不是最少材料的全局证明，也没有增加新的力学通过声明；沿用 Step3 原始结论。

## 运行

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/growing_support.py \
  --groups pose3+6 pose5+7 pose2+10+15 pose2+12+15 pose1+2+8+17 pose6+8+10+19 pose1+2+3+4+5 pose2+9+13+15+17

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/replay_compact.py --source growing_support

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=slides/baseline_algo \
  .venv/bin/python slides/baseline_algo/step4_connect_support/render_compact_images.py --source growing_support

PYTHONPATH=slides/baseline_algo .venv/bin/python -m unittest step4_connect_support.test_growing_support
```

每组新模型、安装变换、接地脚位、实际生长路径和验收在 `step4/data/growing_support/`。
`shape.obj` 对应同目录 `report.json`，不是顶层保留的旧比较模型。
所有公共展示图片（overview、support 单图、全部 pose 单图）使用新生长实体重绘覆盖；
上一版大实体图片备份在 `step4/data/history/before_growing_images/`。
`boxed_support/` 的紧凑大实体与证据仍保留作空间参考及材料量比较。

## 已验证结果

八组／28 个安装姿态均通过独立导出重放，Step3 与 pose1+3 的 3222 个受保护文件未变。
材料减少 96.2–99.0%，新材料量为 62.1–204.7 cm³，使用盒没有扩大。最终缓存批次构造加内部导出重放为每组 3.9–13.5 秒。
两个回归案例检查完整绕障生长和需求保留；两姿态重跑的路径、脚位、使用盒和 OBJ 字节完全相同。

[八组材料对比](../output/B/pose2+9+13+15+17/step4/data/growing_support/batch_report.md) ·
[新展示图](../output/B/pose2+9+13+15+17/step4/data/growing_support/all_groups.png)
