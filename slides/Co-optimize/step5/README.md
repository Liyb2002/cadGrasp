# Step5：从合法布局构造实体 base

Step4 已在搜索中检查最终 XYZ 位置的需求包围与共同连通空间。Step5 冻结这个布局，不再搜索摆放，不重跑原工件力／力矩需求。

| 阶段 | 内容 |
| --- | --- |
| [Step5.1](../step5.1/README.md) | 按每个任务的最终世界质心计算全部地面需求和凸包；画出逐 pose 与共同支撑视图 |
| [Step5.2](../step5.2/README.md) | 在 Step4 保存的合法空间中确定性选择小地脚、连接为一件实体支撑 |
| [Evaluation](../evaluation/README.md) | 静态独立pose空间与共享state空间；同一最终布局下物体、Step4、Step5.2 的整体BBX |

主线在 Step5.2 完成实体，随后单独评估。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5/run.py B --experiment my_final_base --jobs 3
```

默认读取 `data/results/B/data/base_oracle_results/summary.json` 的全部 52 个通过布局，包括原 `illegal/` 组。三个阶段分别输出在每个集合的 `step5/step5.1/{experiment}`、`step5/step5.2/{experiment}` 和 `evaluation/{experiment}`。完成阶段可依据保存的上游指纹复用，未完成批次不能冒充通过。

沿用 objects 和 Step4 的支撑自重为零、地面法向压力包围模型。工件着地时自身地面接触也可参与包围；airborne 时没有这部分接触。地脚的虚拟凸包可以有空洞，不需要铺满或沿整个边界造实体。

当前完成的执行链为 Step5.1 `connected_base_v1` → Step5.2 `connected_base_v2` → Evaluation `connected_base_v2`，全部52组。[执行记录](../data/results/B/data/system_connected_base_v2/pipeline.json) · [实际mesh／地脚／BBX核对](../data/results/B/data/system_connected_base_v2/verification.json)。

数值与模型保存在`data/results/B/`；最终图片与HTML在`output/B/`的同层级目录。候选、帧、日志、进度与已核对的执行副本用完删除。
