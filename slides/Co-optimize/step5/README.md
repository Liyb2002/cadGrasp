# Step5：从合法布局构造实体 base

Step4已检查最终布局的需求包围与共同连通空间。Step5冻结布局，不搜索摆放、不重新采样需求。新Step5.2删除接触材料时检查全部原需求，Step5.3构造后确认真实接触仍满足相同需求。

| 阶段 | 内容 |
| --- | --- |
| [Step5.1](../step5.1/README.md) | 按每个任务的最终世界质心计算全部地面需求和凸包；画出逐 pose 与共同支撑视图 |
| [Step5.2](../step5.2/README.md) | 删减接触材料与多余独立块，并检查全部原始需求 |
| [Step5.3](../step5.3/README.md) | 在保存合法空间中生成地脚、连接完整实体 |
| [Evaluation](../evaluation/README.md) | 静态独立pose空间与共享state空间；同一最终布局下物体、Step4、Step5.3 的整体BBX |

主线先删减，在Step5.3完成实体，随后评估。旧批次保留原Step5.2构造身份；当前图片直接位于每组的step5.2/和step5.3/，无实验子目录或HTML。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5/run.py B --experiment my_final_base --jobs 3
```

默认读取 `data/results/B/data/material_output_all52_20261012_01/batch.json` 的全部52个保存布局，包括`illegal/`组。复用已保存的Step5.1 `material_output_all52_20261012_01`，新阶段数值分别输出在每个集合的`step5/step5.2/{experiment}`、`step5/step5.3/{experiment}`及`evaluation/{experiment}`。完成阶段可依据保存的上游指纹复用；有失败记录的批次不能冒充全部通过。只运行删减与构造时加`--skip-evaluation`。

沿用 objects 和 Step4 的支撑自重为零、地面法向压力包围模型。工件着地时自身地面接触也可参与包围；airborne 时没有这部分接触。地脚的虚拟凸包可以有空洞，不需要铺满或沿整个边界造实体。

历史执行链 Step5.1 `connected_base_v1` → 旧Step5.2 `connected_base_v2` → Evaluation `connected_base_v2`完成52组。[历史执行记录](../data/results/B/data/system_connected_base_v2/pipeline.json) · [历史mesh／地脚／BBX核对](../data/results/B/data/system_connected_base_v2/verification.json)。当前新Step5.2/5.3按用户要求扩展到全部52组，实际通过情况以新批次记录为准。

数值与模型保存在`data/results/B/`；最终图片直接位于`output/B/[set]/step5.2/`和`step5.3/`，不生成HTML或额外实验图片目录。
