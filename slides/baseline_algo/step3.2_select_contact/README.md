# Step 3.2：选择本轮最大联合贡献的候选

读取 Step 3.1 本轮的完整贡献表，按联合满足需求数降序选择一块。此前接触固定，因此最大联合覆盖也就是最大新增覆盖。并列按候选 ID 排序，所有并列项均保留。本步不调整尺寸。

第一轮和后续轮次调用同一个 `choose.run()`；不再单独设置“选第二块”的算法。零即时增益仍允许选择，避免把需要多个接触共同发挥作用的情况误判为无法继续。没有剩余候选时返回空选择，由调度器记录未完成。

每轮输出到 `output/<物体>/<pose>/step3.2_select_contact/round_001/` 等目录：

- `selection.json`：本轮选择、完整排名和并列名单。
- `selected_contact.npz`：本轮选择的原始 Step 2 接触区。
- `contacts_before_optimization.npz`：此前全部固定接触加当前候选。
- `sample_coverage.npz`：此前组合与加入候选后的需求掩码。
- `selection.png`、`ranking.png`、`selection_views.json`：位置、接触近景、联合贡献排名与绘图来源。

图中蓝色表示此前固定的接触，橙色表示当前选中接触。这里的比例属于固定接触反力模型，不能据此声称已经构造出可落地、可插入的支撑实体。

统一运行入口见 [Step 3 scheduler](../step3_scheculer/README.md)。
