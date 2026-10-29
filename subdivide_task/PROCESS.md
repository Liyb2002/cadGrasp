# Subdivide task 流程（2026-10-24）

`run_big.py <run>` 依次运行下面各步，每步一个日志 `data/<run>/logs/<step>.log`：

1. **Step1**：`step1.subdivide`，分成座位组和 task，精确认证后从磁盘重放核对，出图（step1.1–1.3、groups.png、subdivision.json）。
2. **Step3**：`step3.run`，为每个 task 构造 5 mm 接触薄壳，并做独立复验。
3. **Step4 禁区**：`step4.keepout`，生成针、头、KUKA 禁区，并从磁盘重建核对。
4. **Step4 合并**：`step4.combine`，32 个随机样本，按座位组抽样（见 README）。
5. **Step5**：`step5.combined`，加宽、裁剪、减材、地脚；每个成品 `rNN.png` 旁边有 `rNN/` 文件夹。
6. **Evaluation**：`evaluation.combine`，汇总材料体积、包围盒、座位和操作数量。

可达 screen 只取决于物体和工具，与 `whole_task_2` 的 B 缓存相同（`reach.py` 字节一致，缓存 key 已核对），所以从 `data/B/step1_reach/screen.npz` 复制到每个 run。
