# B 初始化历史

当前结果是 `output/B/<pose-set>/step3/step3.1/` 的 whole 注册／贴合支撑／完整工作禁区切除，以及 `step3/step3.2/` 的禁区可视化。

- `before_whole_initialize_20261008/`：本轮修改前的全部 42 组 Step3、当时存在的 Step4，以及旧根目录索引与数据。旧 Step3.2 是包裹计算，旧 Step3.3 是接地环，与当前阶段定义不同。对应旧代码在 `Co-optimize/data/code_history/before_whole_initialize_20261008/`。
- `whole_initialize_before_export_guard_20261008/`：第一次 42 组新初始化及其真实源码快照。导出回读的连续 Boolean 检查在 `pose6+10+13+17+30` 出现近共面边界数值不稳定；随后加上相同禁区下的导出边界保护，并重新生成全部 42 组。不要用这批记录替代当前完成检查的结果。

历史报告的依赖路径和哈希反映其运行时版本，不能指向现在的新 Step3 文件解释。原 case-level Step4 文件保持原字节，仍属于旧初始化；本轮没有重算 Step3.3 或 Step4。
