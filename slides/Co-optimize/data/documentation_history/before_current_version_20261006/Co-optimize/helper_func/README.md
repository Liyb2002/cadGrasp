# 共用工具

`co_common.py` 提供坐标变换、输入读取、实体构造、真实接触提取及承载检查；`current_task.py` 读取保存的原生任务；`exit_clearance.py` 提供 Step4.1 与 Step4.2 共用的退出净空构造；`run_all.py` 运行注册与包裹前置步骤。

`optimization/` 集中当前 Step4.2 使用的采样、短梯度分支、反力锥投影、距离场和求解器基类。公开求解入口在 `../step4.2/`，内部层不作为独立算法入口。

旧实验与诊断脚本归档于 `../data/code_history/`；测试统一在 `../tests/`。
