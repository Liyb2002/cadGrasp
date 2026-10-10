# 共用工具

co_common.py 提供坐标变换、输入读取、实体构造、接触提取与承载检查；current_task.py 读取保存的原生任务；exit_clearance.py 提供 Step4.1 与 Step4.2 共用的退出净空构造。

当前初始化使用 optimization/initial_directions.py 的 LP／确定性半球投影。当前 Step4.2 批量入口默认 multi-step，比较方向与地面内相对平移的多步幅候选；单组 solver.py 仍需显式 --search multi-step。历史单组默认 contact-recovery 经 optimization/local_descent.py 调用 contact_recovery.py，从 Step4.1 保存方向恢复协同缺失接触。显式 local-descent 对照通过 force_descent.py 提供反力松弛目标。其他采样链和候选搜索模块保留作显式历史对照，不是当前默认。

公开入口在各 step 目录，算法以 [当前说明](../algorithm.md) 为准。测试集中于 ../tests/；缓存和历史代码、实验在 ../data/。
