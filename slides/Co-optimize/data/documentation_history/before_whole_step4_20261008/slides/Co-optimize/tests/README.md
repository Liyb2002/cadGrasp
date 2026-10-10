# 当前算法测试

测试覆盖前置注册与包裹、接地围边、连续扫掠和退出净空、真实反力锥、梯度、采样调度与指定退出演示。

新 whole 初始化的直接回归测试在 `test_whole_initialize.py` 和 `test_work_access.py`：整组原生姿态对齐到首个参考 pose、保留原工作角度及完整三角形／无界外向禁区、切除后真实接触、导出网格回读的近共面边界，以及旧初始化路径兼容。`test_wrap.py` 检查原贴合包裹规则，`test_ground_rings.py` 和 `test_step41_render_reuse.py` 检查后续原几何与共享支撑画法。

批量实际网格检查使用 `helper_func/audit_whole_initialize.py B`：完整连续工作锥交叠、导出网格无界工作射线、原始数据／pose_set_search代码／旧 Step4 文件哈希，以及每组两阶段图片来源。它不进行新的搜索或完整压力证书验收。

在项目根目录运行：

```sh
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

旧算法专属测试随旧代码归档于 `../data/code_history/`。

`test_single_pose_chain.py` 检查单 pose 扰动、合法半球、无放回选 pose、链状态延续、失败恢复及 descent 尝试/接受计数。
