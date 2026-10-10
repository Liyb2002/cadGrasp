# 当前算法测试

测试覆盖前置注册与包裹、接地围边、连续扫掠和退出净空、真实反力锥、梯度、采样调度与指定退出演示。

新 whole 初始化的直接回归测试在 `test_whole_initialize.py` 和 `test_work_access.py`：整组原生姿态对齐到首个参考 pose、保留原工作角度及完整三角形／无界外向禁区、切除后真实接触、导出网格回读的近共面边界，以及旧初始化路径兼容。`test_wrap.py` 检查原贴合包裹规则，`test_ground_rings.py` 和 `test_step41_render_reuse.py` 检查后续原几何与共享支撑画法。

新的 whole 搜索回归位于 `test_whole_invariants.py`、`test_whole_delta.py`、`test_whole_reuse.py`，加上 `test_work_access.py` 共 35 项检查：原生世界朝向／高度、原七方程严格重放、删接触后反力基失效、完整工作锥、接触／材料增删一致性、候选无 Boolean，以及实际工作检查失败不能发布 PASS。

`test_whole_pipeline.py` 另有 7 项回归：失败数不变时可改善最坏差距、whole 允许局部退步改善整组，以及采样低估实际材料时拒绝增长并保留可行基准。还检查未写完初始化报告时的断点续跑、过期批次不能写当前结果、相反地面法向的合法向上裕量、子进程成功退出不能替代真实承载证书；可行性基线共 42 项已通过。

`test_compact_volume.py` 有 6 项材料搜索回归：修复分支不能挤掉可行布局、真实体积增长／载荷失效保留原支撑、真实候选保留不同host、过程只画正确父链、候选不执行Boolean且保留全组原载荷，以及实际保存完整基线不触发字段冲突。新材料搜索独立网格／接触／全部原载荷／工作射线审计见 `helper_func/audit_compact_volume.py B --resume`。

新 Step4 保存后检查入口为 `helper_func/audit_whole_step4.py B`：核对实际导出 mesh 已有的完整连续工作锥及边界回读记录，独立检查无界工作射线、真实接触来源与原供力列、原生世界朝向／高度、过程所有 pose 同时激活、图片来源，以及原输入／Step3／历史 Step4／执行源码的指纹。尚未全部结束时可用 `--completed-only`，完成后用 `--resume` 接续已检查的集合。

旧初始化的批量实际网格检查使用 `helper_func/audit_whole_initialize.py B`：完整连续工作锥交叠、导出网格无界工作射线、原始数据／pose_set_search代码／旧 Step4 文件哈希（原命令只适用于替换 Step4 之前；本轮旧文件已完整归档），以及每组两阶段图片来源。它不进行新的搜索或完整压力证书验收。

在项目根目录运行：

```sh
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```

旧算法专属测试随旧代码归档于 `../data/code_history/`。

`test_single_pose_chain.py` 检查单 pose 扰动、合法半球、无放回选 pose、链状态延续、失败恢复及 descent 尝试/接受计数。
