# Co-optimize

## Step4.2 主算法与结果

**Sampling 搜索大方向，物理反馈梯度做局部调整。** 当前主入口为 [solver.py](step4.2/solver.py)，继承 [混合算法](step4.2/algorithm.md)，并加入编译距离场及保守的膨胀扫掠数值修复。

保存的完整批次：B **28/30**，其中 2 组初始通过、26 组恢复；其他 20 个对象各固定一组 5 poses，**6/20** 通过。验收包含全部原始载荷、完整退出与每侧 1% 净空；本轮暂缓连通及完整夹具验收。

结果目录只有 [output/B/](output/B/README.md)，主算法结果统一放在各组的 `step4/step4.2/`。历史实验、其他对象测试和缓存保存在 `data/`。

[![Step4.2 算法](step4.2/algorithm.png)](step4.2/algorithm.png)

## 目录

| 目录 | 内容 |
| --- | --- |
| `step3.1/` | pose 注册 |
| `step3.2/` | 共同非工作表面包裹 |
| `step3.3/` | 最小凸包围边 |
| `step4.1/` | 退出初始化 |
| `step4.2/` | 采样与梯度联合优化退出方向，检查承载和净空 |
| `helper_func/` | 共用输入、几何、物理工具与批次辅助 |
| `vis_func/` | 绘图、视频与图形辅助 |
| `tests/` | 当前算法与前置步骤的测试 |
| `data/` | 内部缓存、历史实验与目录迁移记录 |
| `output/` | 仅 B 的正式结果 |

## 运行入口

在项目根目录运行：

```sh
.venv/bin/python slides/Co-optimize/step3.1/run.py B --jobs 2
.venv/bin/python slides/Co-optimize/step3.2/run.py --help
.venv/bin/python slides/Co-optimize/step3.3/run.py --help
.venv/bin/python slides/Co-optimize/step4.1/run.py --help
.venv/bin/python slides/Co-optimize/step4.2/run.py --help
```

确认算法的批次入口：`step4.2/run_batch.py`，默认使用 12 轮配置和 1,200 个采样提案。新实验默认写入 `data/experiments/selected_hybrid_new_batch/B/`，防止覆盖已有结果。
指定演示入口：`vis_func/animate_exit_directions.py --set pose2+3+4+7`。

算法说明见 [step4.2/algorithm.md](step4.2/algorithm.md)。当前代码只保留确认算法与前置步骤；内部基类位于 `helper_func/optimization/`，测试集中于 `tests/`。旧算法、review、pilot 和泛化脚本在 `data/code_history/`，历史运行结果在 `data/experiments/`。目录迁移不代表重新求解或重新验收。

运行已有测试：

```sh
.venv/bin/python -m unittest discover -s slides/Co-optimize/tests
```
