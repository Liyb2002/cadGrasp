# 当前研究定义：用户指定抓取、薄支撑与几何锁

用户给定物体抓取位置；抓取搜索不是方法贡献。[Step4](step4/README.md) 将固定抓取作为机器人 kinematics 约束。支撑在抓取区域提供薄材料，并设计几何锁，使物体装入后能与支撑保持为一个运动组合体。只要求一个任务姿态能够装入；各任务姿态之间需要在保持抓取的条件下可行地运动，不要求每个姿态都可退出重装。

当前只把锁当作几何保持条件，不计算锁的强度或材料失效。旧 Step4 初始化和两工具搜索分别迁到 [Step5.1](step5.1/README.md)、[Step5.2](step5.2/README.md)，代码和历史输出均保留；这些逐 pose 算法结果不代表新的一次装入加锁模型已经通过。

B `pose1+2+4+6` 的固定用户抓取输入已通过物体抓取筛选：指垫避开工作面，四个 pose 的端点 IK／手臂碰撞及准静态重力检查通过。薄支撑、几何锁、一次装入和任务间路径尚未作为整体验算。见 [研究记录](data/experiments/step4_grasp_research_B_20261007/README.md) 和 [当前设计说明](algorithm.md#新研究流程用户指定抓取薄支撑与几何锁)。

[Pose 1 Franka 抓取视频](operation_demo/grasp/vis/pose1_franka_grasp.mp4)展示固定在地板上的完整机械臂抓住物体；仅 pose1，镜头固定，画面无文字。

以下为历史逐 pose 装卸算法，迁目录不改变其原物理模型。

# 此前 Step4.2：contact-recovery

从 Step4.1 方向直接做接触恢复联合梯度，不随机 propose。联合选出补足力与力矩的接触，并减少全 pose 的扫掠阻挡；允许有界且不损失原通过载荷的平台步。真实材料重建、完整退出／1% 净空和全原始载荷决定接受，完整夹具验收仍未完成。见 [算法](step4.2/algorithm.md)。新实验默认 data/experiments/contact_recovery_new_batch/B，运行 `step4.2/run_batch.py --failed-only --iterations 18 --workers 2`。

以下为此前版本与历史结果。

# Co-optimize

以下记录描述的是现有逐 pose 方向／平移求解器及其历史实验；它尚未实现上面的一次装入、薄支撑与几何锁研究定义。旧随机多 pose 方向、单链 32 候选和不同接触锁搜索均为历史对照。

| 阶段 | 当前行为 |
| --- | --- |
| 选组 | 每 object 30 组：5 合法共同方向、20 合法无共同方向、5 非法 |
| Step3.1 | 注册到共享物体坐标 |
| Step3.2 | 共同非工作表面包裹 |
| Step3.3 | 接地环初始化，必要时向外扩大；物理检查只作诊断 |
| Step4.1 | LP 共同方向；无共同方向时用确定性半球投影求相近合法方向，构造初始切除 |
| Step4.2 | 从 Step4.1 保存方向做联合 gradient_descent，真实重建和全载荷比较决定接受 |

Step3.3 和 Step4.1 是构造／初始化阶段，不因物理诊断否决已完成构造；数值异常如实记录。Step4.2 的真实退出／载荷接受不代表完整夹具的连通、实际支撑接地或强度验收。

实现细节与实验结果：[Step4.2](step4.2/algorithm.md)。图片展示：[B 示例](output/B/README.md)。旧 28/30 等批次结果属于历史算法和当时输入，不能作为新版统计。

## 目录与运行

各 step 目录拥有运行入口；helper_func/ 放共用工具，helper_func/optimization/ 放求解器，vis_func/ 放可视化，tests/ 放测试，data/ 放缓存、其他 object 输入和历史实验。output/ 仅保留 B 的发布结果。

在项目根目录运行以下命令查看参数：

```sh
.venv/bin/python slides/Co-optimize/step3.1/run.py --help
.venv/bin/python slides/Co-optimize/step3.2/run.py --help
.venv/bin/python slides/Co-optimize/step3.3/run.py --help
.venv/bin/python slides/Co-optimize/step4.1/run.py --help
.venv/bin/python slides/Co-optimize/step4.2/run.py --help
```

新 Step4.2 实验写入新建 data/experiments 目录，保留发布结果。历史搜索模式必须显式指定；--candidates 和 --max-proposals 属于对应历史模式，不能解释成默认 contact-recovery 的候选数。

旧文档位于 [documentation_history](data/documentation_history/before_current_version_20261006/Co-optimize/README.md)。本轮只统一文档，没有运行算法批次。

## 当前重画：base 最后考虑

Step4.1 图片暂不显示 Step3.3 接地环，使用 Step3.2 包裹支撑。算法、方向和受力报告保持原样；历史数值仍对应其原有模型。绘图入口：

```sh
.venv/bin/python slides/Co-optimize/vis_func/step41_without_base.py --object B --jobs 2
```


## 扩展研究思路

[research_extensions.md](research_extensions.md) 记录 60° 载荷锥、pose 平移与材料扩展、base 后置、多物体共享、机器人 kinematics、刚度及误差。均为待研究方向，不改变当前实现。
