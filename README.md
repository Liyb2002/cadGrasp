# cadGrasp

共同设计一件刚性支撑，使它在不同摆放 state 下承载多个 object pose，并在满足原始力／力矩需求时尽量减少实体材料。每次装入一个物体、执行任务、取出；取出物体后可以转动或重新摆放空支撑。一个 state 可承载多个 pose，数量不设上限。

当前主线是 [Co-optimize](slides/Co-optimize/README.md) 的 whole 搜索。Direction 连续调整退出方向，Juxtapose 离散改变落座关系，Translation 连续调整已落座位置。所有 pose 从初始化开始共同参与，收益和损失均进入同一个目标。

## XYZ Translation 与 airborne

**已实现并跑通世界 XYZ 平移，允许工件 airborne。** X、Y、Z 使用同一梯度、范数和步幅池。工件朝向保持任务原值，最低点不能穿过地面；着地时可以向上移动，离地后也可以向下返回地面。

airborne 表示工件由支撑承载、工件自身不接触地面。绕质心的原始力／力矩需求和重力不变；离地后立即移除工件的四个物理地面反力列，保留原第七方程及非负 slack。需求数据不再按 grounded／airborne 分两份。最终位置改变系统对地面的力矩，Step5 按最终布局设计 base。见 [物理公式](slides/obj_supp/airborne_equations.md) 和 [Translation 实现](slides/Co-optimize/helper_func/translation/README.md)。

## 当前算法

| 阶段 | 内容 |
| --- | --- |
| Step3.1 | 整组注册到共同参考，生成贴合支撑并扣除所有工作禁区 |
| Step3.2 | 单独显示工作禁区的环绕等轴测图 |
| Step4.1 | 初始化共同／相近合法退出方向，切除完整装卸空间 |
| Step4.2 | 全组 Direction／XYZ Translation 梯度，停滞时 Juxtapose，可行后减材料 |
| [Step5.1](slides/Co-optimize/step5.1/README.md) | 按最终XYZ位置确定性重算系统—地面撒点，画出每个pose与共同支撑的点云 |
| Step5后续 | 按这些需求构造共享base，尚未实现 |

梯度对全部需求到当前非负反力锥的加权平方距离求数值差分。Direction 和 Translation 同时考虑释放与新锁定接触；Juxtapose 是有界离散搜索。局部下降及竞争分支固定需求求积点和权重，接触／材料增删使用缓存，候选不重建实体 Boolean。接触事件平台允许少量明确记录的 sampling。

每个 pose 的 **32,768 个原始力／力矩需求全部通过即 PASS**，判据使用搜索接触模型。工作锥、物体和退出通道在搜索中限制接触。选定结果后复用已有 mask／反力，仅导出固定布局名义 mesh 和图片。整件支撑安装、连通与强度留到后续阶段。

## 已完成的七组实验

`stable_gradient_xyz_force_v3` 在七组 8–10-pose 上新搜索 **7/7 通过**：6 组初次通过，1 组追加自身状态梯度通过。三个新 pose 实例离地。49 项相关检查及保存记录核对通过。

七项新方案名义材料合计 **721.55 cm³**，旧逐组最小答案 **713.01 cm³**（+1.20%）。最终择优采用三个更小的新方案、保留四个旧方案，**655.74 cm³（−8.03%）**。旧答案不作为冷启动。三进程整批含出图 **21.38 分钟**，搜索中位数 **403.6 秒／组**。

[逐组结果与来源](slides/Co-optimize/output/B/stable_gradient_xyz_force_v3_results.md) · [过程与最终图片](slides/Co-optimize/output/B/stable_gradient_xyz_force_v3_index.html) · [算法公式与预算](slides/Co-optimize/step4.2/fast_gradient_algorithm.md)

Step5.1已接入七组最终布局：按实际世界质心将原绕质心需求换算成地面压力中心，复用全部32,768样本，包含XYZ平移和airborne带来的力臂变化。不重跑Step4承载检查。默认沿用不计支撑自重的objects模型，可给定支撑／物体质量比加入自重。[逐pose撒点图集](slides/Co-optimize/output/B/step5.1_index.html) · [公式与输出](slides/Co-optimize/step5.1/README.md)

在仓库根目录运行一个新实验，显式选择 set 和新目录名：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B \
  --sets pose1+2+3+4+5+6+7+8+9+10 --jobs 1 --output-name my_whole_xyz \
  --incumbent-summary slides/Co-optimize/output/B/data/stable_gradient_xyz_force_v3/pipeline.json \
  slides/Co-optimize/output/B/data/stable_gradient_results.json
```

结果保存在 `slides/Co-optimize/output/B/{pose_set}/step4/step4.2/{output_name}/`，本轮新搜索与最终材料择优分开记录。

## 数据与相关目录

`objects/` 每个物体有原生姿态、工作面、需求和 Step2 接触候选。每个 pose 只保存 **15°／30°／60°圆锥半角** 的三组需求，每组32,768个配对力／力矩。下游复用原数据，不重新采样。代码和 Markdown 由 Git 管理，模型、图片、视频及 JSON／NPZ 结果保留在本地。

- [数据格式与预处理](codes/precompute_objects/README.md)
- [当前研究定义](slides/Co-optimize/algorithm.md)
- [三个 operations 与演示](slides/Co-optimize/operation_demo/README.md)
- [pose_set_search 的 whole／incremental 对照](slides/pose_set_search/README.md)
- [baseline 接触与结构构造](slides/baseline_algo/baseline_algo.md)
- [参数与坐标约定](slides/params/README.md)
- [后续研究方向](slides/Co-optimize/research_extensions.md)
