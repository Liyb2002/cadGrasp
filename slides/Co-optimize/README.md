# Co-optimize：多配置共享刚性支撑

同一件支撑可有多个摆放 state，一个 state 可容纳多个 object pose。每次使用期间支撑刚性、静止；物体沿自己的方向装入和取出，取出后可换支撑摆放。当前实现针对同一物体多个任务，目标是满足原力／力矩需求并减少实体材料。

**XYZ Translation 和 airborne 已实现。** 已 Juxtapose 的 pose 可沿世界 X、Y、Z 连续调整位置，保持原任务朝向，最低点不得穿地。离地时移除工件自身地面反力，重力与绕质心的需求保持原值。最终布局对系统—地面的需求交给 Step5。见 [Translation](helper_func/translation/README.md)／[物理公式](../obj_supp/airborne_equations.md)。

| 阶段 | 职责 |
| --- | --- |
| [Step3.1](step3.1/README.md) | 整组注册、贴合包裹、完整工作禁区切除 |
| [Step3.2](step3.2/README.md) | 单独画工作禁区，圆弧显示边界和八个环绕等轴测视角 |
| [Step4.1](step4.1/README.md) | 共同／相近合法退出方向初始化 |
| [Step4.2](step4.2/README.md) | 全组 Direction／XYZ Translation 梯度、离散 Juxtapose、可行减材料 |
| [Step5.1](step5.1/README.md) | 按最终XYZ质心重算系统—地面撒点，逐pose与共同支撑图 |

whole 表示所有配置共同评价。连续操作对全部需求到反力锥的加权平方距离求导；有限 sampling 跨过接触平台，Juxtapose 改变落座结构后继续全组修复。保持原转动支撑复用，state 容量没有两个 pose 的限制。[研究定义](algorithm.md) · [公式与实现](step4.2/fast_gradient_algorithm.md) · [论文算法](step4.2/paper_algorithm.md)

每个 pose 的原32768需求全部通过即 PASS。保存复用已有 mask 和供力列；末尾仅导出固定布局名义 mesh，不做几何微扰或重复需求求解。工作／物体／退出约束在搜索接触锁中参与；base、整体连通与强度属于后续阶段。

## 当前结果

七组8–10-pose新搜索 **7/7通过**：6组初次、1组追加自身状态梯度；3个新pose实例离地。新名义材料721.55cm³，对旧713.01cm³为+1.20%。最终材料择优采用3个新答案、保留4个旧答案，655.74cm³（−8.03%）。三进程整批含出图21.38min，搜索中位403.6s／组。

[逐组结果](output/B/stable_gradient_xyz_force_v3_results.md) · [新搜索与最终图集](output/B/stable_gradient_xyz_force_v3_index.html) · [存档核对](output/B/data/stable_gradient_xyz_force_v3/verification.json)

Step5.1使用七组最终选择（3新／4保留）的实际摆放。每pose32,768点全部保留，XYZ平移通过最终质心进入地面力矩；支撑自重默认不计，可指定质量比。此阶段只给出base的总地面需求，还不构造底座实体。[撒点图集](output/B/step5.1_index.html) · [Step5](step5/README.md)

## 使用

从已有 Step4.1 启动一个明确选组的新实验：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B \
  --sets pose1+2+3+4+5+6+7+8+9+10 --jobs 1 --output-name my_whole_xyz \
  --incumbent-summary slides/Co-optimize/output/B/data/stable_gradient_xyz_force_v3/pipeline.json \
  slides/Co-optimize/output/B/data/stable_gradient_results.json
```

输出：`output/B/{pose_set}/step4/step4.2/{output_name}/`。当前七组 Step3／4.1与历史结果保留；日常运行只选择需要的 set。[初始化图](output/B/index.html) · [operation 演示](operation_demo/README.md) · [测试说明](tests/README.md)
