# Step4.2：连续调整与离散落座跳步

Step3和Step4.1保持原样。入口 `run.py` 调用 `optimizer.py`；后者组织三个helper，数学和几何评价在共享模块中实现。

```text
读取整组Step4.1的方向、布局和实际检查结果
重复：
    direction_choice：全组连续覆盖／材料目标的梯度更新＋步长回溯
    translation：已Juxtapose配置的水平梯度更新＋步长回溯
    若两者停滞：
        juxtapose：有界离散落座选择
        在选中跳步内部调用direction_choice和translation做局部修复
        比较修复后的整组目标，决定是否进入新布局
    若仍停滞或预算结束：退出
细一级积分检查选中布局
少量候选重建真实支撑，检查全部原载荷／工作禁区／退出
若初始Step4.1真实可行，保留它；新解实际可行且更小才能替换
```

所有pose从开始共同参与。修复目标是各pose按原连续需求分布积分的未覆盖比例平均值，材料估计只带极小正则项。全组积分覆盖后，下降材料体积，并保持每个pose的积分覆盖条件。Direction与Translation交替更新同一个全组目标；Juxtapose改变离散host与相对落座，帮助离开局部最优。没有随机设计候选池；中心差分、步长回溯及离散host选择仍需要有限次数的评价。

连续覆盖基于原作用点／方向／力大小耦合，力大小一维通过凸锥与仿射线段的交集区间消去。位置／方向采用确定性求积；接触几何是原三角形上移动边界的分片线性近似。两者均有数值误差，连续域全覆盖尚未获得严格证明。全部旧载荷和原物理容差用于最后的真实接受，原模型没有新增反力上限或面积压力容量。

默认6轮连续更新、最多2次离散跳步、每次至多4个落座、1轮跳步内部修复、4次步长回溯、至多3次实际网格检查。实际检查失败时，在保持host的连续更新上缩短同一步长；每次检查计入这3次预算。新布局还须通过细一级连续求积检查。预算在trace和report中保存。未通过实际检查的候选不能标为成功，也不能替换真实可行基线。

三个接口：

- [direction_choice](../helper_func/direction_choice/README.md)
- [translation](../helper_func/translation/README.md)
- [juxtapose](../helper_func/juxtapose/README.md)
- [共享连续覆盖／接触边界](../helper_func/continuous_support/README.md)

输出位于 `output/B/{set}/step4/step4.2/continuous/`，既有Step4.2和greedy/beam结果不被覆盖。保留过程图、最终真实蓝色支撑图、父链布局、梯度与回溯记录、粗细积分检查、实际网格验证及执行源码。复用现有无文字等轴测画法；过程的连续覆盖近似和最终原载荷结果明确区分。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B --sets pose19+28
```

旧 `whole_pipeline.py` 的Step4.2、`compact.py`及其论文描述保留供已经完成的采样候选实验复现。其十组成功率／体积不属于本次连续版本。
