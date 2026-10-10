# Step4.1 与后续 whole 搜索

1. 读取Step3.1整组注册、包裹、工作禁区与原任务元数据。
2. 在物体相对坐标初始化共同／相近退出方向，保持当前原生任务朝向与初始位置。
3. 从种子切除全部完整装卸路径，保存初始化支撑、方向、接触和原需求诊断。初始受力不足仍交给Step4.2。
4. 复用原等轴测画法，生成方向、sweep及共享支撑图片。
5. [Step4.2](../step4.2/README.md) 从保存布局执行全组Direction／XYZ Translation，连续停滞时离散Juxtapose；允许工件airborne。
6. 全部原32768需求／pose通过后，保持可行减材料，并直接保存已有mask／反力。固定布局导出名义mesh和过程／最终图。

候选接触与材料增删使用缓存，不执行Boolean；几何构造用于初始化和显示mesh。平移修改设计终点，横移轨迹不作为滑道保留。最终系统—地面与base由Step5按实际位置处理。

Step4.1初始化保持原流程，XYZ与airborne自由度在Step4.2进入优化。[当前公式与预算](../step4.2/fast_gradient_algorithm.md)
