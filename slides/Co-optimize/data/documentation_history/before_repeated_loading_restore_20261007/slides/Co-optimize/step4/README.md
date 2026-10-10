# Step4：用户指定抓取，kinematics 作为约束

抓取位置由用户给定，求抓取不是方法贡献。抓取接口包含相对位姿
`T_object_hand`（手坐标映射到物体公共坐标）和夹爪开口。位置一旦作为输入确定，
支撑优化不再移动它来获得通过。示例位置可由人工、现成抓取工具或辅助搜索给出；
这些只准备输入。支撑需要在该抓取区域提供薄材料，同时避开指垫的实际接触面。

固定抓取的约束为

$$T_{world,hand}(t)=T_{world,object}(t)\,T_{object,hand}.$$

`T_object_hand` 固定，物体路径 `T_world_object(t)` 可平移、转动和分段变化。
机器人关节需要实现该手部位姿，并满足关节范围、碰撞与工作面避让。因此固定抓取
不等于固定装入方向或直线路径。`kinematic_constraints.py` 接受完整 SE3 waypoints，
可供 Step5 的初始装入路径和任务姿态间路径验算。装入后物体由支撑上的几何锁保持；
锁的强度当前不计算。

B 的固定输入已通过四个任务姿态的物体抓取端点检查；同一抓取下的自由空间分段平移＋20°转动
也通过采样 IK／碰撞检查。它不包含最终支撑，实际插入仍由 Step5 回验。

```sh
.venv/bin/python slides/Co-optimize/step4/validate_user_grasp.py \
  --grasp slides/Co-optimize/data/experiments/step4_grasp_research_B_20261007/example_user_grasp.json \
  --out /tmp/cadgrasp-fixed-grasp-validation
```

输出是约束和已验证状态，不是最终夹具通过证书。薄支撑、几何锁、一次装入及
任务间路径仍需 Step5 用实际几何回验。`single_load/research_step4.py` 保留为准备
示例输入的辅助工具，不作为新 Step4 的抓取优化算法。

旧的 Step4 初始化和两工具搜索分别迁至 [Step5.1](../step5.1/README.md)
和 [Step5.2](../step5.2/README.md)，原入口为兼容别名，历史结果不删除。
