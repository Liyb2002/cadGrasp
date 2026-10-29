# 给实机执行 agent：SO101 搬运 B

本目录负责搬运臂的 bunny movement。工作臂加工区域／停点的规划仍使用原系统输出。

2026-10-26 更新到 planner 2.0，输出保持 v1 格式。已根据白臂反馈加入斜夹、多初值精修、最终计时重放后的继续搜索，以及接触区域诊断。实机工位建议先用同一站立姿态的 yaw 搬运；具体命令和参数见 [README.md](README.md)，变更说明见 [FEEDBACK_20261026.md](FEEDBACK_20261026.md)。

1. 解压整个 `bunny_movement.zip`，保留 `assets/`、`inputs/` 与 Python 文件的相对路径。先以 `--validate-only` 重放随包示例，确认数值依赖正常，再看 `example_output/preview.png`。
2. 复制 `example_request.json` 为本工位请求。将 `robot.world_from_base` 替换成搬运 SO101 相对工作臂世界坐标的实测变换；填写读回的五个关节和 jaw 当前角度、六关节符号及零点偏置。确认所有角度是 URDF 对应角度，长度是 m。
3. 从系统 `info.json` 按 task 读取源／目标 B 位姿；或用相机定位当前源位姿，填入明确的 4×4 变换。系统包平移是 mm，planner 内部是 m。夹爪 TCP 使用无针的 `gripper_frame_link`，不是加工针尖。
4. 把实际 fixture、桌面附件和停靠中的工作臂及工具加入 `obstacles`。本例没有测量这些物体，因此随包轨迹只对示例场景有效。搬运期间工作臂保持停靠静止；需要工作臂先避让的动作由工作臂执行侧完成。
5. 按真实指尖标定 `gripper` 中两个接触点、最大跨度和允许接触区域，并检查实物质量／夹持力。本实现使用原生单侧旋转 jaw，开合角不是对称夹爪宽度，也不是归一化控制量。
6. 运行 `python bunny_movement.py --request station_request.json --output station_output`，然后同样参数加 `--validate-only`。搜索失败应调整实际摆放／底座／抓取参数或增加合理搜索预算；不得把失败的 IK 或碰撞帧跳过去。
7. 读取 `program.json` 和 `trajectory.npz`。仅当 `result.json` 与 `validation.json` 通过、文件哈希吻合、初始六关节与读回状态吻合时，才进入执行。把 `positions_hardware_deg` 通过已经标定的硬件驱动接口发送；本目录不包含串口或电机指令。
8. 先相机／模型干跑，再低速运行。`close` 结束时确认抓住 B，才能进入 `extract`；`insert` 结束时确认 B 已由目标桌面／fixture 支撑，才能进入 `release`。阶段边界是静止点，适合执行侧等待确认与回读。若实际状态偏离计划，应停止并重规划。

输出关节顺序始终是：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。

不会自动移动工作臂、fixture 或搬运臂底座，也不会在携物途中重抓。若两个目标姿态无法用五轴 SO101 的同一个抓取连接，planner 会失败；需要人工选择中间支撑／重抓方案后拆成两次请求。
