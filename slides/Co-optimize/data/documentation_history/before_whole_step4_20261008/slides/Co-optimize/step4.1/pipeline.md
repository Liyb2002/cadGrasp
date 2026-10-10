# Step4：退出路径与切除材料共同优化

Step4.2 当前仅保留本次 `pose2+3+4+7` 的 `exit_direction_changes.mp4` 与配套 `exit_direction_changes_poster.png`。其余各组及 data 历史目录的旧图片已按用户要求删除；模型和原始数值记录保留，旧报告中的图片路径／哈希为历史记录。删除清单见 `output/B/data/step42_visual_cleanup.json`。


已实现 `step4.1/run.py` 初始化。Step4.2 同时优化退出路径、全部原始力／力矩需求和材料连通；各组当前结果以保存的接受记录为准。

当前 Step4.1 使用确定性初始化：先求所有 pose 地面允许半球的非零交集，严格共同方向和仅边界共同方向均保留。若不存在共同方向，则从固定球面起点开始，交替将参考方向投影到各 pose 合法半球、更新共同参考方向，选择方向分散度最小的已搜索结果。该多起点局部算法不保证全局最优；方向相近不代表承载一定通过。每组方向单独计算和缓存，随后保留原始全载荷检查；失败才进入 Step4.2。历史 native +z 结果不自动重跑。

优先读取 objects/<name>/selected_pose_sets.json；缺失时兼容历史集合读取。prepare(name, groups=...) 只处理明确选择且已有 Step3.3 前置材料的组，不生成缺失前置材料。

用完整物体连续平移扫掠切除 Step3.3 材料，每次都从不可变 Step3.3 初始材料重新计算，因此后续改路径可以恢复材料。退出长度至少 500 mm，且保证末端物体投影完全离开初始支撑。图片显示前 100 mm，不用图片长度进行切除。

运行 `.venv/bin/python slides/Co-optimize/step4.1/run.py` 处理全部 B 保存组。每组唯一图片位于 `output/B/<set>/step4/step4.1/overview.png`，同图按各 pose 展示青色扫掠、红色切除材料和灰色剩余材料。模型为 `removed_support.obj` 和 `remaining_support.obj`。

从实际剩余材料内边界重新提取接触并检查全部原始力／力矩需求，同时记录切除后的实际接地凸包覆盖，作为后续重建接地材料的诊断；不要求保留固定圆环，也不以原环损失判定路径失败。它们是初始化诊断；地面穿透、未连接材料、强度及机器人运动没有接受证书。初始失败不是总体无解，Step4.2 调整路径恢复承载。

Step4.2 运行 `step4.2/run.py`，查看各组 `step4/step4.2/overview.png`。本阶段要求一个连通的正体积支撑，不要求保住原环；无用分量须经过全部原始载荷检验后删除，有价值的分离块通过修改路径恢复连接。绿色标出恢复材料。

当前 Step4.1 配图：运行 `.venv/bin/python slides/Co-optimize/step4.1/run.py --render-only` 处理正常和非法共 30 组。灰色物体、蓝色完整 Step3.3 支撑，红色仅表示当前 pose 自己退出时切掉的材料。每格从完整支撑独立计算，不累计其他 pose 切除；正交等轴测视角，无文字和扫掠体。当前图片记录为 `data/render.json`，原力学报告和共同切除模型仍为历史结果，未对新凸包支撑重跑。

最新 Step4.1 配图改为三张：`exit_directions.png`（每格自己的退出箭头与红色切除）、`final_result.png`（共同切除后的最终初始化支撑）、`exit_sweeps.png`（透明 sweep 与红色相交材料）。全部灰色物体、蓝色支撑、正交等轴测视角。前 100 mm 仅用于 sweep 展示，切除使用完整退出。更新命令仍为 `step4.1/run.py --render-only`。旧 overview.png 已删除。

退出通道统一按 `../exit_clearance.py` 留每侧 1% 原始物体最大尺寸的余量（B：1.548 mm）。Step4.2 每次改变方向都会重新扣除全部当前膨胀扫掠；支撑恢复必须服从同一约束。仅实际承载接触下的无碰撞材料核例外保留接触。历史无余量接受记录需要重新求解。
