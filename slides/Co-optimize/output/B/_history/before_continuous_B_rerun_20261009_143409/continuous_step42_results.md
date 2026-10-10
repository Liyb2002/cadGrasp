# 连续 Step4.2：两组短试跑

本次只替换 Step4.2 的活动入口。Step3、Step4.1、原物理模型、全部原始载荷和已有结果保留。新入口调用三个独立 operation：Direction 与 Translation 对整组收益损失做中心差分和步长回溯，停滞时 Juxtapose 做离散落座跳步，并在跳步内部调用两种连续更新。

每组只运行两轮连续更新，至多一次 Juxtapose、两个离散落座、一次跳步内部修复、三次真实支撑检查。两个例子的连续更新均被接受，未触发主循环 Juxtapose。

| pose set | Step4.1 实体体积 cm³ | 新实体体积 cm³ | 减少 | 含绘图耗时 | 原始载荷重放 |
|---|---:|---:|---:|---:|---:|
| pose19+28 | 139.831 | 139.230 | 0.43% | 约32秒 | 65,536 / 全通过 |
| pose4+5+7 | 90.338 | 81.960 | 9.27% | 约25秒 | 98,304 / 全通过 |

体积指支撑本身的实体材料体积，不是包围盒或物体体积。最终实际网格均通过原载荷、完整工作禁区、退出、接触核心和原1%净空检查；独立从最终网格重算接触后，重放全部163,840个原始载荷，全部通过。pose19+28的完整更新未通过原导出排除检查，沿同一更新缩短一半后通过；这两次真实检查都计入预算。pose4+5+7第一项真实候选通过。

| pose set | 过程图 | 最终所有使用配置 | 数值结果 |
|---|---|---|---|
| pose19+28 | [process.png](pose19+28/step4/step4.2/continuous/process.png) | [final_result.png](pose19+28/step4/step4.2/continuous/final_result.png) | [report.json](pose19+28/step4/step4.2/continuous/data/report.json) |
| pose4+5+7 | [process.png](pose4+5+7/step4/step4.2/continuous/process.png) | [final_result.png](pose4+5+7/step4/step4.2/continuous/final_result.png) | [report.json](pose4+5+7/step4/step4.2/continuous/data/report.json) |

过程图采用同一参考支撑姿态、固定等轴测镜头，蓝色为支撑、灰色为物体、橙色为原工作面、绿色为新增材料、红色为删除材料；最后一格为真实接受网格，无图内文字和禁区覆盖。

连续目标来自原 `needs.json`：保留作用点、加工力方向和力矩的耦合，积分可支撑加工力大小区间。位置和方向的确定性求积、移动接触边界均有近似误差，粗细两级积分覆盖不能证明整个连续需求域严格全覆盖。原32768个保存载荷仅用于最后实际接受，不被替换或重新采样。

另一个独立接口检查实际评估了两个Juxtapose落座，在选中分支内部接受了Direction和Translation更新，但整个分支相对原基线仍较差，所以拒绝；该检查不是成功修复一个失败集合的证据。记录在 [operations_smoke.json](data/continuous_step42_operations_smoke.json)。

56项回归测试通过；8,985个受保护旧文件保持一致。[独立核对结果](data/continuous_step42_verification.json)保存原始载荷重放、真实体积、积分检查及保护范围。[当前算法与代码入口](../../step4.2/continuous_algorithm.md)包含完整流程和默认预算。

两个短试跑只证明新流程运行并接受了有效更新；尚未在原十个7～10-pose集合上做同预算新旧对照，因此不声称新版本整体更快或更紧凑。此前42组及十组的候选采样／greedy／beam结果属于旧版本。
