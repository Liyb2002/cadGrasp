# Step4.1：base 最后考虑

当前所有图片均不显示 Step3.3 接地环，以 Step3.2 wrapped_support.obj 为支撑显示范围；方向与保存的算法结果不变。exit direction / sweep 按 pose 顺序横向排布。direction space 无文字、四个合法和两个向下非法例子、被切除支撑留空；coverage 是淡色半球叠加、无箭头。

绘图记录为 data/base_deferred_render.json。本轮仅重画：未优化方向、未重跑受力、未改变保存支撑模型或 Step4.2。历史 report.json 的数值诊断仍属于原有含 base 模型，不能用于验收无 base 图片。原图保存在 data/history/before_base_deferred/。
