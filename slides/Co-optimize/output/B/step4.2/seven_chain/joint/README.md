# 转动复用优先，再选择性 Juxtapose

集合：pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, pose_7。方法：joint。材料体积 **109.031 cm³**。保留转动复用 5 个 pose；Juxtapose 2 个 pose。

目标是在全部原始力／力矩、退出和工作面约束满足后减少实体支撑体积；支撑摆放数不作惩罚。每格都是同一份蓝色刚性支撑在保存的支撑摆放下、灰色物体在保存的相对配置中，没有为了图像删掉材料。

[最终全部 pose](final_result.png) · [退出路径](exit_sweeps.png) · [原始结果](data/report.json) · [实体支撑](data/support.obj) · [全部非负反力证书](data/pressure_audit.json)。

路径图按 Step4.1 风格显示前 100 mm；实际验收完整路径为 619.1 mm。每个 pose 验收全部 32,768 个原始载荷，含第七维不上抬条件。实体连通、安装接地和强度尚未验收。
