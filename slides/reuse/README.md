# B：一件支架，三个工作姿态

2026-09-26 重新设计。使用已有的 **B / pose_2、pose_3、pose_4**，没有随机补姿态。原始任务数据保持不变；展示时将每个任务的工件与支架一起绕竖直轴摆正，使三个姿态的开口都朝同一方向。保留工件相对地面的倾斜、落地高度、工作面标签及工件与支架的装配关系。

- [唯一的 overview 图片](reuse_overview.png)
- [机器人换姿视频](reuse_workflow.mp4)
- [离线交互查看器](index.html)
- [同一个支架实体](fixture.obj)（带顶点索引的高精度 OBJ，单位：米）

## 这次几何的思路

**让同一件开放支架，用不同区域承接三个姿态的载荷。** 后部连接框把四条承力肋和贴合工件的接触分支连接为一个实体；翻转后，原来的侧面或上部肋可以参与落地。四种颜色固定在支架自身坐标中，整个视频里不会换零件或改变形状。

支架朝向首先根据工件相对地面支点的重力倾倒方向选择。接触区域再根据工作区上的力与力矩联合需求筛选，避免只在视觉上“碰到工件”。不是预先指定 AB、AC、CD 三组头，也不再使用旧图的四个独立方形接触帽。

三个工件姿态和各自 320 mm 的水平退出扫掠体，一起从**同一个完整支架**中扣除。因此，各姿态暂时不用的分支也必须给其他姿态让出装入空间。工作面沿各自外法向的通道一并扣除，并扩张 2 mm；随后清理薄片和脱离主体的碎块。这里没有声称完整 30° 工具接近锥均可达。

这是一版按载荷和空间兼容性构造的几何原型，接触附近的裁切、薄壁和圆角还需优化；OBJ 不是直接制造版。尚未优化材料体积、应力或打印方式，不能据此宣称比三个独立支架更省材料。论文贡献与相关工作的比较见[研究笔记](../../codes/research_notes/multipose_rigid_fixture_design.md)。

## 检查范围

[manifest.json](manifest.json) 记录原始数据哈希、实体连通性、封闭性、接触间隙、落地、工件重叠、水平扫掠重叠和工作面法向通道。扫掠布尔运算的极小有符号体积属于数值误差；判据使用绝对值。原始 B 数据未改写。

[load_check.json](load_check.json) 对生成后的实际接触区域另外做静力抽样：每个姿态检查重力，以及 24 个工作面中心各 8 个 30° 锥边界方向、幅值 0.5 倍工件重力的载荷，共 193 例。工件和支架分别满足六维平衡，接触力在两组方程中大小相等、方向相反。

这项检查沿用当前 baseline 的**无摩擦接触头、无质量支架、地面充分摩擦假设（四射线棱锥，系数 64）**，不是实测材料参数。0.03 mm 内的几何间隙按名义接触处理；实际装配公差仍未解决。抽样通过不等于连续需求域证明，也不等于强度、刚度、抓取可靠性或完整机器人无碰撞验证。

## 视频

只用一台机器人：工件水平退出 → 放到旁边的稳定停车姿态并松手 → 单独抓取并翻转支架 → 落地、松手 → 再抓取工件、调整姿态并水平装入。工件与支架不粘接，也不一起搬运。画面不加文字；2× 展示速度；无投射阴影。**桌面、相机位置、相机朝向和缩放全程固定。** 三个任务共用同一工位，支架开口始终沿世界 +Y 方向面向镜头，换姿只绕这个开口轴翻转。Overview 使用一致的支架相对观察方向，以便对照共用区域。

视频镜头在原取景基础上放大 20%（1.2×），固定取景中心略向下调整，保留机械臂底座及完整操作范围。

视频中的三个工作姿态依次对应 **B / pose_2、pose_3、pose_4**。中途的停车姿态只是换装步骤，不计为第四个工作姿态。`data.js` 的 `poses` 保留原始任务旋转；`videoLayout` 记录逐任务的水平转角和实际展示变换。展示变换逐一保持原始的工件—支架相对位姿，不能把展示中的世界旋转直接当作原始世界旋转。

机器人使用项目已有的 KUKA LBR Med 14 R820 资产。视频中调整了暂存位和任务的水平摆放方向，工件与支架在任务终态的相对关系保持一致。支架从后框的横梁／侧梁抓取，抬离地面后绕固定开口轴翻转，每次约 90°。物体的水平退出行程按完整包围盒分离再加 12 mm 间隙确定。每段搬运比较夹爪两个等价朝向和多个冗余关节姿态，按完整路径的关节转动量及手腕弯转选择动作；空手接近和撤离时允许夹爪自由调整水平朝向，减少不必要的回转。[robot_check.json](robot_check.json) 记录运动学误差、转动量和机器人连杆对地间隙；不把这些指标当成碰撞或抓持证书。

## 重新生成

在仓库根目录依次运行：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/reuse/code/build.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/reuse/code/check_loads.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/reuse/code/robot.py
PLAYWRIGHT_CORE_PATH=/tmp/cadgrasp_visual_tools/node_modules/playwright-core \
  node slides/reuse/code/export.cjs
```

图片只导出 `reuse_overview.png`，不再导出单姿态图、支架单独图或 storyboard。渲染关闭投射阴影，保留表面光照以呈现形状。导出逐帧检查开口世界方向恒定、完整入镜、桌面及镜头完全静止，结果写入 `render_check.json`。`--images-only` 跳过视频编码，仅检查关键帧。导出需要 Chrome、ffmpeg、playwright-core；也可用 `CHROME_PATH` 和 `PLAYWRIGHT_CORE_PATH` 指向自己的安装。Three.js 0.160.0 已随页面提供，遵循 [MIT license](vendor/THREE-LICENSE.txt)。

macOS 默认通过 Metal 使用 GPU 导出；设置 `REUSE_SOFTWARE_RENDERING=1` 可切回软件渲染。
