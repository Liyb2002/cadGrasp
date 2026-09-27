# Step5：局部头部身体与分散脚面

用户已认可当前形态作为后续设计基准。[设计思路与具体构造过程](local_body_design.md) 记录“同一块头部身体在另一姿态成为脚”的含义、本例实现和待做的优化。

静态解释见 [Pose1 六步图解](../output/B/pose1+3/step5/pose1_steps/index.html)／[整组大图](../output/B/pose1+3/step5/pose1_steps/pose1_steps.png)。全组保持 Pose1 摆放；图1、2、4、5、6固定相机，图3俯视同一地面。用青色身体和脚端②解释局部插值，区分真实脚面、需求撒点及虚拟支撑范围，并标出为 Pose3 准备而在当前姿态悬空的部分。

先看 [逐步构造动画](../output/B/pose1+3/step5/construction.html)，可播放、暂停、拖动进度条和逐步切换。已导出 [双姿态同步视频](../output/B/pose1+3/step5/construction.mp4)、[Pose 1 视频](../output/B/pose1+3/step5/construction_pose1.mp4)、[Pose 3 视频](../output/B/pose1+3/step5/construction_pose3.mp4)，每段约 49 秒。20 个步骤依次显示原头、六块初始身体、八处补充脚端、四处短连接及完整实体；左右同步显示同一块新增材料。

当前 **B / pose1+3**：[总览](../output/B/pose1+3/step5/overview.png)、[交互模型](../output/B/pose1+3/step5/index.html)、[仅支架](../output/B/pose1+3/step5/fixture.png)、[同一局部身体的两种摆放](../output/B/pose1+3/step5/local_body.png)、[STL（mm）](../output/B/pose1+3/step5/fixture_mm.stl)。查看器可以选择六块局部身体之一，其余支架隐藏，用同一块真实几何比较两个任务中的位置；这不表示单块可以独立完成任务。

实体取消地面包围框。六块原头附近的身体分别向另一姿态的地面延伸；需要补充跨度的地方，用局部斜向 loft 形成脚端面。各脚面不沿外围连接，身体之间补四处地面上方的短连接。彩色仍仅标记头与附近区域，新增身体为灰白色。

本例体积 **141.426 cm³**，尺寸约 **198 × 196 × 110 mm**。比此前 60.803 cm³ 窄框大；它是局部角色复用的几何实验，不能称作已经省料或完成耦合优化。保留五组颜色／接触标识、**六块实际接触面**，橙色对应两个不同面；原工件任务姿态及各自三头接触保持。

## 怎样检查耦合

连成一件、共用颜色、相交体积大，均不足以说明局部材料复用。当前检查分开回答两个问题：

1. **同一局部身体是否直接连接两种功能面？** [几何见证](../output/B/pose1+3/step5/local_wedge_audit.json) 对六块身体均找到有面积的 head 子三角与另一姿态实际 floor 子三角；两端都在真实边界，整个直接三角楔经 Mesh64 布尔差检查包含于该身体和最终实体。这只证明这些小区域的直接几何关系，不代表整片头或整个身体都有效复用。
2. **这些接触是否对任务有作用？** [端口消融](../output/B/pose1+3/step5/local_role_audit.json) 保持几何不变，分别屏蔽该头、该身体在另一姿态独有的地面反力，再重新分配全部反力。其它身体和连接件仍提供的接触保留。原样本不可行可说明相应端口在当前模型中必要；全部通过则说明冗余，不能据此否定所有复用。整个身体的端口必要，不等于上一步选中的小三角本身必要。

本例六块头的接触均有必要性数值见证；其中 `body0`（pose1 橙头）、`body3`（pose3 橙头）、`body4`（pose3 青色头）的另一姿态独有落脚接触也有必要性见证。其余三块单独屏蔽该落脚接触后仍通过全部原样本，不表示可同时删除这三处。地面归属采用明确的 1 nm 公差，消除布尔差集留下的数值边界碎片，同时保护其它身体及连接件的接触；两姿态未消融的基线均重新通过全部原样本。失败见证由 simplex 与 IPM 同时报不可行，仍是数值证据。

现有无质量刚体模型不描述内部应力。因此这两项合起来仍不是应力复用、刚度、强度或等强度减材证明；较远的脚端延伸也不能仅因属于同一身体就自动算作局部耦合。进一步思考与有限摆放检查见 [设计记录](shared_design.md)。

## 构造中的实际改变

- 头背部约 8 mm 的扩展与另一落地面直接插值，构成原始局部身体；随后用八个脚端区域局部调整形状。拟合参数为本例选定值，保存在 [local_body_case.json](local_body_case.json)，不是全局最优解。
- 所有新增材料同时服从两地面和两完整装卸扫掠；0.4 mm relief 仅作用于新增身体，原头材料恢复后整件复验。
- 一度放在 pose3 `y≈0` 的左脚虽然孤立静力可行，却处于难以连接的窄空间。最终移到 `x=[−58,−50] mm, y=[16,28] mm`，直接接入 pose1 橙头身体。受力使用整件的实际落地面，包括另一姿态的身体自然形成的接触，未要求各 pose 单独配齐独立底座。
- 八个端面都由直接 loft 连接，没有采用绕障地框或格点路线。各局部身体可以重叠；`body0.obj` 至 `body5.obj` 是生成部件，`bridges.obj` 是必要连接，物理结果为它们的单一布尔并集。

## 验收与复现

最终实体封闭、连通，两姿态各 **32,768 / 32,768** 原样本通过；原头材料包含、接触面、两地面及完整连续退出通过。[独立回放](../output/B/pose1+3/step5/independent_audit.json) 从源头接触与实际落地顶点独立重建两刚体矩阵，重放全部 65,536 组反力。沿用无摩擦头、地面有限摩擦见证参数 64、忽略支架自重，不增加载荷或额外零加工力门槛。

[report.json](../output/B/pose1+3/step5/report.json) 是实体验收记录，[design.json](../output/B/pose1+3/step5/design.json) 保存脚端、身体和连接，[equilibrium.npz](../output/B/pose1+3/step5/equilibrium.npz) 保存反力证据。两个附加 audit 的输入哈希对应同一个实体报告；它们独立于主验收，不把科研诊断变成新的载荷验收条件。

```sh
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/build_local_bodies.py
node slides/baseline_algo/step5_connect_support/export_shared_geometry.cjs
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/audit_fixture.py
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/audit_local_roles.py slides/baseline_algo/output/B/pose1+3/step5
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/audit_local_wedges.py slides/baseline_algo/output/B/pose1+3/step5
```

`build_frame_union.py`、`build_coupled_saddle.py`、`build_shared_geometry.py` 保留此前窄框、厚体、杆／专用脚的构造代码，均非当前展示入口。

## 构造动画与视频

`construction_sequence.py` 从当前已验收的输入和 `design.json` 重放实际布尔构造，记录每一步真正新增的材料，并检查各身体及最终并集与已验收实体一致。它不会改写实体验收报告或重新求解受力。动画逐步显现这些材料，解释几何构造顺序；中间帧不表示可用支架、装配运动或制造过程。

`construction_viewer.html` 提供离线播放、单步查看和姿态切换。`render_construction.cjs` 导出三段 1600 × 900、20 fps、H.264 MP4，并检查全部视频解码及浏览器跳转播放。需要本机 Python 几何环境、Node、Playwright、Chrome、`ffmpeg` 和 `ffprobe`；Playwright 与 Chrome 路径可用 `PLAYWRIGHT_CORE_PATH`、`CHROME_PATH` 指定。

```sh
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/construction_sequence.py
node slides/baseline_algo/step5_connect_support/render_construction.cjs
```

可给 Python 入口添加 `--work <缓存目录>` 复用经输入哈希匹配的扫掠缓存；不提供时临时重建。Node 入口添加 `--preview` 仅生成关键帧预览。输出全部直接放在当前 pair 的 `step5/` 下，几何重放记录为 `construction_sequence.json`，视频检查为 `construction_video_check.json`；完整实体查看器会增加“逐步构造 · 视频”入口。

六步静态图从同一份动画几何取实际构造阶段，使用原 32,768 个载荷的地面压力中心，不添加载荷或重做受力求解。`prepare_pose_steps.py` 直接合并各阶段的平面接触集合，避免为绘图重新进行三维布尔运算；最终显示网格逐顶点、逐面匹配已验收实体。`render_pose_steps.cjs` 输出六张独立 PNG、整组图和可放大的阅读页面，与数据及检查记录一起放在 `step5/pose1_steps/` 下，入口为 `index.html`。

```sh
OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/baseline_algo/step5_connect_support/prepare_pose_steps.py
node slides/baseline_algo/step5_connect_support/render_pose_steps.cjs
```

## 历史单 pose 工具与兼容入口

2026-09-17 已按当前要求拆分：

- [Step5：底座](../step5_base/README.md)：从 Step4 需求凸包和 Step3 全部共同方向生成紧凑、可插入的底座。
- [Step6：连接与视频](../step6_connect_support/README.md)：固定该底座和方向，连接原头，验证整件轨迹及承载，生成视频。

本目录保留共享几何、扫掠、反力验证和绘图工具。`belt_assembly.py`、`direction_first.py` 是 Step6 的兼容导入，旧命令入口转向当前 Step6 实现；运行前须先生成 Step5 底座。

原 `output/<object>/<pose>/step5_connect_support/` 文件保留为历史证据，不是新版最终结果。新版输出分别在 `step5_base/` 和 `step6_connect_support/`。全部代码与输出仍使用 Z-up、米制，毫米 STL 明确命名为 `_mm.stl`。
