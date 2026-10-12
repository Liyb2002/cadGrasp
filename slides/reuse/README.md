# B：同一件支撑，Pose 1 和 Pose 3

使用当前 [B/pose1+3 Step5](../baseline_algo/output/B/pose1+3/step5/) 的实际支撑与两个任务姿态，替换旧的三姿态四肋支架。

- [两个 pose 的图片](reuse_overview.png)
- [单机器人换姿视频](reuse_workflow.mp4)
- [离线交互查看器](index.html)
- [支撑 OBJ](fixture.obj)（米，与源 `shape.obj` 字节一致）

图片与视频使用同一件 **135.849081 cm³** 支撑。彩色保留原接触头及 4 mm 邻域，其余身体为灰白色；这些颜色固定在支撑自身坐标中。存量模型仍是五个 ID、六块物理接触面，其中橙色有两块不同的接触面。

## 展示与动画

图片左右为原始 Pose 1、Pose 3 的装配关系。视频沿用一台 KUKA LBR Med 14 R820：从停车位拿起物体、装入 Pose 1 → 水平退出并放回停车位 → 单独抓取支撑、抬起并转到 Pose 3 → 再拿起物体并水平装入。物体与支撑分别搬运，停车姿态不是额外工作任务。

视频中对每个完整装配绕竖直轴转向，让保存的退出方向朝同一侧；物体相对支撑的变换与落地高度保持原值。退出距离按新实体和物体在该方向的包围范围计算。抓取点来自新实体，不再使用旧支架的后框位置；停车姿态从物体的稳定落地姿态中选择。

无画面文字、无投射阴影，桌面、相机和缩放全程固定，2× 展示速度。原有机器人资产与双指夹爪保留。`robot_check.json` 记录末端跟踪误差、关节运动与连杆离地间隙；这是运动学展示，不代表完整碰撞或抓持验证。`render_data/` 保存关键帧，便于检查动作。

## 数据来源

`code/build.py` 只读取保存的 Step3 接触、Step5 模型与配准，不重新构造支撑或求解载荷。`manifest.json` 记录源文件哈希、两任务标识和源报告的验收状态。当前源 Step5 的最终审计关闭；旧三姿态的 `load_check.json` 和专用检查脚本已移除，旧结果不沿用到新模型。

`data.js` 保存新模型、头部颜色区域、原始任务变换、视频摆放变换和重新生成的机器人轨迹。`render_check.json` 记录本次导出来源、固定镜头和 MP4 参数。

## 重新生成

在仓库根目录依次运行：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/private/tmp/cadgrasp-slides-mpl \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/reuse/code/build.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/reuse/code/robot.py
node slides/reuse/code/export.cjs
```

只更新图片和关键帧可用 `node slides/reuse/code/export.cjs --images-only`。也可直接运行 `code/render_cpu.py`。导出使用本地 C++ 三角形光栅化和 ffmpeg，不依赖浏览器或 GPU；首次运行用 clang++ 编译同目录的 `raster.cpp`。`CADGRASP_PYTHON` 可覆盖 Node 入口使用的 Python 路径。

交互查看器仍使用本地 Three.js，遵循 [MIT license](vendor/THREE-LICENSE.txt)。
