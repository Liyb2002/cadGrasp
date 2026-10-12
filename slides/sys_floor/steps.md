# 身体生长步骤图：B / pose1+3

[查看四步图](steps.png) · [绘图代码](steps.py)

按用户最新要求恢复原四步图：保留接触头 → 向最近合法地面长身体 → 扩展局部脚面 → 短桥连接。

所有身体（含本步新增材料）统一为灰白色 `#dce2e2`，接触头及附近区域保留原色；浅灰物体为 24% 不透明度的 B / pose1 网格。四步使用同一支撑坐标系、相机与两套地面撒点。

每个 pose 的原始 32,768 个地面需求全部参与绘图：橙色对应 pose1，蓝色对应 pose3。地面为灰色，虚线仅为凸包标记。

## 来源与状态

这是原方案的历史步骤图复绘，读取归档实体与保存的初始身体缓存，不重新设计 geometry。通过归档报告内的 SHA256 定位各身体和最终 OBJ，并检查阶段包含关系、最终连通与体积；最终体积为 135.849081 cm³。所有 baseline 输入只读。

原图使用六块物理头、五个 ID，存在已记录的共享头身份问题；恢复图片不表示该问题已修复或新算法验收通过。构造算法的共享头检查继续保留。

- [图像元数据](steps_data/metadata.json)：历史图状态、源文件哈希、阶段体积、配色与图像哈希。
- [共享头诊断图](steps_data/registration_diagnostic/steps.png)及其[说明](steps_data/registration_diagnostic/README.md)单独保存，不再覆盖四步图。

## 重新绘制

在仓库根目录执行：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/private/tmp/cadgrasp-slides-mpl \
  /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/sys_floor/steps.py
```

主图输出到 `steps.png`，四个单独场景为 `steps_data/step_01.png` 至 `step_04.png`。添加 `--diagnostic` 时仅向 `steps_data/registration_diagnostic/` 写入诊断图。绘图使用 CPU 三角形光栅化，不依赖浏览器。
