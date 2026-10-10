# Step3.2：整组工作禁区可视化

本步骤只画图。支撑几何、接触和受力诊断都来自新 [Step3.1](../step3.1/README.md)。原 Step3.2 的包裹计算已经并入 Step3.1。

将整组工作面及向外禁区对齐到同一参考 pose，取全部禁区并集。复用 pose_set_search 的深度渲染：唯一 `overview.png` 含 8 个等轴测环绕视角，从左到右、从上到下每次旋转 45°。蓝色是真实支撑，灰色是物体，橙色是全部工作面，半透明琥珀色是工作禁区；没有图内文字。

禁区与同一个显示球相交，形成圆弧外边界。球半径是显示范围，原始禁区仍向外无限延伸；不替代完整工作角度约束，不改变 Step3.1 支撑，不重新求解力／力矩。

输出在 `output/B/<set>/step3/step3.2/`，只有图片 `overview.png`、`README.md` 和 `data/report.json`。报告保存支撑及初始化来源哈希、原禁区设定、显示截断参数和 8 个视角。

在项目根目录运行：

```sh
.venv/bin/python slides/Co-optimize/step3.2/run.py B
.venv/bin/python slides/Co-optimize/step3.2/run.py B --sets pose1+2+4+6
```

需要先有对应的新 Step3.1 输出。[全部 B 图片](../output/B/index.html)。
