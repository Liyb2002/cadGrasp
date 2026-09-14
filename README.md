# cadGrasp

给定工件、工作区域与目标姿态，搜索一件可打印的刚性支撑，使工件在加工载荷下保持静止，并使完整支撑能够沿共同直线装入。当前计算与图像统一为 Z-up，地面为 `z=0`。

- [问题、模型约定与图目](slides/README.md)
- [当前 baseline 算法：Step1–5、停止规则与现有结果](slides/baseline_algo/baseline_algo.md)

当前 baseline 先采样载荷、生成接触头，再最多贪心选择 3 个头并逐个优化尺寸；随后计算地面需求，构造一个连接全部头的框架与底座，验证整件轨迹和共享承载。所有头属于同一个支撑刚体。实际工作面必须避开，加工射线禁区目前关闭。

代码位于 `slides/` 和 `codes/`，物体数据位于 `objects/`。Baseline 输出固定放在
`slides/baseline_algo/output/<object>/pose_<number>/<stage>/`。
Git 保存代码、Markdown、模板和固定测试样例；物体数据及生成的图片、视频、JSON/NPZ 输出保留在本地。新检出环境需要准备对应数据才能重跑案例。

```sh
# 重绘当前 slides；不执行 baseline 搜索
python slides/tools/render.py

# 从 Step1 执行单个姿态的 baseline
python slides/baseline_algo/step3_scheculer/run_all.py B --pose pose_2 --from-step 1
```

使用已安装项目依赖的 `cadgrasp` Python 环境。运行完成、审计通过、设计通过是不同状态，具体以算法说明和当前案例报告为准。
