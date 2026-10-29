# Subdivision 示意图

- `easy_mode.png`：针竖直向下时，最简单的托盘就够了。
  - 针：只有一根。KUKA 在 07 号工位底座上的真实逆解，针竖直向下，落在兔子背上。
  - 粉色：针尖周围 8 mm 的一小块。竖直方向能否够到，按 `data/B/step1_reach/screen.npz` 判断。
  - 蓝色托盘：放在桌上的圆托盘，平底加一圈矮边（底和边都厚 4 mm，边高 14 mm，边离兔子 6 mm），兔子直接坐在底上。这是示意几何，没有经过 Step3–5 认证。
  - 数字记录在 `easy_mode.json`。

- `vertical_reach.png`：针永远竖直向下，兔子可以摆任意姿态，看针能碰到哪些面。
  - 粉色：竖直针能碰到的面，逐面用 Step1 的精确检查（`step1/reach.py` 的 `certify`：针线、针身、针头都不碰物体）。灰色：碰不到。
  - 画出的那根针落在粉色的平面上，取离重心最近的一块。兔子放在桌上。
  - 姿态用 `--euler X Y Z` 设定：在 easy_mode 的直立姿态之后，依次绕世界 x、y、z 轴转，单位为度。直立时能碰到 31.2%，绕 x 轴转 90°（仰面）时 34.6%。
  - 数字记录在 `<输出名>.json`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B subdivide_task/idea/subdivision/code/easy_mode.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B subdivide_task/idea/subdivision/code/vertical_reach.py --euler 0 0 0 --output vertical_reach.png
```
