"""Render the component-by-component derivation next to direction_math.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "vis" / "direction_gradient_explained.png"
FONT_DIR = Path("/usr/share/fonts/opentype/noto")
REGULAR = FontProperties(fname=str(FONT_DIR / "NotoSansCJK-Regular.ttc"))
BOLD = FontProperties(fname=str(FONT_DIR / "NotoSansCJK-Bold.ttc"))
INK = "#17212d"
MUTED = "#526171"
ORANGE = "#bd551e"
BLUE = "#1768a4"

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
fig = plt.figure(figsize=(12, 18), dpi=150, facecolor="white")
checks = []


def label(x, y, value, size=16, color=INK, bold=False, align="left"):
    artist = fig.text(
        x, y, value, fontsize=size, color=color,
        fontproperties=BOLD if bold else REGULAR,
        ha=align, va="center",
    )
    checks.append(artist)
    return artist


def equation(y, value, size=24, x=0.5, align="center", color=INK):
    artist = fig.text(
        x, y, "$" + value + "$", fontsize=size, color=color,
        ha=align, va="center",
    )
    checks.append(artist)
    return artist


def card(top, bottom, number, title):
    fig.add_artist(FancyBboxPatch(
        (0.045, bottom), 0.91, top-bottom,
        boxstyle="round,pad=0.008,rounding_size=0.009",
        transform=fig.transFigure,
        facecolor="#f7f9fc", edgecolor="#dce3ea", linewidth=0.9,
        zorder=-1,
    ))
    label(0.068, top-0.023, str(number), size=21, color=BLUE, bold=True)
    label(0.106, top-0.023, title, size=19, bold=True)


def colored_equation(y, pieces, size=25):
    """Lay out adjacent math fragments using their actual rendered widths."""
    artists = [equation(y, value, size, x=0, align="left", color=color)
               for value, color in pieces]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [a.get_window_extent(renderer).width / fig.bbox.width for a in artists]
    left = (1 - sum(widths)) / 2
    centers = []
    for artist, width in zip(artists, widths):
        artist.set_x(left)
        centers.append(left + width / 2)
        left += width
    return centers


label(0.05, 0.97, "这个方向梯度，怎样展开出来？", size=28, bold=True)
label(0.05, 0.94, "先看力这一项：展开三个分量 → 逐项求导 → 合成点积", size=17, color=MUTED)
equation(0.9,
         r"\frac{\partial D_F}{\partial\mathrm{dir}_x}"
         r"=2(F_{\mathrm{near}}-F_{\mathrm{hard}})^{\mathsf{T}}"
         r"\frac{\partial F_{\mathrm{near}}}{\partial\mathrm{dir}_x}", size=26)
equation(0.86,
         r"\theta=\mathrm{dir}_x,\qquad F=F_{\mathrm{near}}(\theta),\qquad H=F_{\mathrm{hard}}",
         size=22)
label(0.5, 0.835, "固定 H 和其余方向分量，只考察 θ 的微小变化。", size=15, color=MUTED, align="center")

card(0.812, 0.716, 1, "把距离平方展开成三个标量项")
equation(0.757,
         r"D_F=\|F-H\|^2=(F_x-H_x)^2+(F_y-H_y)^2+(F_z-H_z)^2",
         size=23)
label(0.5, 0.731, "x、y、z 是力的三个分量；每个分量都可能随 θ 改变。",
      size=15, color=MUTED, align="center")

card(0.695, 0.546, 2, "先对第一项使用标量链式法则")
equation(0.644, r"u(\theta)=F_x(\theta)-H_x,\qquad"
         r"\frac{\partial u}{\partial\theta}=\frac{\partial F_x}{\partial\theta}", size=23)
equation(0.6,
         r"\frac{\partial u^2}{\partial\theta}"
         r"=\frac{\partial u^2}{\partial u}\,\frac{\partial u}{\partial\theta}"
         r"=2u\,\frac{\partial F_x}{\partial\theta}", size=26)
label(0.5, 0.565, "平方这一层给出 2u；里面的反力分量仍要继续对 θ 求导。",
      size=15, color=MUTED, align="center")

card(0.525, 0.357, 3, "对三个分量分别求导，再相加")
equation(0.47,
         r"\frac{\partial D_F}{\partial\theta}"
         r"=2(F_x-H_x)\frac{\partial F_x}{\partial\theta}", size=25)
equation(0.434, r"\qquad\quad+2(F_y-H_y)\frac{\partial F_y}{\partial\theta}", size=25)
equation(0.398, r"\qquad\quad+2(F_z-H_z)\frac{\partial F_z}{\partial\theta}", size=25)
label(0.5, 0.371, "每一项都是：2 × 该分量的误差 × 该分量随方向的变化率。",
      size=15, color=MUTED, align="center")

card(0.336, 0.188, 4, "三个乘积的和，就是两个向量的点积")
equation(0.285,
         r"e=F-H,\qquad v=\frac{\partial F}{\partial\theta},\qquad"
         r"2(e_xv_x+e_yv_y+e_zv_z)=2e^{\mathsf{T}}v", size=23)
centers = colored_equation(0.242, [
    (r"\frac{\partial D_F}{\partial\theta}=2\,", INK),
    (r"(F-H)^{\mathsf{T}}", ORANGE),
    (r"\quad\frac{\partial F}{\partial\theta}", BLUE),
], size=27)
for label_x, center_x, color in [(0.4, centers[1], ORANGE), (0.7, centers[2], BLUE)]:
    fig.add_artist(plt.Line2D(
        [label_x, center_x], [0.223, 0.23],
        transform=fig.transFigure, color=color, linewidth=1,
    ))
label(0.4, 0.213, "当前误差", size=15, color=ORANGE, align="center")
label(0.7, 0.213, "反力变化率", size=15, color=BLUE, align="center")
label(0.5, 0.194, "T 表示转置：三维行向量乘三维列向量，得到一个标量。",
      size=14, color=MUTED, align="center")

card(0.167, 0.051, 5, "力矩按同样步骤求导，两部分相加")
equation(0.113,
         r"\frac{\partial D}{\partial\mathrm{dir}_x}"
         r"=2(F_{\mathrm{near}}-F_{\mathrm{hard}})^{\mathsf{T}}"
         r"\frac{\partial F_{\mathrm{near}}}{\partial\mathrm{dir}_x}", size=24)
equation(0.073,
         r"\qquad\quad+2(\tau_{\mathrm{near}}-\tau_{\mathrm{hard}})^{\mathsf{T}}"
         r"\frac{\partial\tau_{\mathrm{near}}}{\partial\mathrm{dir}_x}", size=24)

label(0.05, 0.028, "至此展开的是外层的平方求导。反力变化率仍需沿下面这条链计算：",
      size=13, color=MUTED)
label(0.05, 0.011, "direction → 退出扫掠 → 接触 geometry → 最近的可实现力／力矩（局部可微时）",
      size=13, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in checks:
    bbox = artist.get_window_extent(renderer)
    if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.width or bbox.y1 > fig.bbox.height:
        raise RuntimeError(f"Text exceeds image bounds: {artist.get_text()}")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor="white", metadata={
    "Title": "Direction gradient: component-by-component derivation",
    "Description": "Expansion of squared force and torque error; fixed hardest demand, local differentiability.",
})
plt.close(fig)
print(OUT)
