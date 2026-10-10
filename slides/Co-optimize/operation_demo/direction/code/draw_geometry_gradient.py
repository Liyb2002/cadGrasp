"""Explain where the inner force sensitivity comes from, with a PNG only."""
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "vis" / "direction_geometry_gradient.png"
BACKUP = HERE / "vis" / "data" / "history" / "direction_geometry_gradient_zh_20261008.png"
REGULAR = FontProperties(family="STIXGeneral")
BOLD = FontProperties(family="STIXGeneral", weight="bold")
INK, MUTED, BLUE, ORANGE = "#17212d", "#526171", "#1768a4", "#bd551e"
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
fig = plt.figure(figsize=(12, 18), dpi=150, facecolor="white")
checks = []


def label(x, y, text, size=16, color=INK, bold=False, align="left"):
    a = fig.text(x, y, text, fontsize=size, color=color,
                 fontproperties=BOLD if bold else REGULAR,
                 ha=align, va="center")
    checks.append(a)
    return a


def eq(y, text, size=24, color=INK, x=0.5, align="center"):
    a = fig.text(x, y, "$" + text + "$", fontsize=size,
                 color=color, ha=align, va="center")
    checks.append(a)
    return a


def card(top, bottom, number, title):
    fig.add_artist(FancyBboxPatch(
        (0.045, bottom), 0.91, top-bottom,
        boxstyle="round,pad=0.008,rounding_size=0.009",
        transform=fig.transFigure, facecolor="#f7f9fc",
        edgecolor="#dce3ea", linewidth=0.9, zorder=-1))
    label(0.068, top-0.023, str(number), size=21, color=BLUE, bold=True)
    label(0.106, top-0.023, title, size=19, bold=True)


label(0.05, 0.97, "From contact geometry to the direction derivative", size=27, bold=True)
label(0.05, 0.94, "Contact geometry → reaction allocation → nearest force → direction derivative", size=17, color=MUTED)
eq(0.9,
   r"\frac{\partial D_F}{\partial\mathrm{dir}_x}"
   r"=2(F_{\mathrm{near}}-F_{\mathrm{hard}})^{\mathsf{T}}"
   r"\frac{\partial F_{\mathrm{near}}}{\partial\mathrm{dir}_x}", size=26)
label(0.5, 0.86, "Follow the last derivative: how the nearest force changes with exit direction.",
      size=16, color=BLUE, align="center")

card(0.833, 0.693, 1, "Exit direction changes the available contacts")
label(0.5, 0.779, "Exit direction → exit swept volume → available contact positions and normals",
      size=18, align="center")
eq(0.745,
   r"\mathcal{G}(\theta)=\{(p_i(\theta),n_i(\theta))\},\qquad f_i(\theta)=-n_i(\theta)",
   size=22)
label(0.5, 0.712, "θ is a direction parameter. A normal on the same original planar face stays fixed.",
      size=15, color=MUTED, align="center")

card(0.672, 0.446, 2, "Find the nearest achievable force–torque pair")
label(0.5, 0.619, "Each contact contributes a force direction and a moment determined by its position:",
      size=16, color=MUTED, align="center")
eq(0.586,
   r"a_i=\left(f_i,\ (p_i-c)\times f_i\right),\qquad A=[a_1\ \cdots\ a_N]", size=23)
eq(0.547,
   r"w_{\mathrm{hard}}=(F_{\mathrm{hard}},\tau_{\mathrm{hard}}),\qquad"
   r"\lambda^*(\theta)=\underset{\lambda\geq0}{\operatorname{arg\,min}}"
   r"\ \|A(\theta)\lambda-w_{\mathrm{hard}}\|^2", size=22)
eq(0.504,
   r"F_{\mathrm{near}}(\theta)=\sum_i\lambda_i^*(\theta)f_i(\theta)", size=26, color=BLUE)
label(0.5, 0.466, r"$\lambda_i$: reaction magnitude at contact i. Positions affect the allocation through moments.",
      size=15, color=MUTED, align="center")

card(0.425, 0.213, 3, "Differentiate the total reaction: the product rule")
inner = eq(0.353,
   r"\frac{\partial F_{\mathrm{near}}}{\partial\theta}"
   r"=\sum_i\left("
   r"\frac{\partial\lambda_i^*}{\partial\theta}f_i"
   r"+\lambda_i^*\frac{\partial f_i}{\partial\theta}"
   r"\right)", size=27, color=BLUE)
inner.set_bbox({"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b3cee4"})
label(0.5, 0.303, "First term: the optimal reaction allocation changes with geometry.",
      size=17, align="center")
label(0.5, 0.272, "Second term: changing contact-force directions changes the total reaction.",
      size=16, align="center")
label(0.5, 0.243, r"Even with fixed $f_i$, changes in contact positions and moment arms can change $\lambda_i$.",
      size=15, color=MUTED, align="center")
label(0.5, 0.22, "This requires geometry sensitivity and the sensitivity of the NNLS optimum.",
      size=15, color=MUTED, align="center")

card(0.192, 0.065, 4, "Substitute this derivative into the distance chain rule")
eq(0.136,
   r"\frac{\partial D_F}{\partial\theta}"
   r"=\frac{\partial D_F}{\partial F_{\mathrm{near}}}"
   r"\frac{\partial F_{\mathrm{near}}}{\partial\theta}"
   r"=2(F_{\mathrm{near}}-F_{\mathrm{hard}})^{\mathsf{T}}"
   r"\frac{\partial F_{\mathrm{near}}}{\partial\theta}", size=24)
label(0.5, 0.091, "Current mismatch × nearest-force sensitivity → distance sensitivity.",
      size=16, color=ORANGE, align="center")

label(0.05, 0.042, "Formal local derivatives assume a differentiable contact structure and optimum; the target is fixed.",
      size=13, color=MUTED)
label(0.05, 0.021, "The demo computes contact-release proxy gradients; the full sensitivity here is a formal relation.",
      size=13, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in checks:
    bbox = artist.get_window_extent(renderer)
    if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.width or bbox.y1 > fig.bbox.height:
        raise RuntimeError(f"Text exceeds image bounds: {artist.get_text()}")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists() and not BACKUP.exists():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, BACKUP)
temporary = OUT.with_name(OUT.stem + ".tmp.png")
fig.savefig(temporary, dpi=150, facecolor="white", metadata={
    "Title": "Geometry to nearest force to exit-direction sensitivity",
    "Description": "Local differentiable dependency through contact geometry and a nonnegative force allocation.",
})
temporary.replace(OUT)
plt.close(fig)
print(OUT)
