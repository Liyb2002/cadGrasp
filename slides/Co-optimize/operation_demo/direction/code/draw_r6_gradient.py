"""Draw the R6 demand set and the direction_math gradient formulas.

Notation follows direction_math.png. This is documentation rendering only;
it does not load tasks, change geometry, or run the optimization pipeline.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "vis" / "direction_gradient.png"
INK = "#17212d"
BLUE = "#1768a4"
MUTED = "#526171"

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "stix"})
fig = plt.figure(figsize=(14, 22), dpi=170, facecolor="white")
checks = []


def text(y, value, size=17, bold=False, color=INK, x=.045, align="left"):
    artist = fig.text(x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
                      color=color, ha=align, va="center")
    checks.append(artist)
    return artist


def equation(y, value, size=25, color=INK):
    return text(y, "$" + value + "$", size=size, color=color, x=.5, align="center")


def rule(y):
    fig.add_artist(plt.Line2D([.045, .955], [y, y], transform=fig.transFigure,
                             color="#b4bec9", linewidth=.8))


text(.975, "R6 demands and the whole-demand direction gradient", 25, True)
text(.951, "All poses; all original force–torque demands.", 17, color=MUTED)
rule(.932)

text(.905, "1   The R6 demand set", 22, True, color=BLUE)
equation(.863, r"\mathcal{R}_k=\left\{(F,\tau):"
         r"\mathrm{original\ demand\ of\ pose}\ k\right\}\ \subset\ \mathbb{R}^6", 30)
rule(.824)

text(.799, "2   Measure the gap over ALL demands", 22, True,
     color=BLUE)
equation(.758, r"D_k(F,\tau;\mathrm{dir})=\operatorname{dist}^2"
         r"\!\left((S_k(F,\tau),0),\mathcal{C}_k(\mathcal{G}_k(\mathrm{dir}))\right)", 26)
text(.722, "C is the original 7D reaction cone. S scales force–torque units; s fixes each pose's loss scale.", 16)
text(.697, "K counts poses; the seventh coordinate keeps the original no-uplift equation.", 16)
equation(.657, r"D_{\mathrm{all}}(\mathrm{dir})=\frac{1}{2K}\sum_{k=1}^{K}"
         r"\frac{1}{s_k^{2}}\int_{\mathcal{R}_k}"
         r"D_k(F,\tau;\mathrm{dir})\,d\mu_k(F,\tau)", 28)
text(.617, "Covered demands have zero gap; if they become uncovered, their loss enters the same objective.", 16)
equation(.578, r"D_{\mathrm{all}}(\mathrm{dir})\simeq\frac{1}{2K}\sum_{k=1}^{K}"
         r"\frac{1}{s_k^{2}}\sum_{\ell}\omega_{k\ell}"
         r"D_k(F_{k\ell},\tau_{k\ell};\mathrm{dir})", 27)
text(.539, "Positive weights ω and demand points stay fixed throughout each local block and its branches.", 16)
equation(.501, r"\mathrm{dir}_j\ \longrightarrow\ "
         r"\{\mathcal{G}_k(\mathrm{dir})\}_{k=1}^{K}\ \longrightarrow\ "
         r"\{\mathcal{C}_k(\mathcal{G}_k(\mathrm{dir}))\}_{k=1}^{K}"
         r"\ \longrightarrow\ D_{\mathrm{all}}(\mathrm{dir})", 25)
equation(.462, r"\mathcal{G}_k=\{(p_i,n_i):"
         r"\mathrm{contact}\ i\ \mathrm{is\ available\ for\ pose}\ k\}", 24)
text(.430, "Keep material-covered contacts outside every body, working-cone and exit-sweep lock.", 16)
text(.407, "Release counts only when every blocker clears it; newly locked contacts count as losses.", 16)
rule(.382)

text(.357, "3   Differentiate and update the direction", 22, True, color=BLUE)
text(.328, "E has two unit tangent columns: Eᵀ dir = 0 and Eᵀ E = I. Probe each column a:", 16)
equation(.290, r"\mathrm{dir}_j^\pm=\operatorname{legal}"
         r"\!\left(\cos h\,\mathrm{dir}_j\ \pm\ \sin h\,E_{j,a}\right),\quad a=1,2", 26)
text(.255, "Hold the other directions fixed; recompute the whole-set objective at both probes.", 16)
equation(.216, r"g_{j,a}\simeq\frac{D_{\mathrm{all}}(\mathrm{dir}^{j,+})"
         r"-D_{\mathrm{all}}(\mathrm{dir}^{j,-})}{2h},"
         r"\qquad g_j=(g_{j,1},g_{j,2})^{\mathsf{T}}", 29)
equation(.163, r"\mathrm{dir}_{j,\mathrm{new}}=\operatorname{legal}\!\left("
         r"\cos(\eta\|g_j\|)\,\mathrm{dir}_j"
         r"-\sin(\eta\|g_j\|)\,\frac{E_jg_j}{\|g_j\|}\right)", 27)
text(.121, "Try several step sizes; accept a net full-set improvement and keep the directions legal.", 16)
rule(.092)
text(.069, "Finite differences resolve contact events; quadrature evaluates the whole-set objective.", 15, color=MUTED)
text(.046, "PASS uses every original 32,768 demand per pose on search contacts; export reuses that result.", 15, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in checks:
    box = artist.get_window_extent(renderer)
    if box.x0 < .01*fig.bbox.width or box.x1 > .99*fig.bbox.width or box.y0 < 0 or box.y1 > fig.bbox.height:
        raise RuntimeError("Text outside image: " + artist.get_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", metadata={
    "Title": "R6 demand set and whole-demand direction gradient",
    "Description": "Original demand set, all-pose 7D cone distance, fixed quadrature and direction_math spherical finite differences.",
})
plt.close(fig)
print(OUT)
