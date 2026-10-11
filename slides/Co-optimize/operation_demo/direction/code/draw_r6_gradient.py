"""Draw the current forward R6 formula and the actual direction derivative.

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


text(.975, "Continuous R6 and the whole-demand direction gradient", 25, True)
text(.951, r"Notation follows direction_math.png: $F,\tau,D,\mathcal{G},\mathcal{C},\mathrm{dir},\eta$.",
     17, color=MUTED)
rule(.932)

text(.911, "1   Express the continuous demand in R6, directly from geometry", 22, True,
     color=BLUE)
text(.883, "For one working triangle f of pose k, measure the force application point from the COM:")
equation(.853, r"r_f=a_f+u\,\mathrm{edge}_{u,f}+v\,\mathrm{edge}_{v,f}", 28)
text(.826, "a is the first vertex relative to COM; the two edges span the triangle; P is the processing force.", 16)
equation(.797, r"u,v\geq 0,\quad u+v\leq 1,\quad \|P\|\leq 0.5,\quad"
         r" n_f^{\mathsf{T}}P\geq\cos\alpha\,\|P\|", 25)
text(.769, "n is the inward work normal; α is the 15°, 30° or 60° cone half-angle; forces use mg units.", 16)
equation(.737, r"\mathcal{R}_k=\bigcup_{f\ \mathrm{working}}"
         r"\left\{(F,\tau)=(-g-P,\,-r_f\times P)\right\}\ \subset\ \mathbb{R}^6", 27)
text(.708, "Use the bounds above and the original object-ray visibility; g is the gravity force.", 16)
text(.683, "This is a five-parameter domain in R6, with the original work-area / angle / magnitude measure.", 16)
equation(.652, r"z=(1,\,P^{\mathsf{T}},\,(uP)^{\mathsf{T}},\,(vP)^{\mathsf{T}})^{\mathsf{T}},"
         r"\qquad (F,\tau)=B_fz", 27)
text(.625, "z is an exact rearrangement; B is built from geometry and gravity, without fitting load samples.", 16)
text(.602, "Rigid translation and exit direction leave this COM demand unchanged; airborne uses the same R6.",
     16, color=MUTED)
rule(.583)

text(.558, "2   Differentiate the ALL-demand gap through changing contacts", 22, True,
     color=BLUE)
equation(.523, r"D_k(F,\tau;\mathrm{dir})=\operatorname{dist}^2"
         r"\!\left((S_k(F,\tau),0),\mathcal{C}_k(\mathcal{G}_k(\mathrm{dir}))\right)", 26)
equation(.481, r"D_{\mathrm{all}}=\frac{1}{2K}\sum_{k=1}^K\frac{1}{s_k^2}\int D_k\,d\mu_k"
         r"\ \simeq\ \widehat D_{\mathrm{all}}="
         r"\frac{1}{2K}\sum_{k=1}^K\frac{1}{s_k^2}\sum_\ell\omega_{k\ell}D_k", 26)
text(.449, "K, S, s and ω retain their original meanings. Positive quadrature includes covered AND uncovered regions.", 16)
text(.427, "Nodes, weights and scales stay fixed across each local block and its competing Juxtapose branches.", 16)

text(.398, "Reuse the projection calculation inside each valid contact basis:", 17, True)
equation(.369, r"Q_I=I-A_IA_I^+,\qquad"
         r"\widetilde B_fz=(S_k(F,\tau),0),\qquad D_k=\|Q_I\widetilde B_fz\|^2", 25)
text(.34, "A contains the active reaction rays; Q is the residual projector. Keep the seventh equilibrium equation.", 16)
text(.318, "Reuse coefficients / residuals if basis contacts survive and all-ray dual conditions hold; otherwise update NNLS.", 15.5)

equation(.284, r"\mathrm{dir}_j\ \longrightarrow\ \{\mathcal{G}_k\}_{k=1}^K"
         r"\ \longrightarrow\ \{\mathcal{C}_k\}_{k=1}^K\ \longrightarrow\ \widehat D_{\mathrm{all}}", 26)
text(.254, "Release only after every blocker clears; new locks also add loss. Evaluate every affected pose.", 16)
equation(.223, r"\mathrm{dir}_j^\pm=\operatorname{legal}"
         r"\!\left(\cos h\,\mathrm{dir}_j\ \pm\ \sin h\,E_{j,a}\right),\quad a=1,2", 26)
equation(.183, r"g_{j,a}\simeq\frac{\widehat D_{\mathrm{all}}(\mathrm{dir}^{j,+})"
         r"-\widehat D_{\mathrm{all}}(\mathrm{dir}^{j,-})}{2h}", 29)
text(.153, "E has two unit tangent columns. This contact-event secant includes both gains and losses across all poses.", 16)
equation(.122, r"\mathrm{dir}_{j,\mathrm{new}}=\operatorname{legal}\!\left("
         r"\cos(\eta\|g_j\|)\,\mathrm{dir}_j"
         r"-\sin(\eta\|g_j\|)\,\frac{E_jg_j}{\|g_j\|}\right)", 27)
text(.092, r"Joint steps use $\eta=\mathrm{trial\ angle}/\sqrt{\sum_i\|g_i\|^2}$; test several angles and retain net progress.", 16)
rule(.071)
text(.052, "R6 is analytic in load parameters; direction / XYZ gradients still resolve hard contact events numerically.", 15, color=MUTED)
text(.029, "PASS uses every original 32,768 demand per pose on search contacts; export reuses that result.", 15, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in checks:
    box = artist.get_window_extent(renderer)
    if box.x0 < .01*fig.bbox.width or box.x1 > .99*fig.bbox.width or box.y0 < 0 or box.y1 > fig.bbox.height:
        raise RuntimeError("Text outside image: " + artist.get_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", metadata={
    "Title": "Forward continuous R6 and whole-demand direction gradient",
    "Description": "Exact geometry-based force/torque domain, valid projection-cell reuse, and all-pose contact-event secants.",
})
plt.close(fig)
print(OUT)
