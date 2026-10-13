r"""Contact-module requirements: mechanics, connectivity, and insertion.

Regenerate: python slides/obj_supp/demand/demand_equation.py
The force/moment pair and no-uplift equations retain their original notation.
Connectivity and insertion refer to the SAME finite-thickness contact module.
This is the target formulation; the baseline Step3 currently checks heads only.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT = Path(__file__).with_name('demand_equation.png')
INK, PAPER, RULE = '#1b1b1a', '#ffffff', '#d9ddd8'


def main():
    plt.rcParams.update({'font.family': ['DejaVu Sans'], 'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(28, 9), dpi=200, facecolor=PAPER)
    def text(x, y, value, fs=22, **kw):
        return fig.text(x, y, value, ha='center', va='center', fontsize=fs, color=INK, **kw)
    def line(xs, ys):
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=RULE, linewidth=1.4))

    text(.5, .942, 'Contact module: mechanics, connectivity and insertion', 32)
    text(.5, .876, 'Contact forces may change with the load. One module and one insertion direction stay fixed.', 20)
    line([.475, .475], [.195, .825])
    line([.715, .715], [.195, .825])

    x = .247
    text(x, .783, '01–02   Joint mechanics', 26, fontweight='bold')
    text(x, .687, r'$\mathrm{demand}(F_{\rm push},\mathrm{pt})=(F_D,\tau_D)\in\mathbb{R}^{6}$', 27)
    text(x, .592, r'$(F_D,\tau_D)=\left(mg\,\hat z-F_{\rm push},\;-r_{\rm push}\times F_{\rm push}\right)$', 25)
    fig.add_artist(Rectangle((.025, .270), .440, .255, transform=fig.transFigure,
                             facecolor='none', edgecolor=RULE, linewidth=1.4))
    text(x, .499, r'For each load, one shared passive reaction field $F_{\rm supp}$:', 18)
    text(x, .412, r'$\left(\sum F_{\rm supp},\;'
         r'\sum r_{\rm supp}\times F_{\rm supp}\right)=(F_D,\tau_D)$', 27)
    text(x, .318, r'$\sum F_{\rm supp}\cdot\hat z\geq0$', 27)
    text(x, .224, 'Balance: all contacts. No uplift: heads only.', 17)

    x = .595
    text(x, .783, '03   Connected structure', 25, fontweight='bold')
    text(x, .687, r'$\exists\,V_{\rm support}\ \mathrm{connected}$', 25)
    text(x, .592, r'$A_{\rm obj}\subseteq\partial V_{\rm support}$', 27)
    text(x, .487, r'$V_{\rm support}\cap\mathrm{Forbidden}=\varnothing$', 23)
    text(x, .390, 'Forbidden: object interior, working areas,', 17)
    text(x, .342, 'and below-floor regions from all task poses.', 17)
    text(x, .276, 'Contains all selected contact heads.', 17)
    text(x, .224, 'A solid connection with finite thickness.', 17)

    x = .854
    text(x, .783, '04   Common insertion', 25, fontweight='bold')
    text(x, .690, r'$\exists\,d_0$ for the complete module.', 21)
    text(x, .594, r'$\mathrm{Sweep}(V_{\rm support},d_0)\cap\mathrm{int}(\mathrm{obj})=\varnothing$', 21)
    text(x, .504, r'$\mathrm{Sweep}(V_{\rm support},d_0)\cap\{z<0\}=\varnothing$', 21)
    text(x, .412, r'$\mathrm{Sweep}(V_{\rm support},d_0)=$', 22)
    text(x, .348, r'$\{x-t d_0:\ x\in V_{\rm support},\ 0\leq t\leq L\}$', 21)
    text(x, .276, 'Initial installation scene; finite stroke L.', 17)
    text(x, .224, 'The SAME module as in condition 03.', 17)

    line([.025, .975], [.174, .174])
    text(.5, .125, r'$r_{\rm push}=\mathrm{pt}-c$; '
         r'$r_{\rm supp}$: COM-to-contact vector; '
         r'$0\leq|F_{\rm push}|\leq0.5\,mg$; z points upward.', 18)
    text(.5, .065, 'One fixed design across tasks; mechanics applies to every task and admissible load.', 19)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise RuntimeError(f'Text exceeds canvas: {item.get_text()}')
    fig.savefig(OUT, facecolor=PAPER, edgecolor=PAPER, transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
