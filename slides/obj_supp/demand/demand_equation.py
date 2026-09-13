r"""Paired external demand, no uplift, and a common head-only sweep.

Regenerate: python slides/obj_supp/demand/demand_equation.py
Items 1–3 form one mechanics block: a demand in R6, supplied by one passive
reaction field which also obeys no uplift. Pure
gravity is a hard feasibility check at each greedy round; working-load coverage
may be partial. The separate insertion block uses the allowed finite direction catalogue.
No uplift is necessary, not a full support equilibrium or tipping certificate.
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
    fig = plt.figure(figsize=(20, 8), dpi=200, facecolor=PAPER)
    def text(x, y, value, fs=22, **kw):
        return fig.text(x, y, value, ha='center', va='center', fontsize=fs, color=INK, **kw)
    def line(xs, ys):
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=RULE, linewidth=1.4))

    text(.5, .942, 'Step 3: joint mechanics and common insertion', 30)
    text(.5, .878, 'Contact forces may change with the load. The selected heads and insertion direction stay fixed.', 18)
    line([.703, .703], [.177, .825])
    x = .354
    text(x, .783, '01–03   One joint mechanics problem', 25, fontweight='bold')
    text(x, .687, r'$\mathrm{demand}(F_{\rm push},\mathrm{pt})=(F_D,\tau_D)\in\mathbb{R}^{6}$', 27)
    text(x, .592, r'$(F_D,\tau_D)=\left(mg\,\hat z-F_{\rm push},\;-(\mathrm{pt}-c)\times F_{\rm push}\right)$', 25)

    fig.add_artist(Rectangle((.043, .239), .622, .292, transform=fig.transFigure,
                             facecolor='none', edgecolor=RULE, linewidth=1.4))
    text(x, .499, r'For each covered load, find one shared passive reaction field $F_{\rm supp}$:', 17)
    text(x, .412, r'$\left(\sum_{\rm contacts}F_{\rm supp},\;'
         r'\sum_{\rm contacts}r_{\rm supp}\times F_{\rm supp}\right)=(F_D,\tau_D)$', 27)
    text(x, .301, r'$\sum_{\rm heads}F_{\rm supp}\cdot\hat z\geq0$', 27)
    text(x, .207, 'Contacts = heads + workpiece–floor contact. The no-uplift sum includes heads only.', 15)

    x = .846
    text(x, .783, '04   Common insertion', 25, fontweight='bold')
    text(x, .690, r'One insertion direction $a$ for all heads.', 16)
    text(x, .611, r'$\mathrm{Sweep}(\mathrm{supp},a)\cap\mathrm{int}(\mathrm{obj})=\varnothing$', 19)
    text(x, .533, r'$\mathrm{Sweep}(\mathrm{supp},a)\cap\{z<0\}=\varnothing$', 19)
    text(x, .447, r'$\mathrm{Sweep}(\mathrm{supp},a)=$', 20)
    text(x, .389, r'$\{x-ta:\ x\in\mathrm{supp},\ t\geq0\}$', 19)
    text(x, .315, 'Space swept out when withdrawing along −a.', 14)
    text(x, .263, 'supp: selected heads; obj: workpiece.', 15)
    text(x, .207, 'Search the allowed finite direction catalogue.', 13)

    line([.03, .97], [.174, .174])
    text(.5, .137, 'pt: push location; c: center of mass; '
         r'$r_{\rm supp}$: COM-to-contact vector; '
         r'$0\leq|F_{\rm push}|\leq0.5\,mg$; z points upward.', 16)
    text(.5, .090, 'Every round: gravity-only joint feasibility + a common head direction are required. '
         'Working-load coverage grows greedily; at most 3 heads.', 15)
    text(.5, .043, 'No uplift assumes one massless, unanchored support. Step 5 checks full support equilibrium and the complete assembly trajectory.', 15)
    fig.savefig(OUT, facecolor=PAPER, edgecolor=PAPER, transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
