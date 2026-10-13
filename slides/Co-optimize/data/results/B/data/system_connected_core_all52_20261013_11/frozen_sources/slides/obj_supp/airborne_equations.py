r"""One equation slide: a workpiece supported above the floor.

Same force/torque convention as two_equations.py, with fixture contacts only.
The force cap is the complete original interval, not only its upper endpoint.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).with_name('airborne_equations.png')
INK, MUTED, RULE = '#1b1b1a', '#6b6b66', '#d9ddd8'


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(17, 9.1), dpi=200, facecolor='white')

    def text(y, value, x=.055, fs=22, color=INK, **options):
        return fig.text(x, y, value, ha='left', va='center', fontsize=fs, color=color, **options)

    text(.936, 'Workpiece above the floor', fs=36)
    text(.873, 'The fixture carries the full force–moment demand.', fs=25, color=MUTED)
    text(.796, r'$A_{\rm floor,obj}=\varnothing,\qquad F_{\rm floor,obj}=0$'
         '     The workpiece has no direct floor reaction.', fs=24)
    text(.733, r'$r_{\rm push}=q-c,\quad r_{\rm supp}=p-c,\quad\hat z$ points upward;'
         r' gravity is $-mg\hat z$.', fs=21, color=MUTED)
    text(.677, r'$F_{\rm supp}$ is contact force per area on the workpiece, along its inward normal.',
         fs=21, color=MUTED)
    fig.add_artist(Line2D([.055, .945], [.627, .627], transform=fig.transFigure, color=RULE))

    text(.548, 'force', fs=22, color=MUTED)
    text(.548, r'$\int_{A_{\rm supp,obj}} F_{\rm supp}\,dA'
         r'=mg\hat z-F_{\rm push}=F_D$', x=.17, fs=32)
    text(.421, 'torque', fs=22, color=MUTED)
    text(.421, r'$\int_{A_{\rm supp,obj}} r_{\rm supp}\times F_{\rm supp}\,dA'
         r'=-r_{\rm push}\times F_{\rm push}=\tau_D$', x=.17, fs=30)
    text(.329, 'Both integrals use the same fixture contact field; the fixture may stand on the floor.',
         fs=21, color=MUTED)
    fig.add_artist(Line2D([.055, .945], [.276, .276], transform=fig.transFigure, color=RULE))

    text(.221, r'$0\leq\|F_{\rm push}\|\leq0.5mg$'
         r'     $\angle(F_{\rm push},n_{\rm inward})\leq\alpha,$'
         r'     $\alpha\in\{15^\circ,30^\circ,60^\circ\}$', fs=23)
    text(.161, r'Gravity acts at $c$, so it adds no torque about $c$. With no process force: '
         r'$(F_D,\tau_D)=(mg\hat z,0)$.', fs=21, color=MUTED)
    text(.102, r'The COM demand is unchanged by lifting. Grounded: fixture + floor; above floor: fixture only.',
         fs=20, color=MUTED)
    text(.048, r'At fixed $F_{\rm push}$: $\Delta\tau_O=\Delta c\times F_D$'
         ' when moments are measured about the ground origin.', fs=21, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        if not (fig.bbox.contains(bounds.x0, bounds.y0) and fig.bbox.contains(bounds.x1, bounds.y1)):
            raise RuntimeError(f'Text exceeds canvas: {item.get_text()}')
    fig.savefig(OUT, facecolor='white', transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
