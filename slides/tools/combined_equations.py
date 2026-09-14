r"""One slide: workpiece equilibrium, reference floor torque, and insertion.

The floor row reproduces slides/ref.png using scalar pressure and force
magnitudes. Bold F distinguishes vector forces in the workpiece equations.
This is the reference's vertical-pressure floor model, not the full frictional
equilibrium certificate used by the baseline. Supports are massless.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / 'combined_equations.png'
INK, MUTED, RULE = '#1b1b1a', '#686862', '#d9ddd8'


def main():
    plt.rcParams.update({'font.family': ['DejaVu Sans'],
                         'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(20, 11.25), dpi=200, facecolor='white')

    def text(x, y, value, size=25, color=INK, **kwargs):
        return fig.text(x, y, value, fontsize=size, color=color,
                        ha='left', va='center', **kwargs)

    text(.045, .94, 'Four equations, one support design', 36, fontweight='bold')
    text(.045, .885, 'The geometry and insertion direction stay fixed; contact reactions may change with each process load.',
         20, MUTED)

    for y in (.845, .585, .355, .115):
        fig.add_artist(Line2D([.045, .955], [y, y], transform=fig.transFigure,
                              color=RULE, linewidth=1.3))

    text(.045, .795, '01–02', 25, fontweight='bold')
    text(.145, .795, 'Workpiece equilibrium', 25, fontweight='bold')
    text(.775, .795, 'obj_supp', 19, MUTED)
    text(.105, .713,
         r'$\int_{\rm supp\_obj}\; \mathbf{F}_{\rm supp}\,dA'
         r'=mg\,\hat z-\mathbf{F}_{\rm push}$', 27)
    text(.505, .713,
         r'$\int_{\rm supp\_obj}\; r_{\rm supp}\times\mathbf{F}_{\rm supp}\,dA'
         r'=-r_{\rm push}\times\mathbf{F}_{\rm push}$', 27)
    text(.105, .628, 'One admissible contact pressure field satisfies both equations, including the workpiece–floor contact.',
         18, MUTED)

    text(.045, .535, '03', 30, fontweight='bold')
    text(.105, .535, 'Whole-assembly floor torque', 25, fontweight='bold')
    text(.775, .535, 'sys_floor', 19, MUTED)
    text(.105, .457,
         r'$\int_{\rm sys\_floor}\; F_{\rm supp}\,r_{\rm supp}\times\hat z\,dA'
         r'=-\left(F_{\rm push}\,r_{\rm push}\times d_{\rm push}\right)$', 31)
    text(.105, .391,
         r'$F_{\rm supp}\geq0$: floor normal pressure; '
         r'$\mathbf{F}_{\rm push}=F_{\rm push}d_{\rm push}$; '
         'all moment arms are measured from c.', 20, MUTED)

    text(.045, .305, '04', 30, fontweight='bold')
    text(.105, .305, 'Common insertion trajectory', 25, fontweight='bold')
    text(.775, .305, 'trajectory', 19, MUTED)
    text(.105, .234,
         r'$\mathrm{Sweep}(\mathrm{supp},a)\cap'
         r'\left(\mathrm{int}(\mathrm{obj})\cup\{z<0\}\right)=\varnothing$', 29)
    text(.105, .163,
         r'$\mathrm{Sweep}(\mathrm{supp},a)=\{x-ta:\ x\in\mathrm{supp},\ t\geq0\}$'
         '   — withdraw along '+r'$-a$', 22, MUTED)

    text(.045, .081,
         r'$\hat z$: up; floor: $z=0$; $r_{\rm supp}=p_{\rm contact}-c$; '
         r'$r_{\rm push}=q-c$; $0\leq F_{\rm push}\leq0.5\,mg$.', 18, MUTED)
    text(.045, .040,
         'Massless supports. The reference floor row uses vertical pressure; full frictional equilibrium is checked separately.',
         18, MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise RuntimeError(f'Text exceeds canvas: {item.get_text()}')
    fig.savefig(OUT, facecolor='white', edgecolor='white', transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
