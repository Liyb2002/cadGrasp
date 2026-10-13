"""English free-body explanation of the connected support's no-uplift condition.

Illustrative, collinear vertical forces; not a baseline load-case rendering.
Writes only the adjacent explanation PNG. No geometry or baseline outputs change.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
INK = '#202936'
MUTED = '#687483'
FRAME = '#73699b'
DOWN = '#397db8'
GROUND = '#25856c'
BAD = '#c65050'


def configure_font():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


def arrow(ax, x, start, end, color, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x, start), (x, end), arrowstyle='-|>', mutation_scale=24,
        linewidth=3.5, color=color, linestyle='--' if dashed else '-', zorder=6))


def panel(fig, left, contact_vertical):
    good = contact_vertical <= 0
    color = GROUND if good else BAD
    ax = fig.add_axes([left, .285, .43, .49])
    ax.set(xlim=(0, 9), ylim=(-1.5, 8.6), aspect='equal')
    ax.axis('off')

    # Abstract free body: no workpiece, individual heads or implied CAD geometry.
    ax.add_patch(Rectangle((2.5, .8), 3, 3.5, facecolor='#eeeaf5',
                           edgecolor=FRAME, linewidth=3, zorder=3))
    ax.text(4, 2.95, 'SUPPORT', fontsize=20, color=FRAME,
            ha='center', va='center', weight='bold')

    ax.fill_between([.9, 8.2], .36, .8, color='#edf0ed', zorder=0)
    ax.plot([.9, 8.2], [.8, .8], color='#929e97', linewidth=2, zorder=4)
    ax.text(7.55, .37, 'Floor', fontsize=12, color=MUTED, ha='center')

    # Only the resultant contact force and the required floor reaction are shown.
    arrow(ax, 4, 6.8 if good else 4.35, 4.35 if good else 6.8, DOWN)
    ax.text(4, 7.65, 'Sum of all head forces', fontsize=18,
            color=DOWN, ha='center', va='center')
    ax.text(4.55, 5.65, '2 N down' if good else '2 N up', fontsize=17,
            color=DOWN, ha='left', va='center')
    reaction = -contact_vertical
    arrow(ax, 4, .8, .8 + reaction * .8, color, dashed=not good)
    if good:
        ax.text(5.8, 1.6, 'Floor reaction\n2 N up', fontsize=15,
                color=color, ha='left', va='center')
    else:
        ax.text(4.45, -.7, 'Would need\n2 N down', fontsize=15,
                color=color, ha='left', va='center')
        ax.text(3.35, -.7, '×', fontsize=30, color=BAD, ha='center')

    center = left + .215
    fig.text(center, .806, 'Vertical balance is possible' if good else 'Vertical balance is impossible',
             ha='center', fontsize=22, color=color, weight='bold')
    fig.text(center, .253,
             'H = −2 N  →  N = +2 N' if good else
             'H = +2 N  →  N = −2 N',
             ha='center', fontsize=18, color=INK)
    fig.text(center, .208, 'The floor can push up.' if good else 'The floor cannot pull down.',
             ha='center', fontsize=22, color=color, weight='bold')


def main():
    configure_font()
    fig = plt.figure(figsize=(16, 10.5), dpi=170, facecolor='white')
    fig.text(.5, .946, 'Support equilibrium requires a nonnegative ground reaction',
             ha='center', fontsize=27, color=INK, weight='bold')
    fig.text(.5, .894, 'Isolate the support. Neglect its weight. The base is neither anchored nor adhesive.',
             ha='center', fontsize=18, color=MUTED)
    fig.add_artist(plt.Line2D([.5, .5], [.20, .825], color='#dde2e7', linewidth=1.2))
    panel(fig, .055, contact_vertical=-2)
    panel(fig, .515, contact_vertical=2)
    fig.text(.5, .126, 'H + N = 0     and     N ≥ 0     imply     H ≤ 0',
             ha='center', fontsize=24, color=INK, weight='bold')
    fig.text(.5, .079, 'Upward is positive. H: head-force total on the support. N: floor reaction. All arrows act on the support.',
             ha='center', fontsize=15, color=MUTED)
    fig.text(.5, .038, 'Illustrative forces; vertical balance only. Full designs also require moment, friction and insertion checks.',
             ha='center', fontsize=14, color=MUTED)
    output = HERE / 'no_uplift_intuition.png'
    fig.savefig(output, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
