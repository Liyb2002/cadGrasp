"""Render the reordered definition from Co-optimize/idea/representation."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parents[1] / 'idea'
GREEN = '#65A779'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'axes.unicode_minus': False})


def draw_definition():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200, facecolor='white')
    fig.text(.075, .90, 'Pose-set representation', fontsize=26, color='#1D2934')
    fig.text(.5, .735, r'$\mathcal{S}=\{\pi_i\}_{i=1}^{n}$', fontsize=43,
             ha='center', va='center', color='#1D2934')
    fig.text(.5, .65, 'One set contains n task poses', fontsize=16,
             ha='center', color='#63717D')
    fig.text(.5, .515,
             r'$\pi_i=\left(\mathcal{A}_i^{W},\;\Theta_i,\;\hat{\mathbf{d}}_i^{W},\;\mathbf{p}_i^{W}\right)$',
             fontsize=32, ha='center', va='center', color='#1D2934')
    labels = [(.20, r'$\mathcal{A}_i^W\subseteq\partial O_i^W$', 'Working surface', GREEN),
              (.40, r'$\Theta_i\in SO(3)$', 'Pose', '#63717D'),
              (.62, r'$\hat{\mathbf{d}}_i^W\in\mathbb{S}^{2}$', 'World exit direction', '#398CA6'),
              (.83, r'$\mathbf{p}_i^W\in\mathbb{R}^{3}$', 'Absolute world position', '#8A6CB4')]
    for x, math, description, color in labels:
        fig.add_artist(plt.Line2D([x-.074, x+.074], [.395, .395], color=color, lw=3))
        fig.text(x, .333, math, fontsize=20, ha='center', color='#1D2934')
        fig.text(x, .265, description, fontsize=14, ha='center', color=color)
    fig.text(.5, .16,
             r'$\mathbf{x}^{W}=\Theta_i\mathbf{x}^{O}+\mathbf{p}_i^{W}'
             r'\qquad |\mathcal{A}_i^W|=\int_{\mathcal{A}_i^W}dA$',
             fontsize=19, ha='center', color='#425466')
    fig.text(.5, .068, 'Every pose has its own absolute transform in the same world frame W.',
             fontsize=15, ha='center', color='#1D2934')
    fig.savefig(HERE / 'definition.png', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    draw_definition()
