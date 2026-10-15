"""Render black LaTeX-style equations and export matching LaTeX source.

The chain rule is a formal differentiable local-sensitivity description.
Hard contact events are evaluated numerically by the solver; this illustration
asserts no analytic derivative of discrete coverage. No solver is run.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'vis/direction_gradient.png'
TEX = HERE / 'code/direction_gradient.tex'

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix',
                     'text.color': 'black'})
fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='white')

# Shared equations keep the PNG and copyable LaTeX source consistent.
rows = [
    (.85, r'\mathcal{R}_k=\left\{(F,\tau):'
           r'\mathrm{original\ demand\ of\ pose}\ k\right\}\subset\mathbb{R}^{6},'
           r'\qquad k=1,\ldots,N', 28),
    (.665, r'\mathrm{dir}\;\longmapsto\;\mathcal{G}(\mathrm{dir})'
            r'\;\longmapsto\;S(\mathrm{dir})', 36),
    (.56, r'\mathcal{G}:\ \mathrm{exposed\ contact\ geometry},\qquad'
           r'S:\ \mathrm{force\!\!-\!\!torque\ satisfaction}', 21),
    (.40, r'\max_{\mathrm{dir}}\;S(\mathrm{dir}),\qquad'
           r'S(\mathrm{dir})=\frac{1}{N}\sum_{k=1}^{N}'
           r'\operatorname{sat}\!\left(\mathcal{R}_k;\mathcal{G}(\mathrm{dir})\right)', 30),
    (.18, r'\nabla_{\mathrm{dir}}S='
           r'\left(\frac{\partial\mathcal{G}}{\partial\mathrm{dir}}\right)^{\mathsf{T}}'
           r'\nabla_{\mathcal{G}}S', 39),
]
artists = [fig.text(.5, y, '$'+formula+'$', fontsize=size,
                    ha='center', va='center', color='black')
           for y, formula, size in rows]
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in artists:
    box = artist.get_window_extent(renderer)
    if box.x0 < .01*fig.bbox.width or box.x1 > .99*fig.bbox.width or box.y0 < 0 or box.y1 > fig.bbox.height:
        raise RuntimeError('Equation outside image: '+artist.get_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor='white', metadata={
    'Title': 'Direction objective and chain rule',
    'Description': 'Black mathematical notation: original R6 demand sets, exposed contact geometry, mean satisfaction and local chain rule. No analytic discrete-coverage derivative is asserted.',
})
plt.close(fig)

preamble = r'''\documentclass[12pt]{article}
\usepackage[paperwidth=320mm,paperheight=180mm,margin=18mm]{geometry}
\usepackage{amsmath,amssymb}
\pagestyle{empty}
% Formal local sensitivity; hard contact events are evaluated numerically.
\begin{document}
\Large
\begin{gather*}
'''
TEX.write_text(preamble+'\n'+('\n'+r'\\[1.5em]'+'\n').join(
    formula for _, formula, _ in rows)+'\n'+r'''\end{gather*}
\end{document}
''')
print(OUT)
print(TEX)
