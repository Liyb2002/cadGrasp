"""Draw the current all-demand direction gradient using the original notation.

Repo-native mathematical illustration; no search, Boolean or video rendering.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'vis/direction_math.png'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix'})
fig = plt.figure(figsize=(13, 18), dpi=150, facecolor='white')
artists = []

def text(y, value, size=16, bold=False):
    artists.append(fig.text(.045, y, value, fontsize=size,
        fontweight='bold' if bold else 'normal', va='center'))

def equation(y, value, size=24):
    artists.append(fig.text(.5, y, '$' + value + '$', fontsize=size,
        ha='center', va='center'))

text(.972, '1   Measure the gap over ALL force–torque demands', 24, True)
text(.94, 'For every pose k and every original reaction demand (F, τ):')
equation(.899, r'D_k(F,\tau;\mathrm{dir})=\operatorname{dist}^{2}'
    r'\!\left((S_k(F,\tau),0),\mathcal{C}_k(\mathcal{G}_k(\mathrm{dir}))\right)')
text(.863, "C is the original 7D reaction cone. S scales force–torque units; s fixes each pose's loss scale.")
text(.837, 'K counts poses; the seventh coordinate keeps the original no-uplift equation.')
equation(.796, r'D_{\mathrm{all}}(\mathrm{dir})=\frac{1}{2K}\sum_{k=1}^{K}'
    r'\frac{1}{s_k^{2}}\int D_k(F,\tau;\mathrm{dir})\,d\mu_k(F,\tau)', 26)
text(.758, 'Covered demands have zero gap; if they become uncovered, their loss enters the same objective.')
equation(.713, r'D_{\mathrm{all}}(\mathrm{dir})\simeq\frac{1}{2K}\sum_{k=1}^{K}'
    r'\frac{1}{s_k^{2}}\sum_{\ell}\omega_{k\ell}'
    r'D_k(F_{k\ell},\tau_{k\ell};\mathrm{dir})', 26)
text(.672, 'Positive weights ω and demand points stay fixed throughout each local block and its branches.')

text(.623, '2   Direction → contact geometry → ALL demand gaps', 24, True)
text(.588, 'Changing one direction can release or lock useful contacts for any pose:')
equation(.546, r'\mathrm{dir}_j\ \longrightarrow\ '
    r'\{\mathcal{G}_k(\mathrm{dir})\}_{k=1}^{K}\ \longrightarrow\ '
    r'\{\mathcal{C}_k(\mathcal{G}_k(\mathrm{dir}))\}_{k=1}^{K}'
    r'\ \longrightarrow\ D_{\mathrm{all}}(\mathrm{dir})', 24)
equation(.504, r'\mathcal{G}_k=\{(p_i,n_i):\mathrm{contact}\ i\ \mathrm{is\ available\ for\ pose}\ k\}', 23)
text(.469, 'Keep material-covered contacts outside every body, working-cone and exit-sweep lock.')
text(.443, 'Release counts only when every blocker clears it; newly locked contacts count as losses.')

text(.393, '3   Compute a numerical gradient and update on the sphere', 23, True)
text(.359, 'E has two unit tangent columns: Eᵀ dir = 0 and Eᵀ E = I. Probe each column a:')
equation(.32, r'\mathrm{dir}_j^{\pm}=\operatorname{legal}'
    r'\!\left(\cos h\,\mathrm{dir}_j\ \pm\ \sin h\,E_{j,a}\right),\qquad a=1,2', 25)
text(.284, 'Hold the other directions fixed; recompute the whole-set objective at both probes.')
equation(.25, r'g_{j,a}\simeq\frac{D_{\mathrm{all}}(\mathrm{dir}^{j,+})'
    r'-D_{\mathrm{all}}(\mathrm{dir}^{j,-})}{2h},\qquad g_j=(g_{j,1},g_{j,2})^{\mathsf{T}}', 27)
equation(.194, r'\mathrm{dir}_{j,\mathrm{new}}=\operatorname{legal}\!\left('
    r'\cos(\eta\|g_j\|)\,\mathrm{dir}_j'
    r'-\sin(\eta\|g_j\|)\,\frac{E_jg_j}{\|g_j\|}\right)', 27)
text(.151, 'Try several step sizes; accept a net full-set improvement and keep the directions legal.')
text(.125, 'Common-direction coordinates can move several blockers together; geometry queries are cached.')
fig.add_artist(plt.Line2D([.045, .955], [.085, .085], transform=fig.transFigure,
                         color='black', linewidth=.8))
text(.063, 'Finite differences resolve contact events; they are not analytic shape derivatives.', 14)
text(.039, 'Quadrature guides search; final acceptance checks all 32,768 loads/pose and the real support mesh.', 14)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in artists:
    box = artist.get_window_extent(renderer)
    if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
        raise RuntimeError('Text outside image: ' + artist.get_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor='white', metadata={
    'Title': 'Whole-demand direction gradient',
    'Description': 'All-pose normalized squared 7D cone distance, fixed quadrature, spherical finite differences.',
})
plt.close(fig)
print(OUT)
