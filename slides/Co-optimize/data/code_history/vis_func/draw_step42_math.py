"""Render the mathematical explanation discussed with the user."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(14, 20), dpi=160)
fig.patch.set_facecolor('#f5f7fb')
ax.set(xlim=(0, 14), ylim=(0, 20))
ax.axis('off')

def text(x, y, s, size=14, color='#263649'):
    ax.text(x, y, s, fontsize=size, fontproperties=FONT, va='top', color=color)

def panel(top, title, formulas, note, color='#e8f0fa'):
    height = 2.25
    ax.add_patch(FancyBboxPatch((.65, top-height), 11.8, height,
        boxstyle='round,pad=0.12', facecolor=color, edgecolor='#91a9be'))
    text(.95, top-.15, title, 18)
    for offset, formula in enumerate(formulas):
        ax.text(1.05, top-.72-offset*.55, formula, fontsize=15 if len(formula) > 140 else 19, va='top', color='#172e46')
    text(.95, top-1.86, note, 12)

text(.65, 19.7, 'Step4.2 · 承载缺口如何指导退出方向', 25)
text(.65, 19.12, '讨论的数学框架：平滑接触模型与中间步骤接受规则仍需具体设计；不等同于当前实现。', 12)
panels = [
 (18.4, '① Step4.1 初始化与共同切除',
  [r'$d_i^{(0)}=Q_i^T e_z,\quad \|d_i\|=1,\quad d_i^T n_i\geq0$',
   r'$R(d)=S_0\setminus\bigcup_{i=1}^{n}W_i(d_i)$'],
  'i：pose 编号；Qi：共享坐标到世界坐标的旋转；每个 pose 初始沿自身世界 +z 退出。'),
 (15.8, '② 几何连接：退出方向影响接触',
  [r'$d=(d_1,\ldots,d_n)\ \longmapsto\ a(d)=(a_1,\ldots,a_m)$',
   r'$0\leq a_j(d)\leq1,\qquad f\in\mathcal{C}_i(a)$'],
  'j：候选接触编号；aj 为平滑保留程度；合法接触力集合 Ci(a) 的具体关系需要设计。'),
 (13.2, '③ 每条载荷分别求最小缺口，再聚合',
  [r'$\ell_{ik}(a)=\min_{f\in\mathcal{C}_i(a)}\|D(G_i f-w_{ik})\|_2$',
   r'$L_\tau(a)=\tau\log\sum_{i,k}\exp(\ell_{ik}(a)/\tau)\ \to\ \max_{i,k}\ell_{ik}$'],
  'k：载荷编号；D 统一力与力矩尺度；每条载荷独立分配接触力；τ → 0 时趋近最大缺口。', '#e8f5ef'),
 (10.6, '④ 链式求导：物理价值 × 几何敏感度',
  [r'$g_i=\nabla_{d_i}L_\tau=\sum_j\frac{\partial L_\tau}{\partial a_j}\nabla_{d_i}a_j$',
   r'$v_j=\max\left(0,-\frac{\partial L_\tau}{\partial a_j}\right)$'],
  '对全部 n 个退出方向求导；最难载荷属于 pose 2，也可能需要调整 pose 1 的通道。', '#e8f5ef'),
 (8.0, '⑤ 沿合法方向小幅转动，重新构造真实实体',
  [r'$g_i^{\mathrm{tan}}=(I-d_i d_i^T)g_i$',
   r'$d_i^\prime=\Pi_{H_i}(d_i-\eta g_i^{\mathrm{tan}}),\quad H_i=\{u:\|u\|=1,\ u^T n_i\geq0\}$'],
  '梯度是转动指令；η 为步长。用新方向重新切除、提取真实接触、重新检查原始载荷。', '#fff2db'),
 (5.4, '⑥ 共同遮挡：允许有依据的中间进展',
  [r'$B(d)=\sum_j v_j\sum_i b_{ij}(d_i)\qquad(v_j\ \mathrm{fixed\ per\ step})$',
   r'$L_{\rm real}(d^\prime)<L_{\rm real}(d)\quad\mathrm{or}\quad[L_{\rm real}(d^\prime)\leq L_{\rm real}(d)\ \mathrm{and}\ B(d^\prime)<B(d)]$'],
  'bij 为通道 i 的遮挡程度。第一条通道让开后，接触仍可能被第二条挡着；这是几何进展。', '#fff2db'),
]
for top, title, formulas, note, *colors in panels:
    panel(top, title, formulas, note, colors[0] if colors else '#e8f0fa')
    if top != 5.4:
        ax.add_patch(FancyArrowPatch((6.55, top-2.4), (6.55, top-2.56),
            arrowstyle='-|>', mutation_scale=18, color='#557390', linewidth=2))
ax.plot([12.7, 13.2, 13.2], [4.1, 4.1, 14.7], color='#557390', linewidth=2)
ax.add_patch(FancyArrowPatch((13.2, 14.7), (12.65, 14.7), arrowstyle='-|>', mutation_scale=18, color='#557390'))
text(13.4, 10, '重\n新\n评\n估', 14)
text(.75, 2.65, '共同遮挡示例：两条通道都让开，真实接触才恢复', 16)
ax.text(.9, 2.08,
    r'$(b_{1j},b_{2j}):(1,1)\ \to\ (0,1)\ \to\ (0,0)\qquad a_j:0\ \to\ 0\ \to\ 1$',
    fontsize=20, va='top', color='#172e46')
text(.75, 1.3, '最终接受：全部原始载荷通过 + 完整退出与余量合法 + 单连通正体积支撑。', 15)
text(.75, .8, '承载梯度不自动保证连通；遮挡减少也不等于承载已经通过。', 13)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(ROOT/'step4.2/algorithm_math.png')
fig.savefig(ROOT/'step4.2/algorithm_math.pdf')
plt.close(fig)
