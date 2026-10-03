"""Hard constraints and explicit searched-completion score calculations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name('value.png')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(20, 12), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 20), ylim=(0, 12)); ax.axis('off')
    def text(x,y,s,size=18,color='#20272e',ha='center',**kw):
        return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',**kw)
    def box(x,y,w,h,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.13',
                                  facecolor=color,edgecolor='#cbd3db',linewidth=1.3))
    text(10,11.55,'How the value target is calculated',30,weight='bold')
    text(4,10.85,'FOUR HARD CONSTRAINTS',21,weight='bold')
    text(13.9,10.85,'SEARCH A COMPLETION, THEN MEASURE ITS COST',21,weight='bold')
    rows=[('01  Force + moment balance','Final: every load in every pose passes','#edf2ff'),
          ('02  Whole-support no uplift','Final: checked with the same reactions','#edf2ff'),
          ('03  Connected structure','No common connection path: branch dies','#edf7f1'),
          ('04  Common insertion / exit','No common exit direction: branch dies','#edf7f1')]
    for y,(title,sub,color) in zip([9.2,7.8,6.4,5.0],rows):
        box(.45,y,7.15,1.12,color)
        text(.73,y+.76,title,20,ha='left',weight='bold')
        text(.73,y+.29,sub,16,ha='left',color='#657483')
    text(4.02,4.61,'Hard constraints are not weighted score terms.',16,color='#a34242',weight='bold')
    box(.45,2.4,7.15,1.75,'#fff7e8')
    text(4.02,3.73,'EXIT PATHS ACROSS POSES',19,weight='bold')
    text(4.02,3.21,'Transform all directions into one object frame.',16)
    text(4.02,2.72,'Keep each pose\'s whole set of legal directions.',16,color='#657483')
    text(4.02,1.65,'No forced shared head.\nIntermediate prefixes need not pass mechanics.',16,color='#657483')

    box(8.15,9.12,11.35,1.14,'#f3f5f7')
    text(13.82,9.83,r'Current set $S$  →  add $h$  →  search until successful set $S^*$',20)
    text(13.82,9.39,r'$S^*$ must pass all poses; measure both terms on this SAME completion.',16,color='#657483')

    box(8.15,6.94,11.35,1.83,'#edf2ff')
    text(8.5,8.39,'1. COUNT THE HEADS ADDED AFTER h',19,ha='left',weight='bold')
    text(13.82,7.84,r'$N_{\rm remaining}=|S^*|-|S|-1$',29)
    text(13.82,7.28,'Example: 4 heads now → add h (5) → succeed with 8.  N = 8 − 4 − 1 = 3.',17)

    box(8.15,3.67,11.35,2.9,'#edf7f1')
    text(8.5,6.18,'2. FIND THE MOST ALIGNED LEGAL FINAL EXITS',19,ha='left',weight='bold')
    text(13.82,5.7,r'$D_k(S_k^*)$: legal unit exit directions of pose $k$, in object coordinates.',17)
    text(13.82,4.97,r'$D_{\rm final}=\min_{d_k\in D_k(S_k^*)}\ \frac{1}{\binom{m}{2}}'
         r'\sum_{i<j}\left(\frac{\arccos(d_i^{T}d_j)}{\pi}\right)^2$',28)
    text(13.82,4.27,'Pick ONE legal direction per pose; minimize the average pairwise squared angle.',16)
    text(13.82,3.9,'Two poses: smallest legal angle 0° → D = 0; 90° → 0.25; 180° → 1.',17)

    box(8.15,1.83,11.35,1.45,'#fff7e8')
    text(13.82,2.73,r'$C(S,h;S^*)=N_{\rm remaining}+\lambda D_{\rm final}$',29)
    text(13.82,2.18,'Lower is better. Compare searched completions by this combined cost.',18,weight='bold')

    text(10,.95,'OFFLINE: search and measure C to create labels.     ONLINE: the value network predicts C for each action.',18,weight='bold')
    text(10,.43,'Found head count is not proven minimal. Budget failures are recorded separately. Step4 verifies actual solid geometry and occupied volume.',14,color='#657483')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for item in ax.texts:
        bounds=item.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0,bounds.y0) or not fig.bbox.contains(bounds.x1,bounds.y1):
            raise RuntimeError(f'Text outside canvas: {item.get_text()}')
    fig.savefig(OUT,facecolor='white'); plt.close(fig); print(OUT)


if __name__=='__main__':
    main()
