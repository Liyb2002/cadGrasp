"""Render the completed group-generalization experiment as a standalone figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transfer_experiment import ROOT

def main():
    a=json.loads((ROOT/'audit.json').read_text());c=a['comparison']
    labels=['Original\nmemorized tasks','Frozen network\n10 new groups','Five-group model\ntraining groups','Five-group model\n5 held-out groups','Five-group model\noriginal 10 groups']
    rates=[1]+[c[k]['passed']/c[k]['groups'] for k in ['frozen','five_train','five_test','original_ten_test']]
    counts=['10/10']+[f"{c[k]['passed']}/{c[k]['groups']}" for k in ['frozen','five_train','five_test','original_ten_test']]
    quality=[1]+[c[k]['near_best_count']/c[k]['groups'] for k in ['frozen','five_train','five_test','original_ten_test']]
    fig,(ax,table)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.35,1]})
    import numpy as np
    x=np.arange(5)
    bars=ax.bar(x-.18,rates,width=.36,color='#477bae',label='Hard constraints pass')
    ax.bar(x+.18,quality,width=.36,color='#49a38d',label='Cost within 0.05 of best recorded')
    ax.legend(loc='upper right',fontsize=8)
    ax.set_ylim(0,1.13);ax.set_yticks([0,.2,.4,.6,.8,1],['0%','20%','40%','60%','80%','100%']);ax.set_ylabel('Actual Step3 success rate')
    ax.set_xticks(range(5),labels,fontsize=9);ax.set_title('Memory works; transfer is a separate test',loc='left',fontweight='bold')
    for b,t in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.035,t,ha='center',fontweight='bold')
    table.axis('off');rows=[]
    for key,label in [('five_train','TRAIN'),('five_test','TEST')]:
        for r in c[key]['results']:
            rows.append([label,r['group'],'PASS' if r['passed'] else 'FAIL',str(r['heads'])])
    t=table.table(cellText=rows,colLabels=['Split','Pose group','Step3','Heads'],loc='center',cellLoc='left',colWidths=[.15,.48,.2,.17]);t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.65)
    table.set_title('Five groups train; five groups stay held out',loc='left',fontweight='bold')
    fig.text(.04,.025,'Same 15 pose/head identities, new combinations. Random initialization for the five-group model.\nAll 32,768 loads per pose at successful terminals. Cost = heads + exit dispersion. Failed rows show heads attempted, not solutions.',fontsize=10)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(ROOT/'results.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
