"""Visualize the independently checked physical model gap and insertion obstruction."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import OUTPUTS,OBJECTS,sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I


def run(name):
    folder=OUTPUTS/name/pose_name()/'step3_scheculer/verification'
    path=folder/'previous_two_contacts_verification.json'
    report=json.loads(path.read_text())
    diagnosis=report['independent_floor_supports']
    first,second=diagnosis['contacts'][:2]
    paper='#ffffff';ink='#22302f';orange='#ed8e2b';blue='#347fb8';red='#b84837'
    fig,axs=plt.subplots(2,2,figsize=(17.5,10),dpi=140)
    fig.patch.set_facecolor(paper)
    for ax in axs.ravel():
        ax.set_facecolor(paper)
    fig.suptitle(f'{name} / Continuous contact coverage does not certify independent supports',
                 x=.04,ha='left',fontsize=21,color=ink)
    ax=axs[0,0]
    ax.set(xlim=(0,10),ylim=(0,10))
    ax.axis('off')
    ax.set_title(f'Second contact {second["id"]}: free body of its support',loc='left',fontsize=15,pad=14)
    ax.add_patch(Rectangle((3.5,2.6),3,2.3,facecolor=orange,alpha=.8))
    ax.plot([1,9],[2.5,2.5],color='#76807d',linewidth=3)
    ax.text(5,3.6,'massless support',ha='center',fontsize=13)
    ax.annotate('',xy=(5,8.5),xytext=(5,4.9),arrowprops=dict(arrowstyle='->',color=red,lw=3))
    ax.text(5.4,7.2,'Workpiece pushes\nthis support UP',fontsize=14,color=red)
    ax.annotate('',xy=(2.7,5.5),xytext=(2.7,2.5),arrowprops=dict(arrowstyle='->',color=blue,lw=3))
    ax.text(.8,6.2,'Floor can only\npush upward',fontsize=13,color=blue)
    ax.text(5,.9,'No downward external force is available.',ha='center',fontsize=14,color=red)
    ax=axs[0,1]
    proof=first['horizontal_insertion']
    xy=np.asarray(proof['projected_normals'])
    hull=ConvexHull(xy)
    polygon=xy[hull.vertices]
    ax.fill(polygon[:,0],polygon[:,1],color=blue,alpha=.15)
    for v in xy:
        ax.plot([0,v[0]],[0,v[1]],color=blue,alpha=.35,linewidth=.7)
    ax.scatter(xy[:,0],xy[:,1],s=12,color=blue)
    ax.scatter([0],[0],s=75,color=red,zorder=5)
    ax.axhline(0,color='#adb6b2',linewidth=.6);ax.axvline(0,color='#adb6b2',linewidth=.6)
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_title(f'First contact {first["id"]}: horizontal normal projections',loc='left',fontsize=15,pad=14)
    ax.set_xlabel('outward normal x');ax.set_ylabel('outward normal y')
    ax.text(.03,.04,'Origin is strictly inside the hull.\nNonpenetrating horizontal velocity: v = 0.',
            transform=ax.transAxes,fontsize=12,color=red,bbox=dict(facecolor=paper,edgecolor='none',alpha=.92))
    ax=axs[1,0]
    ax.axis('off')
    ax.text(0,.93,'The missing per-support equilibrium condition',fontsize=16,color=ink)
    ax.text(.03,.70,r'$R_j=\sum_i\lambda_{ji}n_{ji,z}\ \geq\ 0$',fontsize=26,color=ink)
    ax.text(.03,.49,f'All inward normal y values of {second["id"]}:\n[{second["inward_y_min"]:.4f}, {second["inward_y_max"]:.4f}]',fontsize=15,color=ink)
    ax.text(.03,.29,r'$n_{ji,z}<0,\quad\lambda_{ji}\geq0\quad\Longrightarrow\quad\lambda_{ji}=0$',fontsize=21,color=red)
    ax.text(.03,.09,'Summing the bodies cancels their contact forces.\nWhole-assembly balance does not restore this condition.',fontsize=13,color=ink)
    ax=axs[1,1]
    ax.axis('off')
    ax.text(0,.93,'What was checked',fontsize=16,color=ink)
    lines=[('Previous two fixed contact patches:','continuous load domain VERIFIED'),
           ('Separate, massless, unanchored supports:','original sample 0 is a COUNTEREXAMPLE'),
           ('First patch, straight horizontal insertion:','OBSTRUCTED at perfect contact')]
    for i,(label,value) in enumerate(lines):
        ax.text(.02,.75-.23*i,label,fontsize=13,color=ink)
        ax.text(.02,.67-.23*i,value,fontsize=14,color=blue if i==0 else red)
    fig.text(.04,.035,'Assumption for the free-body test: each support touches only the workpiece and floor; no anchors or connections between supports.',fontsize=12,color='#63706b')
    fig.subplots_adjust(left=.055,right=.98,top=.89,bottom=.10,hspace=.22,wspace=.18)
    fig.savefig(folder/'verification.png',facecolor=paper)
    plt.close(fig)
    I.save(folder/'verification_views.json',dict(object=name,source_sha256=sha256(path),
        drawing_sha256=sha256(__file__),image_sha256=sha256(folder/'verification.png')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        run(name)
