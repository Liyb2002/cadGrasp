"""Enlarged sectional illustration, not a solved peak-pressure map."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Polygon, Circle

HERE=Path(__file__).resolve().parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(13,9),facecolor='white')
blue='#539ec8';orange='#e1a34d';red='#d52536';ink='#233344'
# 18 mm peg inside 19 mm opening, shown in an illustrative tilted section.
theta=np.arcsin(19/np.hypot(18,21))-np.arctan2(18,21)
R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
corners=np.array([[-9,-10.5],[9,-10.5],[9,10.5],[-9,10.5]])@R.T
for x in (-12.5,9.5):ax.add_patch(Rectangle((x,-14),3,29,fc=orange,ec='#bd8134',lw=2))
ax.add_patch(Rectangle((-12.5,-17),25,3,fc=orange,ec='#bd8134',lw=2))
ax.add_patch(Polygon(corners,fc=blue,ec='#307598',lw=2))
for p in (corners[3],corners[1]):
    ax.add_patch(Circle(p,1.6,fc=red,alpha=.12,zorder=5))
    ax.plot([p[0],p[0]],[p[1]-.55,p[1]+.55],color=red,lw=11,solid_capstyle='round',zorder=6)
# Force arrows are peg forces ON the socket, not computed magnitudes.
p=corners[3];q=corners[1]
ax.annotate('',xy=(p[0]-6,p[1]),xytext=(p[0]-.4,p[1]),arrowprops=dict(arrowstyle='-|>',color=red,lw=3,mutation_scale=23))
ax.annotate('',xy=(q[0]+6,q[1]),xytext=(q[0]+.4,q[1]),arrowprops=dict(arrowstyle='-|>',color=red,lw=3,mutation_scale=23))
ax.text(-23,13,'榫压向插槽的力',color=red,fontsize=16)
ax.text(13,-7,'榫压向插槽的力',color=red,fontsize=16)
ax.annotate('红色：可能集中承压的窄边',xy=p,xytext=(-25,20),fontsize=18,color=red,arrowprops=dict(arrowstyle='->',color=red,lw=2))
ax.annotate('另一侧的边缘接触',xy=q,xytext=(12,-20),fontsize=18,color=red,arrowprops=dict(arrowstyle='->',color=red,lw=2))
ax.text(0,1,'腰带上的榫',ha='center',color='white',fontsize=22,weight='bold')
ax.text(0,-3,'蓝色',ha='center',color='white',fontsize=16)
ax.annotate('橙色：插槽',xy=(11,4),xytext=(17,8),fontsize=19,color='#a36a22',arrowprops=dict(arrowstyle='->',color='#a36a22',lw=2))
ax.text(0,-16,'末端挡块',ha='center',va='center',fontsize=13,color='white')
ax.set(xlim=(-29,30),ylim=(-24,25));ax.set_aspect('equal');ax.axis('off')
fig.suptitle('插口放大：力矩可让压力集中在两侧窄边',fontsize=27,color=ink,y=.96)
fig.text(.5,.14,'同样的力，接触面积越小，局部压强越大。',ha='center',fontsize=21,color=ink)
fig.text(.5,.095,'若接触带宽为 0.1 mm：100 个载荷的压力下界平均约为支撑的 4.4 倍，最差约 3.9 倍。',ha='center',fontsize=15,color=ink)
fig.text(.5,.045,'剖面示意；红色并非已求解的峰值分布。倾斜姿态与接触宽度需另行验证；箭头不表示力的大小。',ha='center',fontsize=12,color='#68727b')
fig.subplots_adjust(left=.02,right=.98,bottom=.19,top=.89)
path=HERE.parent/'socket_pressure_closeup.png';fig.savefig(path,dpi=190);plt.close(fig);print(path)
