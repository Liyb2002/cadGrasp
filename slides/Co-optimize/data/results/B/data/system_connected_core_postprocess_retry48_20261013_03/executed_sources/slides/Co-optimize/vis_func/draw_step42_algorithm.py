from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,ax=plt.subplots(figsize=(15,17.4),dpi=160)
fig.patch.set_facecolor('#f4f7fb');ax.set_facecolor('#f4f7fb');ax.set_xlim(0,15);ax.set_ylim(1.8,19);ax.axis('off')
def txt(x,y,s,size=14,color='#263649',weight='normal',ha='left'):
 ax.text(x,y,s,fontproperties=font,fontsize=size,color=color,ha=ha,va='top',fontweight=weight,linespacing=1.65)
def box(y,h,title,body,color='#e8f0fa',edge='#88a7c8',x=.7,w=11.45):
 ax.add_patch(FancyBboxPatch((x,y-h),w,h,boxstyle='round,pad=0.15,rounding_size=0.14',facecolor=color,edgecolor=edge,linewidth=1.4))
 txt(x+.3,y-.17,title,18);txt(x+.3,y-.66,body,13)
def arrow(a,b,color='#557390',style='-|>'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=20,lw=2,color=color,connectionstyle='arc3'))
txt(.7,18.65,'Step4.2  ·  Sampling + Gradient 并行求解',25)
txt(.7,18.02,'双流求解｜采样持续探索，短程物理梯度细化候选；共享真实结果池',13,'#9b6022')
box(17.25,1.35,'输入与变量', '从 Step4.1 初始化；固定物体、pose、原始支撑和每个 pose 的 32,768 条载荷。\n变量：每个 pose 的退出方向 di（两个自由度），满足单位长度和自身合法半球。')
arrow((6.4,15.72),(6.4,15.37))
box(15.2,1.65,'① 精确几何与物理：当前支撑到底缺什么？', '全部退出通道共同切除原始支撑；保留每侧 1% 余量及兼容承载接触核。\n从实际接触求反力锥投影，残差 r = Af − b；全部原始载荷参与构造检查。\n用真实投影分离平面扫描全部载荷，补充困难工作集。')
arrow((6.4,13.37),(6.4,13.02))
box(12.85,1.65,'② 物理反馈：恢复哪块接触最有用？', '潜在接触来自原始支撑的实际接触边界；归一化反力生成元为 a。\n新增反力的可达收益：max(0, −a·r)² / 2；同时包含力方向和力矩信息。\n给有用且较易恢复的接触分配价值，冻结一轮局部优化的价值权重。', '#e8f5ef','#82b59e')
ax.plot([6.4,6.4],[11.02,10.9],color='#557390',lw=2)
ax.plot([3.4,9.45],[10.9,10.9],color='#557390',lw=2)
arrow((3.4,10.9),(3.4,10.67));arrow((9.45,10.9),(9.45,10.67))
box(10.5,4.,'③ Sampling 探索流', '共同趋势投影到各合法半球。\n球面覆盖＋高价值接触方向。\n复用反力与分离平面证明。\n少量共同／单方向微调。\n持续探索新的可行区域；\n不等待梯度分支结束。', '#e8f5ef','#82b59e',x=.7,w=5.4)
box(10.5,4.,'④ Gradient 细化流', '从不同、有希望的真实候选启动。\n最多八个分支，每个最多两轮。\n优化 G = Σj wj cj(d)；\n用 SLSQP 联合调整全部方向。\n满足合法半球和步长约束；\n停滞后换起点；末尾有界恢复。', '#fff2db','#d2aa61',x=6.75,w=5.4)
arrow((3.4,6.32),(3.4,6.1));arrow((9.45,6.32),(9.45,6.1))
ax.plot([3.4,9.45],[6.1,6.1],color='#557390',lw=2)
arrow((6.4,6.1),(6.4,5.97))
box(5.8,1.65,'⑤ 共享候选池：精确评估与中间进展', '精确重构候选，重新提取接触、检查全部载荷；局部距离场不作接受证明。\n接受真实承载缺口下降，或同一冻结价值下的非线性遮挡代价下降。\n允许必要的中间承载退步；两条流的真实结果进入候选池，保留较好与不同的状态。', '#fff2db','#d2aa61')
# Feedback loop on right
arrow((12.33,4.9),(13.15,4.9))
ax.plot([13.15,13.15],[4.9,14.35],color='#557390',lw=2)
arrow((13.15,14.35),(12.33,14.35))
txt(13.45,11.3,'反\n复\n迭\n代',16,'#557390')
txt(13.45,7.6,'重\n算\n几\n何\n与\n物\n理',13,'#557390')
arrow((6.4,3.97),(6.4,3.62))
box(3.45,1.35,'最终接受（本轮先不要求连通）', '每个 pose 全部原始载荷通过；完整退出和每侧 1% 余量通过精确几何检查。\n连通性另行记录；支撑接地覆盖、强度与机器人路径不包含在本轮成功定义中。', '#e8f5ef','#82b59e')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
root=Path(__file__).resolve().parents[1]
fig.savefig(root/'step4.2/algorithm.png',dpi=160)
plt.close(fig)
