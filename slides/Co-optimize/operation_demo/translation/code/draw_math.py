"""Copy direction's first section exactly; replace only direction with translation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
source=HERE.parent/'direction/vis/direction_math.png'
original=Image.open(source).convert('RGB')
width,height=original.size
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix'})
fig=plt.figure(figsize=(width/128,height/128),dpi=128,facecolor='white')
def text(y,s,size=14,bold=False):
    fig.text(.052,y,s,fontsize=size,fontweight='bold' if bold else 'normal',va='top')
def eq(y,s,size=16):
    fig.text(.5,y,s,fontsize=size,ha='center',va='top')

# Section 1 is copied directly from the source PNG below, without new notation.
text(.821,'2   Translation → contact geometry → force–torque gap',18,True)
text(.794,'A translation maps to contact geometry, which maps to an achievable')
text(.775,'force–torque set, which determines the gap at the hardest demand:')
eq(.746,r'$\mathrm{trans}\ \to\ \mathcal{G}(\mathrm{trans})\ \to\ \mathcal{C}(\mathcal{G}(\mathrm{trans}))\ \to\ D(F_{\mathrm{hard}},\tau_{\mathrm{hard}}).$')
text(.718,'Define G as the available contact regions with positions and normals:')
eq(.693,r'$\mathcal{G}=\{(p_i,n_i)\}.$')
text(.668,'The chain rule gives:')
eq(.647,r'$\frac{\partial D(F_{\mathrm{hard}},\tau_{\mathrm{hard}})}{\partial\mathrm{trans}}=\frac{\partial D(F_{\mathrm{hard}},\tau_{\mathrm{hard}})}{\partial\mathcal{C}}\circ\frac{\partial\mathcal{C}}{\partial\mathcal{G}}\circ\frac{\partial\mathcal{G}}{\partial\mathrm{trans}}.$',17)
text(.608,'A formal chain relation for smooth dependence of the gap on translation.',13)

text(.571,'3   Compute one gradient component and update',18,True)
text(.544,'Let the hardest demand be')
eq(.522,r'$F_{\mathrm{hard}}=(F_x,F_y,F_z),\quad\tau_{\mathrm{hard}}=(\tau_x,\tau_y,\tau_z),\quad\mathrm{trans}=(\mathrm{trans}_x,\mathrm{trans}_y,\mathrm{trans}_z).$',14)
text(.493,'Keep the hardest demand fixed. Define the nearest achievable pair:')
eq(.470,r'$(F_{\mathrm{near}},\tau_{\mathrm{near}})=\underset{(F,\tau)\in\mathcal{C}(\mathcal{G}(\mathrm{trans}))}{\operatorname{arg\,min}}\ \|(F,\tau)-(F_{\mathrm{hard}},\tau_{\mathrm{hard}})\|^2.$',15)
text(.433,'Then')
eq(.413,r'$D(F_{\mathrm{hard}},\tau_{\mathrm{hard}})=\|F_{\mathrm{near}}-F_{\mathrm{hard}}\|^2+\|\tau_{\mathrm{near}}-\tau_{\mathrm{hard}}\|^2.$',15)
text(.382,'The gradient with respect to translation is')
eq(.357,r'$\nabla_{\mathrm{trans}}D(F_{\mathrm{hard}},\tau_{\mathrm{hard}})=\left(\frac{\partial D}{\partial\mathrm{trans}_x},\frac{\partial D}{\partial\mathrm{trans}_y},\frac{\partial D}{\partial\mathrm{trans}_z}\right).$',16)
text(.317,'The x component (y and z follow the same rule)',14,True)
eq(.292,r'$\frac{\partial D(F_{\mathrm{hard}},\tau_{\mathrm{hard}})}{\partial\mathrm{trans}_x}=$',17)
eq(.256,r'$2(F_{\mathrm{near}}-F_{\mathrm{hard}})^{\mathsf{T}}\frac{\partial F_{\mathrm{near}}}{\partial\mathrm{trans}_x}$',17)
eq(.219,r'$+\ 2(\tau_{\mathrm{near}}-\tau_{\mathrm{hard}})^{\mathsf{T}}\frac{\partial\tau_{\mathrm{near}}}{\partial\mathrm{trans}_x}.$',17)
text(.183,'Update the translation within its native floor plane:')
eq(.160,r'$\mathrm{trans}_{\mathrm{new}}=\mathrm{trans}-\eta\nabla_{\mathrm{trans}}D(F_{\mathrm{hard}},\tau_{\mathrm{hard}}).$',16)
text(.126,'Use the floor-tangent components of the gradient; do not normalize.',13)
text(.105,'Backtrack η, recompute all poses and loads, and accept an improving',13)
text(.087,'legal layout that preserves previously feasible demands.',13)
fig.add_artist(plt.Line2D([.052,.95],[.067,.067],transform=fig.transFigure,color='black',lw=.7))
text(.058,'Formal smooth-distance formulas. At contact switches use geometric',11)
text(.043,'guidance, then actual-load acceptance. Use the original scaled metric',11)
text(.028,'and no-uplift constraint throughout. No relocation channel is carved.',11)
out=HERE/'vis/translation_math.png'
fig.savefig(out,dpi=128,facecolor='white');plt.close(fig)
# Exact original first section, including every label and equation.
result=Image.open(out).convert('RGB')
section_bottom=round(height*.171)
result.paste(original.crop((0,0,width,section_bottom)),(0,0))
result.save(out)
print(out)
