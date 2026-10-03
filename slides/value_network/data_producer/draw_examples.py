"""Two explanatory figures: 20 positions on Pose 1, and its 20 best completions."""
from pathlib import Path
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from prepare import ROOT, I, prepare
from step4_connect_support.clean_render import Renderer, piece, CPU

OUT = ROOT/'slides/value_network/data/B'
SOURCE = OUT/'pose1+3'
INK='#22313b'; ORANGE='#e98b27'; BLUE='#3685b7'


def patch_mesh(triangles):
    vertices=np.asarray(triangles).reshape(-1,3)
    return trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),process=False)


def screen(points,camera):
    focus,basis,span=camera
    p=(np.asarray(points)-focus)@basis.T
    return np.c_[.5+p[:,0]/span,.5+p[:,1]/span]


def scene(renderer,problem,pool,indices,highlight=None,size=800):
    obj=piece(problem.domain.mesh,'#a5b2bb');obj['opacity']=.30
    parts=[CPU.placed(obj)]
    for i in indices:
        mesh=patch_mesh(pool[i]['contact']['triangles_m'])
        patch=piece(mesh,ORANGE if i==highlight else BLUE,.000003)
        parts.append(CPU.placed(patch))
    camera=CPU.fit(problem.domain.mesh.vertices,[-.8,-1,.65],1.,padding=1.24)
    image=renderer.render(parts,camera,size)
    return image,camera




def position_figure(renderer,problem,pool,selected,top):
    fig=plt.figure(figsize=(16,11),dpi=190,facecolor='white')
    fig.text(.5,.945,'Pose 1: the 20 best contact positions',ha='center',fontsize=27,color=INK,weight='bold')
    fig.text(.5,.897,'B / Pose 1 + Pose 3 task  |  start with this head  |  lower value is better',ha='center',fontsize=17,color=INK)
    ax=fig.add_axes([.19,.12,.62,.72]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    image,camera=scene(renderer,problem,pool,[],size=1500)
    ax.imshow(image,extent=(0,1,0,1))
    coords=screen([pool[r['index']]['contact']['center_m'] for r in selected],camera)
    top_ids={r['id'] for r in selected}
    markers=[ORANGE if r['id'] in top_ids else BLUE for r in selected]
    ax.scatter(coords[:,0],coords[:,1],s=105,c=markers,edgecolors='white',linewidths=1.6,zorder=5)
    # Balanced callouts in projection order keep labels from colliding.
    order=np.argsort(coords[:,0]); sides=[order[:10],order[10:]]
    for side,ids in enumerate(sides):
        ids=sorted(ids,key=lambda i:coords[i,1])
        for y,i in zip(np.linspace(.13,.87,len(ids)),ids):
            r=selected[i]; is_top=r['id'] in top_ids
            label=f'#{i+1} C{r["index"]+1:03d}   {r["value"]:.3f}'
            ax.annotate(label,xy=coords[i],xytext=(-.13 if side==0 else 1.13,y),
                ha='right' if side==0 else 'left',va='center',fontsize=15,
                color=ORANGE if is_top else INK,weight='bold' if is_top else 'normal',
                bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='none',alpha=.94),
                arrowprops=dict(arrowstyle='-',color=markers[i],lw=1.05,alpha=.7),annotation_clip=False)
    fig.text(.5,.083,'The 20 lowest measured values on Pose 1; all positions are ranked by value.',ha='center',fontsize=14,color=INK)
    fig.text(.5,.045,r'Value = additional heads needed for BOTH poses + exit-direction dispersion ($\lambda=1$).',ha='center',fontsize=15,color=INK)
    fig.savefig(OUT/'20_positions.png',facecolor='white');plt.close(fig)
    return dict(ids=[r['id'] for r in selected],projection=coords.tolist(),camera=dict(focus=camera[0].tolist(),basis=camera[1].tolist(),span=float(camera[2])))


def solution_figure(renderer,problems,pools,catalogues,top):
    fig=plt.figure(figsize=(25,36),dpi=150,facecolor='white')
    fig.text(.5,.979,'Successful solutions for the 20 best Pose 1 contacts',ha='center',fontsize=27,color=INK,weight='bold')
    fig.text(.5,.96,'Orange = the candidate being scored   |   Blue = heads completing its solution   |   Arrows = legal head exit directions',ha='center',fontsize=17,color=INK)
    left=.025;gap=.012;width=(.95-4*gap)/5
    for rank,row in enumerate(top):
        x=left+(rank%5)*(width+gap)
        offset=(rank//5)*.215
        # Reuse the two-pose card layout, scaled into four rows.
        def yy(y): return .72 + y*.255 - offset
        panel=FancyBboxPatch((x,yy(.105)),width,.774*.255,boxstyle='round,pad=.005,rounding_size=.008',
                            transform=fig.transFigure,facecolor='#f8fafb',edgecolor='#d7dfe4',linewidth=1,zorder=-1)
        fig.add_artist(panel)
        fig.text(x+width/2,yy(.873),f'#{rank+1}  C{row["index"]+1:03d}',ha='center',fontsize=20,color=ORANGE,weight='bold')
        fig.text(x+width/2,yy(.831),f'Value {row["value"]:.3f}',ha='center',fontsize=18,color=INK,weight='bold')
        for k,y in [(0,.548),(1,.273)]:
            group=row['completion_indices'][k]
            highlight=row['index'] if k==0 else None
            ax=fig.add_axes([x+.004,yy(y),width-.008,.25*.255]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
            image,camera=scene(renderer,problems[k],pools[k],group,highlight,size=800)
            ax.imshow(image,extent=(0,1,0,1))
            centers=np.array([pools[k][i]['contact']['center_m'] for i in group])
            xy=screen(centers,camera)
            for label_number,(point,i) in enumerate(zip(xy,group)):
                color=ORANGE if i==highlight else BLUE
                ax.scatter(*point,s=30,c=color,edgecolors='white',linewidths=.6,zorder=4)
                ax.annotate(f'C{i+1:03d}',point,xytext=(6,12 if label_number%2==0 else -15),textcoords='offset points',fontsize=10,color=color,
                            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))
            direction=np.asarray(catalogues[k][row['direction_ids'][k]])
            # This is a task-world direction, projected by the SAME camera as its object.
            center=problems[k].domain.mesh.vertices.mean(axis=0)
            tail=center+np.array([0,0,.14])*problems[k].domain.mesh.extents.max()
            ends=screen(np.array([tail,tail+direction*.40*problems[k].domain.mesh.extents.max()]),camera)
            ax.annotate('',xy=ends[1],xytext=ends[0],arrowprops=dict(arrowstyle='-|>',color='#7558a5',lw=2.6,mutation_scale=15))
            ax.set_title(f'Pose {k*2+1}  /  {len(group)} heads',fontsize=15,color=INK,pad=1)
        fig.text(x+width/2,yy(.246),' + '.join(f'C{i+1:03d}' for i in row['completion_indices'][0]),ha='center',fontsize=11,color=INK)
        fig.text(x+width/2,yy(.221),' + '.join(f'C{i+1:03d}' for i in row['completion_indices'][1]),ha='center',fontsize=11,color=BLUE)
        fig.text(x+width/2,yy(.185),f'{row["total_heads"]} heads total; {row["remaining_heads"]} more after this head',ha='center',fontsize=13,color=INK,weight='bold')
        fig.text(x+width/2,yy(.155),f'Best exit angle: {row["minimum_angle_deg"]:.1f}°',ha='center',fontsize=13,color=INK)
        fig.text(x+width/2,yy(.123),f'{row["remaining_heads"]} + {row["dispersion"]:.3f} = {row["value"]:.3f}',ha='center',fontsize=16,color=INK,weight='bold')
    fig.text(.5,.041,'All 20: 32,768 / 32,768 loads pass in EACH pose, with no uplift; common head directions and connection paths pass.',ha='center',fontsize=17,color='#28734b')
    fig.text(.5,.022,'Shown: actual selected contact surfaces. Lower cost found in 32 attempts; ties use ID order. Final support solid is not constructed here.',ha='center',fontsize=14,color='#657483')
    fig.savefig(OUT/'20_solutions.png',facecolor='white');plt.close(fig)


def main():
    result=json.loads((SOURCE/'values.json').read_text())
    audit=json.loads((SOURCE/'audit.json').read_text());assert audit['passed']
    I.check_hashes(result['config']['source_inputs'])
    problems,pools,catalogues,_=prepare('B',['pose_1','pose_3'],SOURCE/'inputs')
    successful=sorted([r for r in result['rows'] if r['pose']=='pose_1' and r['value'] is not None],key=lambda r:(r['value'],r['id']))
    top=successful[:20];selected=top
    # Only use completions that are explicitly recorded in the full-load replay audit.
    verified={(r['pose'],tuple(r['indices'])) for r in audit['fresh_terminal_replays'] if r['passed']}
    for row in top:
        for k,g in enumerate(row['completion_indices']):assert (result['config']['poses'][k],tuple(g)) in verified
    renderer=Renderer()
    provenance=position_figure(renderer,problems[0],pools[0],selected,top)
    solution_figure(renderer,problems,pools,catalogues,top)
    I.save(SOURCE/'two_figures.json',dict(complete=True,source=I.sha256(SOURCE/'values.json'),
        positions=provenance,top20=[r['id'] for r in top],
        outputs={name:I.sha256(OUT/name) for name in ['20_positions.png','20_solutions.png']},
        code=I.hashes([Path(__file__)]),actual_contact_geometry=True,complete_support_constructed=False))
    print(OUT/'20_positions.png');print(OUT/'20_solutions.png')


if __name__=='__main__':main()
