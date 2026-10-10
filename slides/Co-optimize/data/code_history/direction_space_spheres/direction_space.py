"""Visualize native-floor exit hemispheres in the common object frame.

No swept-solid construction or force solver is run by this visualization.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from optimization.initial_directions import initialize_close_directions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib import cm
import argparse

COLORS=['#2676c7','#e57b20','#27a56d','#b955b7','#ce4c51','#8b773c']

def basis(normal):
    axis=np.eye(3)[np.argmin(abs(normal))]
    u=np.cross(normal,axis);u/=np.linalg.norm(u)
    return u,np.cross(normal,u)

def chart_data(normals,directions,reference):
    u,v=basis(reference);den=directions@reference
    coords=np.c_[directions@u,directions@v]/den[:,None]
    return u,v,coords

def view(ax,reference):
    ax.view_init(elev=np.degrees(np.arcsin(reference[2])),azim=np.degrees(np.arctan2(reference[1],reference[0])))
    for setter in [ax.set_xlim,ax.set_ylim,ax.set_zlim]:setter(-1.12,1.12)
    ax.set_box_aspect([1,1,1]);ax.set_proj_type('ortho');ax.set_axis_off()

def surface():
    phi,theta=np.meshgrid(np.linspace(0,2*np.pi,97),np.linspace(0,np.pi,49))
    return np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)

def draw_object(ax,mesh):
    vertices=(mesh.vertices-mesh.bounds.mean(axis=0))*(.42/mesh.extents.max())
    ax.add_collection3d(Poly3DCollection(vertices[mesh.faces],facecolor='#aeb4ba',edgecolor='none',alpha=.85))

def render(name,group,directions=None,metadata=None):
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    out=root/group['id']/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    states=[state(name,p) for p in group['poses']]
    normals=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for task,T,mesh in states])
    initialized,computed_metadata=initialize_close_directions(normals)
    if directions is None:directions=initialized
    elif isinstance(directions,dict):directions=np.array([directions[p] for p in group['poses']])
    else:directions=np.asarray(directions)
    metadata=computed_metadata if metadata is None else metadata
    reference=np.asarray(metadata['reference_direction']);common=computed_metadata['common_direction_status']!='no_nonzero_common_direction'
    u,v,coords=chart_data(normals,directions,reference)
    n=len(normals);fig=plt.figure(figsize=(4*n,9),facecolor='white');grid=fig.add_gridspec(2,n,height_ratios=[1,1.28])
    phi,theta=np.meshgrid(np.linspace(0,2*np.pi,65),np.linspace(0,np.pi/2,25))
    boundaries=[]
    for i,(pose,normal,direction) in enumerate(zip(group['poses'],normals,directions)):
        e1,e2=basis(normal)
        hemi=np.cos(theta)[...,None]*normal+np.sin(theta)[...,None]*(np.cos(phi)[...,None]*e1+np.sin(phi)[...,None]*e2)
        t=np.linspace(0,2*np.pi,240);edge=np.cos(t)[:,None]*e1+np.sin(t)[:,None]*e2;boundaries.append(edge)
        ax=fig.add_subplot(grid[0,i],projection='3d',computed_zorder=False)
        ax.plot_surface(*hemi.transpose(2,0,1),color=COLORS[i],alpha=.24,linewidth=0,shade=True,rstride=1,cstride=1)
        ax.plot(*edge.T,color=COLORS[i],lw=1.4)
        draw_object(ax,states[0][2]);ax.quiver(0,0,0,*direction,length=1,color=COLORS[i],linewidth=2.5,arrow_length_ratio=.1,zorder=10)
        ax.set_title(pose.replace('_',' ')+': allowed hemisphere',fontsize=12);view(ax,reference)
    split=max(1,n//2);ax=fig.add_subplot(grid[1,:split],projection='3d',computed_zorder=False)
    sphere=surface();counts=(sphere@normals.T>=-1e-10).sum(axis=-1)
    palette=plt.get_cmap('YlGnBu',n+1)(np.arange(n+1));palette[-1]=[.15,.72,.35,1]
    cmap=ListedColormap(palette);norm=BoundaryNorm(np.arange(-.5,n+1.5),n+1)
    ax.plot_surface(*sphere.transpose(2,0,1),facecolors=cmap(norm(counts)),alpha=.72,shade=False,linewidth=0,rstride=1,cstride=1)
    for i,edge in enumerate(boundaries):ax.plot(*edge.T,color=COLORS[i],lw=.8,alpha=.8)
    for i,d in enumerate(directions):ax.scatter(*d,color=COLORS[i],s=30,zorder=20)
    ax.scatter(*reference,color='black',marker='*',s=90,zorder=21);view(ax,reference)
    ax.set_title('Hemisphere overlap: '+('common region exists' if common else 'no common region (LP certificate)'),fontsize=12)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.55,pad=.02,ticks=range(n+1),label='Number of poses allowing this direction')
    zoom=fig.add_subplot(grid[1,split:]);radius=max(np.max(abs(coords))*1.7,np.tan(np.deg2rad(.35)))
    a=np.linspace(-radius,radius,300);X,Y=np.meshgrid(a,a);chart=np.stack([X,Y],axis=-1)
    allowed=normals@reference+X[...,None]*(normals@u)+Y[...,None]*(normals@v)
    common_mask=np.all(allowed>=0,axis=-1)
    angle=lambda q:np.degrees(np.arctan(q))
    zoom.contourf(angle(X),angle(Y),common_mask.astype(float),levels=[.5,1.5],colors=['#5cd989'],alpha=.3)
    for i,normal in enumerate(normals):
        values=allowed[...,i]
        if values.min()<0<values.max():
            zoom.contourf(angle(X),angle(Y),values,levels=[0,max(float(values.max()),1e-12)],colors=[COLORS[i]],alpha=.11)
            zoom.contour(angle(X),angle(Y),values,levels=[0],colors=[COLORS[i]],linewidths=1.6)
        elif values.min()>=0:zoom.text(.02,.98-.05*i,group['poses'][i].replace('_',' ')+': entire window allowed',transform=zoom.transAxes,color=COLORS[i],va='top',fontsize=9)
        endpoint=angle(coords[i]);zoom.plot([0,endpoint[0]],[0,endpoint[1]],color=COLORS[i],lw=1.8)
        zoom.scatter(*endpoint,color=COLORS[i],s=50,label=group['poses'][i].replace('_',' '),zorder=5)
        zoom.annotate(group['poses'][i].replace('pose_',''),endpoint,xytext=(6,5+3*i),textcoords='offset points',color=COLORS[i])
    zoom.scatter(0,0,color='black',marker='*',s=110,label='reference',zorder=6)
    zoom.set(xlim=angle(np.array([-radius,radius])),ylim=angle(np.array([-radius,radius])),xlabel='Local horizontal angle (degrees)',ylabel='Local vertical angle (degrees)',title='Near the reference: nearest legal directions')
    zoom.set_aspect('equal');zoom.grid(alpha=.18);zoom.legend(loc='lower right',fontsize=9)
    pair_angles=np.degrees(np.arccos(np.clip(directions@directions.T,-1,1)))
    fig.suptitle(f"{name} / {group['id']}  |  {'common direction' if common else 'no common direction'}  |  max pair angle {pair_angles.max():.3f} deg",fontsize=15,y=.98)
    fig.text(.5,.018,'Floor constraint only: n_i dot d_i >= 0. All hemispheres use the SAME registered object frame. Green = common region. Black star = reference.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.035,right=.97,bottom=.1,top=.9,wspace=.3,hspace=.25)
    target=out/'direction_space.png';fig.savefig(target,dpi=165);plt.close(fig)
    record=dict(complete=True,presentation_only=True,pose_set=group['id'],object=name,poses=group['poses'],normals_fixture=normals.tolist(),directions_fixture=directions.tolist(),reference_direction=reference.tolist(),initializer=metadata,common_direction_exists=common,common_direction_certificate=group.get('common_direction'),maximum_pair_angle_deg=float(pair_angles.max()),chart_basis=[u.tolist(),v.tolist()],chart_direction_coordinates=coords.tolist(),sphere_grid_is_visualization_only=True,floor_constraint='native up normal dot exit direction >= 0',scope='Native floor hemispheres only; no obstacle-free or force-feasible claim',provenance=provenance([ROOT/'objects'/name/'selected_pose_sets.json']+[ROOT/'objects'/name/'poses'/p/'setup.npz' for p in group['poses']],[Path(__file__),Path(__file__).with_name('direction_space_template.html'),HERE/'helper_func/optimization/initial_directions.py']),artifacts={'../direction_space.png':I.sha256(target)})
    write_interactive(out,record,boundaries,sphere)
    record['artifacts']['../direction_space.html']=I.sha256(out/'direction_space.html')
    save(out/'data/direction_space.json',record)
    return record


def write_interactive(out,record,boundaries,sphere):
    import html
    payload=json.dumps(dict(record=record,boundaries=[b.tolist() for b in boundaries],sphere=sphere.tolist(),colors=COLORS))
    template=Path(__file__).with_name('direction_space_template.html').read_text()
    (out/'direction_space.html').write_text(template.replace('__DATA__',payload).replace('__TITLE__',html.escape(record['object']+'/'+record['pose_set'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='B');p.add_argument('--sets',nargs='+');args=p.parse_args()
    groups=read_selected_pose_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    for g in groups:
        result=render(args.object,g);print('DIRECTION SPACE',g['id'],result['maximum_pair_angle_deg'],flush=True)

if __name__=='__main__':main()
