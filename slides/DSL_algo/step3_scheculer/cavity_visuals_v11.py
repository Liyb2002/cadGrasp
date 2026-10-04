"""Tight overview and common-fixture overlays; presentation only, no geometry replay."""
from pathlib import Path
import io,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image,ImageDraw,ImageFont
import trimesh
from step3_scheculer import contacts as I
from step3_scheculer.review_operation_dsl import load
from step3_scheculer.run_dsl import saved_task

SOURCE=Path(__file__).resolve()
PALETTE=['#0072B2','#D55E00','#009E73','#CC79A7','#E6AB02']

def bbox(image):
    pixels=np.asarray(image.convert('RGB'))
    yy,xx=np.where(np.any(pixels<240,axis=2))
    return (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)) if len(xx) else (0,0,*image.size)

def axes(cloud):
    fig=plt.figure(figsize=(8,8),dpi=150,facecolor='white')
    ax=fig.add_axes([0,0,1,1],projection='3d',computed_zorder=False)
    center=(cloud.min(0)+cloud.max(0))/2
    extent=np.maximum(np.ptp(cloud,axis=0),np.ptp(cloud,axis=0).max()*.05)
    for axis,c,e in zip(('x','y','z'),center,extent):getattr(ax,'set_'+axis+'lim')((c-e*.54,c+e*.54))
    ax.set_box_aspect(extent,zoom=1.03);ax.set_axis_off();ax.view_init(elev=25,azim=-55)
    ax.set_proj_type('ortho')
    return fig,ax

def solid(ax,triangles,color,alpha=1,zorder=1):
    ax.add_collection3d(Poly3DCollection(triangles,facecolors=color,edgecolors=color,linewidths=0,alpha=alpha,
        shade=True,lightsource=LightSource(315,45),zorder=zorder))

def raster(fig):
    stream=io.BytesIO();fig.savefig(stream,dpi=150,facecolor='white');plt.close(fig);stream.seek(0)
    image=Image.open(stream).convert('RGB');return image.crop(bbox(image))

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def overview(group,tasks,state,mesh,size,old):
    canvas=Image.new('RGB',size,'white');draw=ImageDraw.Draw(canvas);n=len(tasks)
    scales=[]
    for k,(task,b,o) in enumerate(zip(tasks,state.bases,state.offsets)):
        vertices=(mesh.vertices-o)@b.T
        left=round(k*size[0]/n);right=round((k+1)*size[0]/n)
        width=right-left-12;height=size[1]-12
        oldbox=bbox(old.crop((left,0,right,size[1])))
        oldw,oldh=oldbox[2]-oldbox[0],oldbox[3]-oldbox[1]
        best=None;best_score=0
        for elev,azim in ((25,-55),(35,-70),(15,-70),(25,-35),(25,-90)):
            fig,ax=axes(np.vstack([vertices,task.domain.mesh.vertices]));ax.view_init(elev=elev,azim=azim)
            solid(ax,task.domain.mesh.triangles,'#77acd0',.23,1)
            solid(ax,vertices[mesh.faces],'#a3aab0',1,2)
            for j,c in enumerate(state.groups[k]):solid(ax,c['triangles_m'],plt.get_cmap('tab10')(j),1,3)
            trial=raster(fig);scale=min(width/trial.width,height/trial.height)
            score=min(trial.width*scale/oldw,trial.height*scale/oldh)
            if score>best_score:best_score=score;best=(trial,scale)
            if best_score>=2:break
        image,scale=best
        image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
        canvas.paste(image,(left+(right-left-image.width)//2,6+(height-image.height)//2))
        text=task.pose.replace('_',' ').title();draw.text((left+18,12),text,font=font(25),fill='#26343b')
        oldbox=bbox(old.crop((left,0,right,size[1])))
        oldw,oldh=oldbox[2]-oldbox[0],oldbox[3]-oldbox[1]
        scales.append(dict(pose=task.pose,old_content_pixels=[oldw,oldh],new_content_pixels=list(image.size),linear_scale_min=min(image.width/oldw,image.height/oldh)))
    return canvas,scales

def shared(group,tasks,state,mesh):
    objects=[t.domain.mesh.vertices@b+o for t,b,o in zip(tasks,state.bases,state.offsets)]
    span=np.ptp(np.vstack([mesh.vertices]+objects),axis=0).max()
    arrows=[]
    for task,v,path,b in zip(tasks,objects,state.paths,state.bases):
        d=np.asarray(path['initial_object_exit_world'],float)@b;d/=np.linalg.norm(d)
        start=(v.min(0)+v.max(0))/2
        arrows.append((start,d))
    cloud=np.vstack([mesh.vertices]+objects+[s+d*span*.48 for s,d in arrows])
    fig,ax=axes(cloud)
    for k,(task,v) in enumerate(zip(tasks,objects)):solid(ax,v[task.domain.mesh.faces],PALETTE[k],.22,1)
    solid(ax,mesh.triangles,'#7d858e',1,2)
    for k,(start,direction) in enumerate(arrows):
        ax.quiver(*start,*direction,length=span*.48,color=PALETTE[k],linewidth=3.2,arrow_length_ratio=.16,zorder=4)
    scene=raster(fig)
    canvas=Image.new('RGB',(1800,1600),'white');draw=ImageDraw.Draw(canvas)
    draw.text((900,22),group.name+' — Shared fixture space',font=font(36),fill='#26343b',anchor='mt')
    draw.text((900,76),'Alternative placements overlap; arrows show object withdrawal in the fixed fixture frame',font=font(22),fill='#55616a',anchor='mt')
    scale=min(1700/scene.width,1330/scene.height);scene=scene.resize((round(scene.width*scale),round(scene.height*scale)),Image.Resampling.LANCZOS)
    canvas.paste(scene,((1800-scene.width)//2,123+(1330-scene.height)//2))
    labels=[('Single connected support','#7d858e')]+[(t.pose.replace('_',' ').title(),PALETTE[k]) for k,t in enumerate(tasks)]
    widths=[390]+[180]*len(tasks);x=(1800-sum(widths))//2
    for (label,color),width in zip(labels,widths):
        draw.rounded_rectangle((x,1515,x+30,1545),radius=3,fill=color)
        draw.text((x+42,1514),label,font=font(21),fill='#26343b');x+=width
    draw.text((900,1570),'Poses are alternatives, not simultaneous objects.',font=font(19),fill='#66717a',anchor='ms')
    return canvas,[dict(pose=t.pose,start_fixture_m=s.tolist(),unit_exit_fixture=d.tolist()) for t,(s,d) in zip(tasks,arrows)]

def refresh(changed):
    """Refresh only affected dependencies; retain all historic generator hashes."""
    paths=[]
    for group in sorted((I.OUTPUTS/'B').glob('pose*+*')):
        paths.extend([group/'step4/report.json',group/'step4/data/report.json',group/'step4/data/dsl_cavity_support/report.json',group/'step5_evaluate/report.json'])
    paths.extend((I.OUTPUTS/'B/pose1+3/step5_evaluate'/n for n in ('cavity_comparison.json','cavity_final_validation.json')))
    records={p:json.loads(p.read_text()) for p in paths}
    affected=set()
    for _ in range(len(records)+1):
        updates=0
        for path,r in records.items():
            dirty=False
            for name,digest in r.get('provenance',{}).get('code',{}).items():
                target=(I.ROOT/name).resolve()
                if target==SOURCE and I.sha256(target)!=digest:r['provenance']['code'][name]=I.sha256(target);dirty=True
            for name,digest in r.get('artifacts',{}).items():
                target=(path.parent/name).resolve()
                if target in changed and I.sha256(target)!=digest:r['artifacts'][name]=I.sha256(target);dirty=True
            for name,digest in r.get('provenance',{}).get('inputs',{}).items():
                target=(I.ROOT/name).resolve()
                if target in changed and I.sha256(target)!=digest:r['provenance']['inputs'][name]=I.sha256(target);dirty=True
            if dirty:
                r['presentation_only_update']=True
                # Only current presentation reports gain the new source; historical construction proofs retain original code.
                if path.parent.name in ('dsl_cavity_support','step4'):
                    r.setdefault('provenance',{}).setdefault('code',{}).update(I.hashes([SOURCE]))
                I.save(path,r);changed.add(path.resolve());affected.add(path);updates+=1
        if not updates:break
    else:raise RuntimeError('Unexpected cyclic report dependency')
    for path in affected:
        r=records[path]
        if 'provenance' in r and r.get('complete'):I.check_report(path)
        else:
            for name,digest in r.get('artifacts',{}).items():assert I.sha256(path.parent/name)==digest
    return affected

def main():
    groups=sorted((I.OUTPUTS/'B').glob('pose*+*'));changed=set();reports=[]
    protected={}
    for group in groups:
        stage=group/'step3_scheculer/dsl_cavity';r=I.check_report(stage/'report.json')
        tasks=[saved_task(group,p) for p in r['poses']];state,_=load(stage/'final',tasks)
        root=group/'step4';mesh=trimesh.load(root/'shape.obj',force='mesh',process=False)
        protected[root/'shape.obj']=I.sha256(root/'shape.obj');protected[root/'geometry_certificate.npz']=I.sha256(root/'geometry_certificate.npz')
        metric=json.loads((group/'step5_evaluate/report.json').read_text())['metrics']
        old=Image.open(root/'overview.png').convert('RGB');size=old.size
        vis=group/'visualization';vis.mkdir(exist_ok=True)
        backup=vis/'overview_before_tight_framing.png'
        if not backup.exists():shutil.copy2(root/'overview.png',backup)
        old=Image.open(backup).convert('RGB')
        picture,scales=overview(group,tasks,state,mesh,size,old)
        for target in (root/'overview.png',root/'data/dsl_cavity_support/overview.png'):
            picture.save(target);changed.add(target.resolve())
        image,arrows=shared(group,tasks,state,mesh);image.save(vis/'shared_space.png')
        report=dict(complete=True,presentation_only=True,group=group.name,overview_canvas_pixels=list(size),overview_scaling=scales,
            shared_space_canvas_pixels=list(image.size),absolute_exit_arrows=arrows,placement_formula='saved native vertices @ final bases[k] + final offsets[k]',
            geometry_and_metrics_changed=False,exported_geometry_replay=False,
            provenance=dict(inputs=I.hashes([root/'shape.obj',stage/'final/state.json']),code=I.hashes([SOURCE])),
            artifacts={n:I.sha256(vis/n) for n in ('shared_space.png','overview_before_tight_framing.png')})
        I.save(vis/'report.json',report);reports.append((group,metric,scales));print(group.name,[(v['pose'],round(v['linear_scale_min'],2)) for v in scales],flush=True)
    affected=refresh(changed)
    for path,digest in protected.items():assert I.sha256(path)==digest
    for group,metric,scales in reports:
        assert I.check_report(group/'step5_evaluate/report.json')['metrics']==metric
        assert len(list((group/'step4').glob('*.png')))==2
        assert Image.open(group/'step4/overview.png').size==tuple(json.loads((group/'visualization/report.json').read_text())['overview_canvas_pixels'])
        I.check_report(group/'step4/report.json');I.check_report(group/'visualization/report.json')
    print('ALL',len(groups),'groups; unchanged geometry / metrics; affected reports',len(affected),flush=True)
if __name__=='__main__':main()
