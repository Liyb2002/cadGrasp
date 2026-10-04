"""Two English construction images from accepted geometry; no physical replay."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import trimesh
import experiment as E
from step4_connect_support import clean_render as V,publish_compact as P

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def piece(mesh,color,opacity=1):
    p=P.piece(mesh,color);p['opacity']=opacity;return P.CPU.placed(p)
def mesh(path):return trimesh.load(path,force='mesh',process=False)
def title(picture,label,subtitle=''):
    page=Image.new('RGB',(picture.width,picture.height+92),'white');page.paste(picture,(0,82))
    d=ImageDraw.Draw(page);d.text((18,12),label,font=font(22),fill='#253844');d.text((18,46),subtitle,font=font(15),fill='#62727b');return page

def arrow(im,camera,start,end):
    focus,basis,span=camera;xy=(np.array([start,end])-focus)@basis.T
    xy=np.c_[im.width/2+xy[:,0]*im.height/span,im.height/2-xy[:,1]*im.height/span]
    d=ImageDraw.Draw(im);a,b=xy;v=(b-a)/np.linalg.norm(b-a);side=np.array([-v[1],v[0]])
    d.line([tuple(a),tuple(b)],fill='#875bc0',width=5)
    d.polygon([tuple(b),tuple(b-v*15+side*7),tuple(b-v*15-side*7)],fill='#875bc0')
    d.text(tuple(b+[8,-8]),'+Z object exit',font=font(15),fill='#875bc0')

def render(group):
    source=group/'step4/data/growing_support';report=E.I.check_report(source/'report.json')
    if not report.get('passed'):raise ValueError('Only accepted full supports may be shown')
    plan=report['direction_objective'];problems=E.read_problems(group.name,report['poses'])
    body=mesh(source/'shape.obj');roots=mesh(source/'roots.obj');partial=mesh(source/'partial_growth.obj')
    bases=np.array(report['placement']['bases']);offsets=np.array(report['placement']['offsets'])
    renderer=V.Renderer();pictures=[]
    for p,b,o in zip(problems,bases,offsets):
        installed=body.copy();installed.vertices=(body.vertices-o)@b.T
        cloud=np.vstack([installed.vertices,p.domain.mesh.vertices]);camera=P.CPU.fit(cloud,[-.7,-1.,.7],1.,padding=1.25)
        im=renderer.render([piece(installed,'#bfc7c9'),piece(p.domain.mesh,'#8cb6ce',.48)],camera,660)
        start=p.domain.mesh.bounds.mean(0);arrow(im,camera,start,start+[0,0,.07])
        pictures.append(title(im,p.pose.replace('_',' '),'Same support; world +Z withdrawal'))
    camera=P.CPU.fit(body.vertices,[-.7,-1.,.7],1.,padding=1.2)
    pictures.append(title(renderer.render([piece(body,'#bfc7c9')],camera,660),'One connected support',f"Material: {report['volume_cm3']:.1f} cm3"))
    canvas=Image.new('RGB',(660*len(pictures),pictures[0].height),'white')
    for k,p in enumerate(pictures):canvas.paste(p,(k*660,0))
    canvas.save(group/'step4/overview.png')
    task=problems[0];obj=task.domain.mesh.copy();obj.vertices=obj.vertices@bases[0]+offsets[0]
    sweep=E.S.swept_solid(task.domain.mesh,np.array([0.,0.,E.S.SWEEP_LENGTH]))
    if isinstance(sweep,tuple):sweep=sweep[0]
    # Exact continuous sweep, cropped only for illustration.
    sweep_mesh=E.S.unpack(sweep) if not isinstance(sweep,trimesh.Trimesh) else sweep
    sweep_mesh.vertices=sweep_mesh.vertices@bases[0]+offsets[0]
    lo=np.minimum(obj.bounds[0],body.bounds[0])-.005;hi=np.maximum(obj.bounds[1],body.bounds[1])+[.005,.005,.04]
    display=V.collar(sweep_mesh,lo,hi)
    outlines=[]
    from scipy.spatial import ConvexHull
    for p,b,o in zip(problems,bases,offsets):
        xy=E.pressure_centers(p.targets/p.scale,p.domain.com)[0]
        polygon=np.c_[xy[ConvexHull(xy).vertices],np.full(len(ConvexHull(xy).vertices),.0004)]@b+o
        for a,end in zip(polygon,np.roll(polygon,-1,axis=0)):
            outlines.append(piece(trimesh.creation.cylinder(radius=.00035,segment=[a,end],sections=6),'#347dbb'))
    stages=[('1. Contact surfaces','Original contact starts',roots,None),('2. Exit exclusion','Full sweep; display cropped',roots,display),('3. Grow connections','Blue outlines: required ground coverage',partial,None),('4. Accepted support','One pose shown; all poses checked',body,obj)]
    tiles=[];cloud=np.vstack([body.vertices,obj.vertices]);camera=P.CPU.fit(cloud,[-.7,-1.,.7],1.,padding=1.28)
    for label,sub,solid,other in stages:
        parts=[piece(solid,'#e2ab57' if label.startswith('1') else '#bfc7c9')]
        if label.startswith(('3','4')):parts.extend(outlines)
        if other is not None:parts.append(piece(other,'#8cb6ce',.35))
        im=renderer.render(parts,camera,660)
        if label.startswith('4'):arrow(im,camera,obj.bounds.mean(0),obj.bounds.mean(0)+[0,0,.07])
        tiles.append(title(im,label,sub))
    canvas=Image.new('RGB',(2640,tiles[0].height),'white')
    for k,p in enumerate(tiles):canvas.paste(p,(k*660,0))
    canvas.save(group/'step4/construction_steps.png')
    E.save(group/'step4/data/visualization.json',dict(complete=True,geometry_replayed=False,geometry_changed=False,
        prose_language='English',panel4_object_poses=1,provenance=dict(inputs=E.I.hashes([source/'report.json',source/'shape.obj']),code=E.I.hashes([Path(__file__)])),
        artifacts={f'../{n}':E.I.sha256(group/'step4'/n) for n in ['overview.png','construction_steps.png']}))
    print('RENDERED',group.name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+');args=p.parse_args()
    names=args.groups or [n for n,_ in E.specs()]
    for n in names:
        path=E.OUT/n/'comparison.json'
        if path.exists() and E.json.loads(path.read_text()).get('passed'):render(E.OUT/n)
