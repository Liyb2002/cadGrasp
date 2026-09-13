"""Show centers allocated by true area and their circular candidates."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import circles as P,render as D,diagnostic as G

from step1.cases import pose_name


def sample_contacts(domain,data,report,views,count=5):
    """Pick separated visible examples for illustration, without ranking forces."""
    valid=np.flatnonzero(data.valid)
    if not len(valid):return []
    centers=data.centers_m[valid];normals=domain.mesh.face_normals[data.center_faces[valid]]
    seen=np.zeros(len(valid),bool)
    for basis in views:
        seen|=~domain.mesh.ray.intersects_any(centers+1e-6*normals,np.tile(basis[2],(len(valid),1)))
    pool=valid[seen] if seen.sum()>=min(count,len(valid)) else valid
    chosen=[int(max(pool,key=lambda i:report['patches'][i]['area_m2']))]
    while len(chosen)<min(count,len(pool)):
        remaining=np.array([i for i in pool if i not in chosen],int)
        distances=np.linalg.norm(data.centers_m[remaining,None]-data.centers_m[chosen],axis=2)
        radii=data.radius_m[remaining,None]+data.radius_m[chosen]
        separation=distances-radii
        chosen.append(int(remaining[np.argmax(separation.min(axis=1))]))
    return chosen


def isolated_detail(domain,data,index,basis,width,size):
    """Show an otherwise hidden inner contact on its original source faces."""
    a,b=data.offsets[index:index+2]
    base=domain.mesh.triangles[np.unique(data.source_faces[a:b])]
    patch=data.triangles[a:b]
    triangles=np.concatenate([base,patch])
    colors=np.concatenate([np.tile(D.GREY,(len(base),1)),np.tile(D.ORANGE,(len(patch),1))])
    picture,_=D.raster(triangles,colors,data.centers_m[index],basis,width,size,
                        overlay=np.arange(len(base),len(triangles)))
    ink=ImageDraw.Draw(picture);x=y=size/2;r=5
    assert data.valid[index]
    ink.ellipse((x-r,y-r,x+r,y+r),fill='#267fcb',outline='#fff9e7',width=1)
    return picture


def run(name):
    domain,data,report=P.read(name)
    folder=P.OUTPUTS/name/pose_name()/'step2_local_support'
    views=[D.axes([.6,.55,-.9]),D.axes([-.6,-.65,.9])]
    chosen=sample_contacts(domain,data,report,views)
    valid=np.flatnonzero(data.valid).tolist()
    work=P.W.WorkVolume.read(folder/'work_volume.json') if P.POLICY.ENFORCE_PROCESS_ACCESS else None
    shell,cap,diagnostic=G.centers_page(name,pose_name(),domain,data,report,work,folder)
    for filled in (True,):
        paper=Image.new('RGB',(1900,1550),D.PAPER);ink=ImageDraw.Draw(paper)
        title=f'{name} / {pose_name()} / Step 2 / {len(chosen)} sample contacts' if filled else f'{name} / {pose_name()} / Step 2 / {len(valid)} accepted centers'
        ink.text((35,25),title,font=D.font(40),fill=D.INK)
        subtitle=f'Examples from {len(valid)} accepted contacts   |   All contact orientations allowed   |   Wrap ≤ 90°'
        ink.text((40,94),subtitle,font=D.font(26),fill=D.INK)
        for j,basis in enumerate(views):
            focus,width=D.overall_camera(domain,basis)
            picture,_=D.scene(domain,data,basis,focus,width,890,show_patches=filled,show_hidden=not filled,
                              indices=chosen if filled else valid,ground=j==0)
            if work is not None:
                picture=G.shell_overlay(picture,domain,shell,cap,basis,focus,width,890,j==0)
            paper.paste(picture,(35+940*j,145))
        isolated=[]
        for j,index in enumerate(chosen):
            row=report['patches'][index]
            basis=D.axes(domain.mesh.face_normals[data.center_faces[index]])
            width=max(3.2*data.radius_m[index],domain.mesh.extents.max()*.05)
            picture,shown=D.scene(domain,data,basis,data.centers_m[index],width,330,selected=index,ground=False)
            if not shown:
                picture=isolated_detail(domain,data,index,basis,width,330)
                isolated.append(index)
            x=55+365*j
            ink.text((x+165,1060),row['id'],font=D.font(26),fill=D.INK,anchor='mm')
            angle=f'Wrap: {row["wrap_angle_degrees"]:.1f}°'
            ink.text((x+165,1093),angle,font=D.font(23),fill=D.INK,anchor='mm')
            paper.paste(picture,(x,1110))
            if index in isolated:ink.text((x+165,1460),'Isolated contact surface',font=D.font(21),fill=D.INK,anchor='mm')
        footer='Orange: contact examples. Green: work surface (excluded from contacts). Process-access volume omitted.' if work is None else 'Orange: contact examples. Green: work surface. Purple: finite access-volume preview (not collision geometry).'
        ink.text((40,1514),footer,font=D.font(22),fill=D.INK)
        paper.save(folder/('circles.png' if filled else 'centers.png'))
    P.save(folder/'views.json',dict(complete=True,object=name,pose=pose_name(),circles_sha256=P.sha256(folder/'circles.npz'),
                                   drawing_sha256=P.sha256(__file__),renderer_sha256=P.sha256(D.__file__),
                                   max_wrap_angle_degrees=report['max_wrap_angle_degrees'],
                                   bearing_direction_constraint=report['bearing_direction_constraint'],
                                   views=[dict(basis=b.tolist(),floor_shown=i==0) for i,b in enumerate(views)],
                                   centers_displayed=list(range(report['center_count'])),circles_displayed=chosen,
                                   sample_selection='Five visible, spatially separated accepted contacts for illustration; not a contribution ranking.',
                                   rejected_centers_displayed=True,diagnostic=diagnostic,
                                   diagnostic_code_sha256=P.sha256(G.__file__),
                                   access_preview_code_sha256=P.sha256(G.V.__file__),
                                   process_access_enforced=P.POLICY.ENFORCE_PROCESS_ACCESS,
                                   work_volume_sha256=P.sha256(folder/'work_volume.npz') if work is not None else None,
                                   artifacts={f:P.sha256(folder/f) for f in ['centers.png','circles.png','centers_labeled_1.png','centers_labeled_2.png']},
                                   examples=chosen,isolated_surface_examples=isolated))
    print(name,'wrote centers.png and circles.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or P.OBJECTS:run(name)
