"""Step 5 illustrations using the shared, axis-free CAD renderer."""
from step1.needs import COORD
import numpy as np
import json
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from step2_local_support import render as R
from step3_scheculer import contacts as I

BLUE=np.array([64.,119.,165.])
TEAL=np.array([47.,139.,143.])
MUTED='#65706c'
ORANGE_INK='#a65c14'
VIEW=R.axes([.68,.8,-1.])


def fit(points,basis=VIEW,margin=1.13):
    coordinates=np.asarray(points)@basis.T
    low,high=coordinates.min(axis=0),coordinates.max(axis=0)
    return ((low+high)/2)@basis,float(max(high[:2]-low[:2])*margin),basis


def floor_triangles(points,margin):
    low=np.min(points,axis=0);high=np.max(points,axis=0)
    x0,y0=COORD.floor(low)-margin;x1,y1=COORD.floor(high)+margin
    return np.array([[x0,0.,y0],[x1,0.,y0],[x1,0.,y1],[x0,0.,y1]])[[[0,1,2],[0,2,3]]]


def geometry_points(domain,modules,translations=None):
    shifts=translations if translations is not None else [np.zeros(3)]*len(modules)
    return np.vstack([domain.mesh.vertices]+[a['union_vertices_m']+s for (_,a),s in zip(modules,shifts) if s is not None])


def camera(domain,modules,basis=VIEW,points=None):
    points=geometry_points(domain,modules) if points is None else points
    ground=floor_triangles(points,.025*float(domain.mesh.extents.max()))
    return fit(np.vstack([points,ground.reshape(-1,3)]),basis)


def arrays(domain,contacts,modules,translations=None,show_object=True,ground=None):
    pieces=[];colors=[];overlay=[];unlit=[];count=0
    def add(triangles,color,bias=False,flat=False):
        nonlocal count
        if not len(triangles):return
        pieces.append(triangles)
        colors.append(np.tile(color,(len(triangles),1)) if np.asarray(color).ndim==1 else color)
        if bias:overlay.extend(range(count,count+len(triangles)))
        if flat:unlit.extend(range(count,count+len(triangles)))
        count+=len(triangles)
    if ground is not None:add(ground,R.FLOOR,flat=True)
    if show_object:
        palette=np.tile(R.GREY,(len(domain.mesh.faces),1));palette[domain.work_ids]=R.GREEN
        add(domain.mesh.triangles,palette)
    shifts=translations if translations is not None else [np.zeros(3)]*len(modules)
    lookup={c['candidate_id']:c for c in contacts}
    for (entry,data),shift in zip(modules,shifts):
        if shift is None:continue
        add(data['union_vertices_m'][data['union_faces']]+shift,BLUE)
        for index,label in enumerate(data['part_labels']):
            if not str(label).startswith(('ground_strip_','ground_pad_')):continue
            va,vb=data['vertex_offsets'][index:index+2];fa,fb=data['face_offsets'][index:index+2]
            add(data['part_vertices_m'][va:vb][data['part_faces'][fa:fb]]+shift,np.asarray(entry.get('ground_color',TEAL)),bias=True)
        contact=lookup.get(entry['candidate_id'])
        if contact is not None:add(contact['triangles_m']+shift,R.ORANGE,bias=True)
    if not pieces:return np.empty((0,3,3)),np.empty((0,3)),[],[]
    return np.concatenate(pieces),np.concatenate(colors),overlay,unlit


def render(data,view,size):
    triangles,colors,overlay,unlit=data;focus,width,basis=view
    if not len(triangles):return Image.new('RGB',(size,size),R.PAPER),np.full((size,size),-1,np.int32)
    return R.raster(triangles,colors,focus,basis,width,size,overlay=np.asarray(overlay,int),unlit=unlit)


def layer(data,view,size):
    """Retain rendered depth to reuse the static scene in animation frames."""
    picture,ids=render(data,view,size)
    depth=np.full((size,size),-np.inf)
    yy,xx=np.nonzero(ids>=0)
    if len(xx):
        triangles,_,overlay,_=data;focus,width,basis=view
        projected=R.project(triangles,focus,basis,width,size)
        projected[np.asarray(overlay,int),:,2]+=width*1e-5
        a,b,c=np.moveaxis(projected[ids[yy,xx]],1,0)
        det=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        u=((b[:,1]-c[:,1])*(xx+.5-c[:,0])+(c[:,0]-b[:,0])*(yy+.5-c[:,1]))/det
        v=((c[:,1]-a[:,1])*(xx+.5-c[:,0])+(a[:,0]-c[:,0])*(yy+.5-c[:,1]))/det
        depth[yy,xx]=u*a[:,2]+v*b[:,2]+(1-u-v)*c[:,2]
    return np.asarray(picture).copy(),depth


def merge_layers(static,moving):
    pixels=static[0].copy();visible=moving[1]>static[1]
    pixels[visible]=moving[0][visible]
    return Image.fromarray(pixels)


def locators(picture,contacts,view,size):
    ink=ImageDraw.Draw(picture);focus,width,basis=view
    entries=sorted([(R.project(c['center_m'],focus,basis,width,size)[:2],c['candidate_id']) for c in contacts],key=lambda v:v[0][1])
    occupied=[]
    for point,name in entries:
        x,y=point;r=max(4,size/150)
        ink.ellipse((x-r,y-r,x+r,y+r),outline=ORANGE_INK,width=2)
        tx=float(np.clip(x+13,8,size-105));ty=float(np.clip(y-24,10,size-40))
        while any(abs(tx-px)<100 and abs(ty-py)<29 for px,py in occupied):ty+=30
        ty=min(ty,size-34);occupied.append((tx,ty))
        if abs(ty-y)>35:ink.line([(x,y),(tx,ty+12)],fill=ORANGE_INK,width=1)
        ink.text((tx,ty),name,font=R.font(max(18,int(size*.026))),fill=ORANGE_INK)


def contact_overlay(picture,contacts,view,size):
    """Explicit X-ray overlay of the actual interface, including hidden faces.

    The unchanged 3D coordinates are projected normally. Only occlusion is
    removed; callers must label the illustration as an X-ray interface view.
    """
    if not contacts:return picture
    triangles=np.concatenate([c['triangles_m'] for c in contacts])
    colors=np.tile(R.ORANGE,(len(triangles),1))
    overlay,ids=render((triangles,colors,[],range(len(triangles))),view,size)
    picture.paste(overlay,mask=Image.fromarray(np.uint8(ids>=0)*255))
    return picture


def reference_camera(name,contact,round_number,domain):
    """Reuse Step 3's actual viewing orientation after checking contact identity."""
    folder=I.folder(name,'step3.3_optimize_contact',round_number)
    source=folder/'adjusted_contact.npz'
    reference=I.read_contacts(source)[0]
    for key in contact:np.testing.assert_array_equal(reference[key],contact[key])
    path=folder/'adjustment_views.json'
    if path.exists():
        record=json.loads(path.read_text())
        I.check_hashes(record['code'])
        if record['adjustment_sha256']!=I.sha256(folder/'adjustment.json'):
            raise RuntimeError('Step 3 view does not match the current contact optimization')
        for artifact,digest in record['artifacts'].items():
            if I.sha256(folder/artifact)!=digest:raise RuntimeError('Step 3 reference figure changed')
        return np.asarray(record['basis']),dict(source=str(path.relative_to(I.ROOT)),
            source_sha256=I.sha256(path),contact_file=str(source.relative_to(I.ROOT)),
            contact_file_sha256=I.sha256(source),contact_arrays_exactly_equal=True,
            basis=record['basis'],focus_m=record['focus_m'],width_m=record['width_m'])
    basis=R.axes(domain.mesh.face_normals[contact['center_face']])
    return basis,dict(source=None,reason='Step 3 round drawings were skipped; use the contact normal',
        contact_arrays_exactly_equal=True,contact_file=str(source.relative_to(I.ROOT)),
        contact_file_sha256=I.sha256(source),basis=basis.tolist())


def scene(domain,contacts,modules,size,view=None,show_object=True,xray=False,labels=True,
          show_interfaces=False,show_ground=True):
    view=camera(domain,modules) if view is None else view
    points=geometry_points(domain,modules)
    ground=floor_triangles(points,.025*float(domain.mesh.extents.max())) if show_ground else None
    if xray:
        back,_=render(arrays(domain,[],[],show_object=False,ground=ground),view,size)
        back=Image.blend(Image.new('RGB',(size,size),R.PAPER),back,.28)
        if show_object:
            body,body_ids=render(arrays(domain,[],[],show_object=True),view,size)
            back.paste(Image.blend(back,body,.38),mask=Image.fromarray(np.uint8(body_ids>=0)*255))
        front,ids=render(arrays(domain,contacts,modules,show_object=False),view,size)
        if show_interfaces:front=Image.blend(Image.new('RGB',(size,size),R.PAPER),front,.72)
        back.paste(front,mask=Image.fromarray(np.uint8(ids>=0)*255));picture=back
    else:picture,_=render(arrays(domain,contacts,modules,show_object=show_object,ground=ground),view,size)
    if show_interfaces:contact_overlay(picture,contacts,view,size)
    if labels:locators(picture,contacts,view,size)
    return picture


def arrow(ink,tail,tip,color='#4077a5',width=4):
    tail=np.asarray(tail,float);tip=np.asarray(tip,float);delta=tip-tail
    length=np.linalg.norm(delta)
    if length<1e-9:return
    a=delta/length;b=np.array([-a[1],a[0]])
    ink.line([tuple(tail),tuple(tip)],fill=color,width=width)
    ink.polygon([tuple(tip),tuple(tip-16*a+6*b),tuple(tip-16*a-6*b)],fill=color)


def dashed(ink,points,color='#2f8b8f',width=3):
    for a,b in zip(points[:-1],points[1:]):
        length=np.linalg.norm(b-a)
        if length==0:continue
        for s in np.arange(0,length,20):
            ink.line([tuple(a+(b-a)*s/length),tuple(a+(b-a)*min(s+12,length)/length)],fill=color,width=width)


def plan_view(domain,floor,report,modules,size,contacts=()):
    paper=Image.new('RGB',(size,size),R.PAPER);ink=ImageDraw.Draw(paper)
    scale=float(domain.mesh.extents.max());arrows=[]
    points=[COORD.floor(domain.mesh.vertices),floor['required_hull_xz_m'],np.asarray(report['ring']['outer_xz_m'])]
    for entry,data in modules:
        center=np.asarray(entry['anchor_xz_m']);direction=COORD.floor(entry['direction'])
        tail=center-.16*scale*direction;label=center-.22*scale*direction
        arrows.append((tail,center,label,entry));points.extend([COORD.floor(data['ground_corners_m']),label[None]])
    points=np.vstack(points);low=points.min(axis=0);high=points.max(axis=0)
    center=(low+high)/2;width=float(max(high-low)*1.28);factor=size/width
    def xy(p):return (np.asarray(p)-center)*[factor,-factor]+size/2
    def polygon(points,fill):ink.polygon([tuple(p) for p in xy(points)],fill=fill)
    silhouette=COORD.floor(domain.mesh.vertices);boundary=silhouette[ConvexHull(silhouette).vertices]
    polygon(boundary,'#e8e9e5')
    required=floor['required_hull_xz_m'];polygon(required,'#f3dfbc')
    ink.line([tuple(p) for p in xy(np.vstack([required,required[0]]))],fill='#b97e2c',width=3)
    if report['ground']:
        hull=np.asarray(report['ground']['supplied_hull_xz_m'])
        dashed(ink,xy(np.vstack([hull,hull[0]])))
    loop=np.asarray(report['ring']['inner_xz_m'])
    dashed(ink,xy(np.vstack([loop,loop[0]])),color='#92a6a0',width=2)
    for entry,data in modules:
        color=tuple(entry.get('ground_color',[47,139,143]))
        for footprint in entry['ground_polygons_xz_m']:polygon(footprint,color)
    lookup={c['candidate_id']:c for c in contacts}
    for entry,data in modules:
        contact=lookup.get(entry['candidate_id'])
        if contact is None:continue
        contact_pixel=xy(COORD.floor(contact['center_m']));anchor_pixel=xy(entry['anchor_xz_m'])
        dashed(ink,np.array([contact_pixel,anchor_pixel]),color=ORANGE_INK,width=2)
        x,y=contact_pixel;r=max(4,size/180)
        ink.ellipse((x-r,y-r,x+r,y+r),fill=tuple(map(int,R.ORANGE)),outline=ORANGE_INK,width=1)
    for tail,tip,label,entry in arrows:
        arrow(ink,xy(tail),xy(tip))
        kind='ground' if entry['ground_polygons_xz_m'] else 'head'
        ink.text(tuple(xy(label)),f"{entry['candidate_id']} {kind}\n{entry['bearing_deg']:.1f}°",font=R.font(int(size*.024)),fill='#346987',anchor='mm',align='center')
    px,py=xy(COORD.floor(floor['original_pivot_m']));r=6
    ink.line([(px-r,py-r),(px+r,py+r)],fill=R.INK,width=3)
    ink.line([(px+r,py-r),(px-r,py+r)],fill=R.INK,width=3)
    desired=width*.17*1000;unit=10.**np.floor(np.log10(desired))
    length=max(v for v in [unit,2*unit,5*unit] if v<=desired)
    x,y=36,size-52;end=x+length/1000*factor
    ink.line([(x,y),(end,y)],fill=R.INK,width=3)
    for v in [x,end]:ink.line([(v,y-5),(v,y+5)],fill=R.INK,width=2)
    ink.text((x,y+10),f'{length:g} mm',font=R.font(20),fill=MUTED)
    return paper
