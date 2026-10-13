"""Five contact angle sets and five actual swept corridors for every example."""
import argparse
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw
import study as R
import angles as A
from step2_local_support import render as D
from step5_connect_support import visual_details as V

CLEAR='#238d80';LOCAL='#c3645e';REMOTE='#85539a';UNKNOWN='#daa634'
HEAD=np.array([56.,130.,184.]);CORRIDOR=np.array([112.,164.,181.])


def composite(paper, picture, ids, opacity=1.):
    mask=Image.fromarray(np.uint8(ids>=0)*255)
    if opacity<1:picture=Image.blend(paper,picture,opacity)
    paper.paste(picture,mask=mask)


def view(domain, patch, cells, size, angle=None, length=None, hit=None):
    basis=D.axes([.65,-.85,.65])
    vertices=np.vstack([domain.mesh.vertices,*[p.vertices for p in cells]])
    focus,width,basis=V.fit(vertices,basis,margin=1.28)
    object_image,_=D.raster(domain.mesh.triangles,np.tile(D.GREY,(len(domain.mesh.faces),1)),focus,basis,width,size)
    paper=Image.blend(Image.new('RGB',(size,size),D.PAPER),object_image,.28)
    if angle is not None:
        a=A.direction(angle)
        sweeps=[A.hull_mesh(np.vstack([p.vertices,p.vertices-length*a])) for p in cells]
        triangles=np.concatenate([p.triangles for p in sweeps])
        image,ids=D.raster(triangles,np.tile(CORRIDOR,(len(triangles),1)),focus,basis,width,size)
        composite(paper,image,ids,.30)
    triangles=np.concatenate([p.triangles for p in cells])
    image,ids=D.raster(triangles,np.tile(HEAD,(len(triangles),1)),focus,basis,width,size)
    composite(paper,image,ids)
    # Contact is intentionally visible through the faded object and head.
    triangles=patch['triangles_m']
    image,ids=D.raster(triangles,np.tile(D.ORANGE,(len(triangles),1)),focus,basis,width,size)
    composite(paper,image,ids)
    if hit is not None and len(hit):
        image,ids=D.raster(hit,np.tile(np.array([218.,62.,57.]),(len(hit),1)),focus,basis,width,size,unlit=range(len(hit)))
        composite(paper,image,ids)
    ink=ImageDraw.Draw(paper)
    point=D.project(patch['center_m'],focus,basis,width,size)[:2]
    ink.ellipse((*tuple(point-7),*tuple(point+7)),outline='#ad601c',width=2)
    anchor=np.array([size-122.,24.])
    ink.line([tuple(point),tuple(anchor+[-8,12])],fill='#ad601c',width=2)
    ink.rounded_rectangle((anchor[0]-5,anchor[1]-4,size-6,anchor[1]+29),radius=4,fill=D.PAPER)
    ink.text(tuple(anchor),patch['candidate_id'],font=D.font(23),fill='#ad601c')
    if angle is not None:
        screen=A.direction(angle)@basis[:2].T*np.array([1.,-1.])
        screen/=np.linalg.norm(screen)
        middle=np.array([80.,size-35.])
        V.arrow(ink,middle-48*screen,middle+48*screen,'#2c709b',3)
        ink.text((145,size-47),f'+a: {angle%360:.1f}°',font=D.font(21),fill='#2c709b')
    else:
        ink.text((17,size-37),'X-ray view / actual head position',font=D.font(20),fill='#52636c')
    return paper


def sector(ink,center,outer,inner,low,high,color):
    angles=np.deg2rad(np.linspace(low,high,max(2,int(np.ceil((high-low)/.5))+1)))
    vectors=np.c_[np.cos(angles),-np.sin(angles)]
    points=np.vstack([center+outer*vectors,center+inner*vectors[::-1]])
    ink.polygon([tuple(p) for p in points],fill=color)


def wheel(row,size):
    paper=Image.new('RGB',(size,size),D.PAPER);ink=ImageDraw.Draw(paper)
    center=np.array([size/2,size/2]);outer=.35*size
    sector(ink,center,outer,.76*outer,0,360,LOCAL)
    sector(ink,center,.65*outer,.50*outer,0,360,'#e0e3e4')
    for lo,hi in row['local']['intervals_deg']:
        sector(ink,center,.65*outer,.50*outer,lo,hi,'#9bbfdb')
        sector(ink,center,outer,.76*outer,lo,hi,UNKNOWN)
    for key,color in [('geometry_blocked_intervals_deg',REMOTE),('clear_intervals_deg',CLEAR),('unresolved_intervals_deg',UNKNOWN)]:
        for lo,hi in row[key]:sector(ink,center,outer,.76*outer,lo,hi,color)
    for angle,label in [(0,'0° / +X'),(90,'90° / +Y'),(180,'180°'),(270,'270°')]:
        p=center+(outer+32)*np.array([np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))])
        ink.text(tuple(p),label,font=D.font(19),fill=D.INK,anchor='mm')
    for i,example in enumerate(row['example_directions']):
        angle=np.deg2rad(example['angle_deg']);u=np.array([np.cos(angle),-np.sin(angle)])
        color=CLEAR if example['clear'] else '#a03e3a'
        # Radial arrows represent vector a, not the side from which the head starts.
        V.arrow(ink,center,center+outer*.43*u,color,2)
        ink.text(tuple(center+(outer+9)*u),str(i+1),font=D.font(18),fill=color,anchor='mm')
    for item in row['isolated_direction_checks']:
        angle=np.deg2rad(item['angle_deg']);p=center+.88*outer*np.array([np.cos(angle),-np.sin(angle)])
        ink.ellipse((*tuple(p-4),*tuple(p+4)),fill=CLEAR if item['clear'] else LOCAL,outline='white',width=1)
    return paper


def interval_text(intervals):
    if not intervals:return 'None'
    return '  U  '.join(f'[{a:.3f}°, {b:.3f}°]' for a,b in intervals)


def text_lines(ink,xy,message,font_size=24,width=76,fill=D.INK):
    y=xy[1]
    for line in textwrap.wrap(message,width=width):
        ink.text((xy[0],y),line,font=D.font(font_size),fill=fill)
        y+=font_size+9
    return y


def summary(name,domain,contacts,rows,heads,owners,out):
    page=Image.new('RGB',(2200,3040),D.PAPER);ink=ImageDraw.Draw(page)
    ink.text((45,25),f'{name} / Single-contact insertion / Five angle sets',font=D.font(43),fill=D.INK)
    ink.text((48,89),'Fixed head orientation; horizontal straight insertion. Only the object is an obstacle.',font=D.font(28),fill=D.INK)
    ink.text((48,137),'Inner ring: local normals allow motion. Outer: green = whole path clear; purple = geometry blocks; red = local block; yellow = unresolved.',font=D.font(23),fill='#53646c')
    for i,(patch,row) in enumerate(zip(contacts,rows)):
        top=205+550*i
        cells=[h for h,o in zip(heads,owners) if o['candidate_id']==patch['candidate_id']]
        ink.text((48,top),f'{patch["candidate_id"]} / radius {patch["radius_m"]*1000:.3f} mm',font=D.font(29),fill=D.INK)
        page.paste(view(domain,patch,cells,450),(35,top+44))
        page.paste(wheel(row,450),(535,top+44))
        x=1050
        ink.text((x,top+48),'Whole-path clear intervals',font=D.font(28),fill=CLEAR)
        y=text_lines(ink,(x,top+91),interval_text(row['clear_intervals_deg']),26,67,CLEAR)
        y=text_lines(ink,(x,y+20),'Local-only intervals: '+interval_text(row['local']['intervals_deg']),23,78,'#456c8b')
        unknown=sum(b-a for a,b in row['unresolved_intervals_deg'])
        ink.text((x,y+17),f'Unresolved total: {unknown:.5f}°   |   Full path length: {row["length_m"]*1000:.1f} mm',font=D.font(23),fill='#76612d')
        for j,example in enumerate(row['example_directions']):
            xx=x+(j%3)*345;yy=top+340+(j//3)*64
            color=CLEAR if example['clear'] else '#a03e3a'
            ink.text((xx,yy),f'{j+1}. {example["angle_deg"]:.2f}°  '+('CLEAR' if example['clear'] else 'BLOCKED'),font=D.font(23),fill=color)
        ink.text((x,top+484),f'Five corridors: {patch["candidate_id"]}.png',font=D.font(22),fill='#53646c')
        ink.line([(40,top+534),(2160,top+534)],fill='#d5d9d9',width=2)
    ink.text((48,2970),'Angles describe the movement vector +a; the head starts on the opposite side, -a. Green intervals check continuous motion and continuous angles.',font=D.font(22),fill='#53646c')
    ink.text((48,3006),'Mesh geometry and stated tolerance; boundary uncertainty is retained in angles.json. No floor, connectors, other contacts or rotations are included.',font=D.font(22),fill='#53646c')
    page.save(out/'contact_angles.png')


def contact_page(name,domain,patch,row,cells,intersections,out):
    page=Image.new('RGB',(2200,1390),D.PAPER);ink=ImageDraw.Draw(page)
    ink.text((45,24),f'{name} / {patch["candidate_id"]} / Five insertion directions',font=D.font(43),fill=D.INK)
    ink.text((48,88),f'Head radius: {patch["radius_m"]*1000:.3f} mm   |   Same contact, same head, same final pose',font=D.font(28),fill=D.INK)
    page.paste(view(domain,patch,cells,450),(30,145))
    page.paste(wheel(row,450),(520,145))
    y=text_lines(ink,(1050,184),'Whole-path clear: '+interval_text(row['clear_intervals_deg']),28,66,CLEAR)
    y=text_lines(ink,(1050,y+27),'Blue: final head. Orange: contact surface. Translucent blue: its full swept corridor.',25,70)
    y=text_lines(ink,(1050,y+20),'Red geometry: independently computed object/sweep intersection for one colliding head cell.',25,70,'#a03e3a')
    text_lines(ink,(1050,y+20),'Arrows show insertion +a. The corridor extends backward along -a and may continue beyond the picture.',24,72,'#53646c')
    for i,case in enumerate(row['example_directions']):
        x=25+435*i;color=CLEAR if case['clear'] else '#a03e3a'
        ink.text((x+12,648),f'{i+1}. {case["angle_deg"]:.2f}°',font=D.font(30),fill=D.INK)
        ink.text((x+12,696),'CLEAR' if case['clear'] else 'BLOCKED',font=D.font(27),fill=color)
        hit=intersections[f'{patch["candidate_id"]}_{i}']
        page.paste(view(domain,patch,cells,420,case['angle_deg'],row['length_m'],hit),(x,752))
    ink.text((48,1222),'Every corridor is swept from the actual finite-thickness head. The faded object stays fixed, and hidden heads remain visible.',font=D.font(24),fill=D.INK)
    ink.text((48,1268),'Clear means the entire straight path is collision-free within tolerance. The local normal condition alone is insufficient.',font=D.font(24),fill=D.INK)
    ink.text((48,1320),'No floor or base in this experiment. Cases show one contact head moving independently; these five contacts are not an assembly.',font=D.font(23),fill='#53646c')
    page.save(out/f'{patch["candidate_id"]}.png')


def run(name):
    out=R.OUTPUT/name;report=R.I.check_report(out/'angles.json')
    audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['angles_sha256']==R.P.sha256(out/'angles.json')
    assert audit['intersection_geometry_sha256']==R.P.sha256(out/'example_intersections.npz')
    domain,contacts,_=R.inputs(name)
    heads,owners=R.build_heads(domain.mesh,contacts,report['normal_depth_m'])
    hits=R.I.load_npz(out/'example_intersections.npz')
    summary(name,domain,contacts,report['contacts'],heads,owners,out)
    images=['contact_angles.png']
    for patch,row in zip(contacts,report['contacts']):
        cells=[h for h,o in zip(heads,owners) if o['candidate_id']==patch['candidate_id']]
        contact_page(name,domain,patch,row,cells,hits,out)
        images.append(patch['candidate_id']+'.png')
    R.I.save(out/'views.json',dict(object=name,angles_sha256=R.P.sha256(out/'angles.json'),
        audit_sha256=R.P.sha256(out/'audit.json'),drawing_code_sha256=R.P.sha256(__file__),
        rendering_code=R.I.hashes([Path(D.__file__),Path(V.__file__)]),
        images={p:R.P.sha256(out/p) for p in images},
        object_transparency='28% opacity X-ray backdrop; heads and collision evidence in foreground',
        corridors='True per-convex-cell sweeps, clipped only by the image frame; no whole-head convex hull',
        locator_markers_enlarged=True))
    print(name,'wrote contact_angles.png and five contact detail pages',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or R.P.OBJECTS:run(name)
