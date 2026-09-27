"""Render three tasks with one contact module and one shared ground base."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'slides/tools'))
import slide_scene as S
from fixture_geometry import VIEW, verify

BLUE = (37, 112, 188)
ORANGE = (235, 157, 48)
RED = (186, 48, 61)
S.GREEN = (78, 184, 111)


def interface_details(sheet, cases, top):
    """The close-ups share a local view so the repeated docking is unambiguous."""
    pen = ImageDraw.Draw(sheet)
    c = cases[0]
    empty = trimesh.Trimesh(vertices=np.empty((0,3)), faces=np.empty((0,3),int), process=False)
    domain = SimpleNamespace(mesh=empty, work_ids=np.array([],int))
    for i, separated in enumerate((True, False)):
        slider = c['slider_detail'].copy()
        delta = c['direction'] * (.030 if separated else 0.)
        if separated:
            slider.apply_translation(delta)
        parts = [(slider, BLUE), (c['channel'], ORANGE)]
        cloud = np.vstack([m.vertices for m,_ in parts])
        view = c['basis'] @ np.array([.8, 1.5, 1.3])
        basis = S.R.axes(view)
        bounds = np.array([(cloud@basis.T).min(0),(cloud@basis.T).max(0)])
        cam = S.Camera(bounds.mean(0)@basis,basis,.105,500)
        # Keep real nearby object/frame/base context in the same camera, but
        # ghost it. The sharp joint is rendered separately on top so fading
        # context cannot hide the rectangular opening. This is an explanatory
        # transparency view, not a claim of visibility through opaque material.
        context_object = c['object'].copy().apply_translation(delta)
        context_contact = c['contact'].copy().apply_translation(delta)
        context_domain = SimpleNamespace(mesh=context_object, work_ids=np.array([],int))
        context,_,_ = S.render(context_domain,parts=[(context_contact,(180,180,180)),
            (c['base'],(180,180,180))],cam=cam,ground=False)
        pic = Image.blend(context.filter(ImageFilter.GaussianBlur(1.3)),
                          Image.new('RGB',context.size,'white'),.87)
        joint,_,ids = S.render(domain,parts=parts,cam=cam,ground=False)
        pic.paste(joint,(0,0),Image.fromarray(np.uint8(ids>=0)*255))
        if separated:
            draw = ImageDraw.Draw(pic)
            p = c['port'] + c['basis'][:,0]*.022
            S.arrow(draw,cam.project(p+c['direction']*.032)[:2],cam.project(p)[:2],BLUE,6,18)
        sheet.paste(pic,(45+i*490,top))
        S.text(pen,(295+i*490,top+475),'One straight push (30 mm)' if separated else 'Seated at the end stop',27,S.MUTED)
    S.text(pen,(1110,top+145),'Same blue module. Plain rectangular joint.',42,S.INK,'lm')
    S.text(pen,(1110,top+215),'One fixed orange base. Three task-specific sockets.',33,S.MUTED,'lm')
    S.text(pen,(1110,top+275),'Interface position and push direction follow the pose.',31,S.MUTED,'lm')
    S.text(pen,(1110,top+335),'Joint highlighted; surrounding geometry faded.',31,S.MUTED,'lm')
    S.text(pen,(1110,top+395),'End stop only; no lock against withdrawal.',29,S.MUTED,'lm')


def draw(cases, path, forces=False):
    count=len(cases)
    size=1000 if count==3 else 1500
    prepared=[]
    for c in cases:
        domain=SimpleNamespace(mesh=c['object'],work_ids=c['work_ids'])
        parts=[(c['base'],ORANGE),(c['contact'],BLUE)]
        cloud=np.vstack([m.vertices for m,_ in parts])
        cam=S.camera(domain,size=size,extra=cloud,ground_points=cloud,view=VIEW)
        prepared.append((domain,parts,cloud,cam))
    width=max(p[3].width for p in prepared)
    top=180
    detail_top=top+size-15
    sheet=Image.new('RGB',(3000,detail_top+600),'white')
    pen=ImageDraw.Draw(sheet)
    S.text(pen,(65,65),'One contact module. One fixed base. Three tasks.',51,S.INK,'lm')
    S.text(pen,(65,128),'Blue: shared contact module + rectangular peg     Orange: one base with three fixed sockets     Green: working area',30,S.MUTED,'lm')
    for i,(c,(domain,parts,cloud,cam)) in enumerate(zip(cases,prepared)):
        cam.width=width
        pic,_,_=S.render(domain,parts=parts,cam=cam,ground_points=cloud)
        ink=ImageDraw.Draw(pic)
        S.text(ink,(55,55),c['label'],38,S.INK,'lm')
        subtitle=f'Task {i+1} / dock {i+1}' + (' · illustrative pose' if c['pose_kind']=='illustrative_tilt' else '')
        S.text(ink,(55,103),subtitle,28,S.MUTED,'lm')
        if forces:
            for point,normal in zip(c['points'],c['normals']):
                a,b=cam.project(point)[:2],cam.project(point+.043*normal)[:2]
                S.arrow(ink,a,b,'white',14,29)
                S.arrow(ink,a,b,RED,8,24)
                ink.ellipse((*tuple(a-5),*tuple(a+5)),fill=RED,outline='white',width=2)
        sheet.paste(pic,(i*size,top))
    interface_details(sheet,cases,detail_top)
    note='Geometry concept: loads, tool clearance, robot grasp and module retention are not validated.'
    S.text(pen,(65,sheet.height-65),note,27,S.MUTED,'lm')
    if forces:
        S.text(pen,(65,sheet.height-27),'Red arrows illustrate contact-normal directions, not solved support forces.',25,RED,'lm')
    sheet.save(path)
    print(path,flush=True)


def main():
    from shared_base import build
    cases,_,meta=build()
    report=verify(cases,meta)
    (HERE/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    draw(cases,HERE.parent/'three_poses.png')


if __name__=='__main__':
    main()
