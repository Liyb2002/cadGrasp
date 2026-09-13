"""Inspect successful or failed three-head schedules using their actual contacts."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.stage_imports import load_stage
from step1.cases import selected_pose,pose_name
from step2_local_support import render as R
C=load_stage('score','contribution')
V=load_stage('select','draw')
COLORS=np.array([[49,127,195],[245,139,37],[146,101,180]],float)


def run(name):
    out=I.OUTPUTS/name/pose_name()/'step3_scheculer'
    schedule=I.check_report(out/'schedule.json')
    domain,_,_=C.P.read(name)
    contacts=I.read_contacts(out/'final_contacts.npz')
    data=I.as_data(contacts) if contacts else None
    base=domain.mesh.triangles
    colors=np.tile(R.GREY,(len(base),1));colors[domain.work_ids]=R.GREEN
    triangles=[base];palette=[colors]
    for j,c in enumerate(contacts):
        triangles.append(c['triangles_m']);palette.append(np.tile(COLORS[j],(len(c['triangles_m']),1)))
    triangles=np.concatenate(triangles);palette=np.concatenate(palette)
    picture=Image.new('RGB',(1920,1090),R.PAPER);ink=ImageDraw.Draw(picture)
    passed=schedule['continuous_coverage_proved']
    ink.text((28,22),f'{name} / {pose_name()} / Step3 / '+('COVERAGE VERIFIED' if passed else 'INCOMPLETE COVERAGE'),font=R.font(34),fill=R.INK)
    ink.text((28,80),f"{len(contacts)} / 3 heads; coverage {schedule['covered_percent']:.3f}%; {len(schedule['common_withdrawal_directions']['ids'])} shared withdrawal directions",font=R.font(25),fill=R.INK)
    ink.text((28,128),f"Stop: {schedule['status']}",font=R.font(25),fill=R.INK)
    views=[]
    for j in range(max(1,len(contacts))):
        basis=V.contact_camera(domain,data,j) if contacts else R.axes([1, -1, .5])
        focus,width=R.overall_camera(domain,basis)
        scene,ids=R.raster(triangles,palette,focus,basis,width,620,overlay=np.arange(len(base),len(triangles)))
        picture.paste(scene,(15+635*j,230))
        label=f"View toward {contacts[j]['candidate_id']}" if contacts else 'No admissible head selected'
        ink.text((30+635*j,185),label,font=R.font(25),fill=R.INK)
        views.append(dict(basis=basis.tolist(),focus_m=focus.tolist(),width_m=float(width)))
    y=860
    for row in schedule['rounds']:
        ink.text((28,y),f"Round {row['round']}: {row['candidate_id']}  |  after sizing: {row['after_optimization_covered_percent']:.3f}%  |  directions: {len(row['insertion']['common_directions']['ids'])}",font=R.font(24),fill=R.INK)
        y+=38
    if not contacts:
        ink.text((28,y),'Every candidate failed geometry, common-direction or rest-equilibrium eligibility.',font=R.font(24),fill=R.INK)
    ink.text((28,1005),'Each accepted round: common head path + gravity equilibrium + shared no-uplift constraint.',font=R.font(22),fill=R.INK)
    ink.text((28,1045),'Colored regions are actual selected contacts. Whole-frame collision and finite-base tipping remain to be checked.',font=R.font(20),fill=R.INK)
    picture.save(out/'schedule.png')
    I.save(out/'schedule_views.json',dict(complete=True,selected_ids=schedule['selected_ids'],views=views,
        provenance=dict(inputs=I.hashes([out/'schedule.json',out/'final_contacts.npz']),
                        code=I.hashes([Path(__file__),Path(R.__file__),Path(V.__file__)])),
        artifacts={'schedule.png':I.sha256(out/'schedule.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects', nargs='*', default=['A1-f', 'B', 'C5']);parser.add_argument('--pose',default=None)
    args=parser.parse_args()
    with selected_pose(args.pose):
        for name in args.objects: run(name)
