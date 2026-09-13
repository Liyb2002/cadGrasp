"""Draw one rigid support or its explicit failure; animate all parts together."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step5_connect_support import connect as C,visual_details as V
from step1.cases import pose_name
R=V.R


def references(floor,successful):
    pairs=[]
    for foot in floor['report']['ground_footprints']:
        cid=foot['candidate_id']
        if cid in successful:continue
        parts=C.H.pad_parts(foot);joined,_=C.S.union_parts(parts,1.)
        entry=dict(candidate_id=cid,ground_color=[180,185,186],reference_only=True)
        arrays=C.S.pack_parts(parts,[f'ground_pad_{k:03d}' for k in range(len(parts))],joined)
        pairs.append((entry,arrays))
    return pairs


def overview(name,domain,contacts,modules,floor,report,out):
    page=Image.new('RGB',(2400,1290),R.PAPER);ink=ImageDraw.Draw(page)
    ink.text((35,25),f'{name} / {pose_name()} / Step 5 / Fixed feet',font=R.font(42),fill=R.INK)
    ink.text((38,88),f"{report['support_count']} / {report['contact_count']} individual supports constructed and insertable",font=R.font(29),fill='#24565b')
    shown=modules+references(floor,{r['candidate_id'] for r,_ in modules})
    view=V.camera(domain,shown)
    page.paste(V.scene(domain,contacts,shown,1130,view=view,xray=True,show_interfaces=True),(25,135))
    page.paste(V.scene(domain,contacts,shown,1130,view=view,show_object=False,show_interfaces=True),(1240,135))
    ink.text((38,1160),'Blue: completed connections. Colored pads: completed supports. Grey pads: unconnected fixed targets.',font=R.font(24),fill=R.INK)
    state=(f"Construction in progress: {report['processed_count']} / {report['contact_count']} processed; assembly check pending"
           if report.get('complete') is False else
           'Complete assembly geometry verified' if report['geometric_assembly_verified'] else
           f"Compatible subset: {len(report['assemblable_subset_ids'])} supports; full assembly not verified")
    ink.text((38,1200),state,font=R.font(26),fill=V.MUTED)
    bearing='certified for all allowed loads' if report['step4_bearing_certificate_verified'] else 'not certified for all allowed loads'
    ink.text((38,1242),'Step 4 bearing: '+bearing+'. Individual success does not certify an incomplete set.',font=R.font(23),fill=V.MUTED)
    # Keep the previous complete preview if rendering or saving is interrupted.
    temporary=out/'.connection.png'
    page.save(temporary)
    temporary.replace(out/'connection.png')


def progress_snapshot(name,domain,contacts,modules,floor,records,out):
    """Publish actual partial geometry before another support or assembly runs."""
    if records:
        record=records[-1];cid=record['candidate_id']
        contact=next(c for c in contacts if c['candidate_id']==cid)
        pairs=[pair for pair in modules if pair[0]['candidate_id']==cid]
        if not pairs:
            pairs=[pair for pair in references(floor,set()) if pair[0]['candidate_id']==cid]
        cell=Image.new('RGB',(720,790),R.PAPER);ink=ImageDraw.Draw(cell)
        ink.text((15,12),cid+(' / constructed' if record['passed'] else ' / not constructed'),
                 font=R.font(27),fill=R.INK)
        cell.paste(V.scene(domain,[contact],pairs,700,xray=True,show_interfaces=True),(10,65))
        destination=out/'supports'/cid;destination.mkdir(parents=True,exist_ok=True)
        cell.save(destination/'support.png')
    report=dict(complete=False,processed_count=len(records),contact_count=len(contacts),
                support_count=len(modules),geometric_assembly_verified=False,
                assemblable_subset_ids=[],
                step4_bearing_certificate_verified=bool(floor['report'].get('continuous_domain_coverage_proved')))
    overview(name,domain,contacts,modules,floor,report,out)


def positions(name,domain,contacts,modules,floor,report,out,schedule):
    columns=min(3,max(1,len(contacts)));rows=max(1,int(np.ceil(len(contacts)/columns)))
    page=Image.new('RGB',(40+columns*720,150+rows*830),R.PAPER);ink=ImageDraw.Draw(page)
    ink.text((30,22),f'{name} / {pose_name()} / Individual results',font=R.font(36),fill=R.INK)
    ink.text((32,75),'Every selected contact is shown. Failed connections retain their fixed foot targets.',font=R.font(22),fill=V.MUTED)
    lookup={entry['candidate_id']:(entry,data) for entry,data in modules}
    ghosts={entry['candidate_id']:(entry,data) for entry,data in references(floor,set(lookup))}
    statuses={r['candidate_id']:r for r in report['support_results']};views={}
    for i,contact in enumerate(contacts):
        cid=contact['candidate_id'];entry=lookup.get(cid,ghosts.get(cid));pairs=[entry] if entry else []
        round_number=next(r['round'] for r in schedule['rounds'] if r['candidate_id']==cid)
        basis,reference=V.reference_camera(name,contact,round_number,domain);views[cid]=reference
        view=V.camera(domain,pairs,basis)
        picture=V.scene(domain,[contact],pairs,700,view=view,xray=True,show_interfaces=True)
        cell=Image.new('RGB',(720,830),R.PAPER);label=ImageDraw.Draw(cell)
        success=statuses[cid]['passed']
        label.text((15,12),cid+(' / constructed' if success else ' / not constructed'),font=R.font(27),fill='#24565b' if success else '#a95843')
        cell.paste(picture,(10,56))
        status=statuses[cid]['status'].replace('_',' ')
        words=status.split();lines=[];line=''
        for word in words:
            if len(line)+len(word)>48:lines.append(line);line=word
            else:line=(line+' '+word).strip()
        if line:lines.append(line)
        label.text((15,760),'\n'.join(lines),font=R.font(19),fill=V.MUTED)
        destination=out/'supports'/cid;destination.mkdir(parents=True,exist_ok=True);cell.save(destination/'support.png')
        page.paste(cell,(20+(i%columns)*720,125+(i//columns)*830))
    if not contacts:ink.text((35,160),'No selected contact geometry is available.',font=R.font(27),fill=V.MUTED)
    page.save(out/'contact_positions.png');return views


def ground_sheet(name,domain,floor,report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from step4_floor_contact.draw import boundary,color
    fig,ax=plt.subplots(figsize=(10,10),dpi=150);fig.patch.set_facecolor('#ffffff');ax.set_facecolor('#ffffff')
    hull=boundary(COORD.floor(domain.mesh.vertices))*1000;ax.fill(*hull.T,color='#c5cbcc',alpha=.4)
    success={r['candidate_id'] for r in report['supports']}
    for j,foot in enumerate(floor['report']['ground_footprints']):
        good=foot['candidate_id'] in success
        for polygon in foot['pads_xy_m']:
            xy=np.asarray(polygon)*1000;ax.fill(*xy.T,color=color(j) if good else '#b4b9ba')
        points=np.asarray(foot['hull_xy_m'])*1000
        ax.plot(*np.vstack([points,points[0]]).T,color=color(j),alpha=.45,ls='--',lw=1)
        label=points[np.argmax(points[:,1])]
        ax.annotate(foot['candidate_id']+('' if good else ' (no connection)'),label,xytext=(3,5),textcoords='offset points',fontsize=9,color=color(j))
    ax.set_aspect('equal');ax.set_axis_off();ax.margins(.1)
    fig.suptitle(f'{name} / {pose_name()} / Fixed Step 4 feet',fontsize=19,color='#253a3e')
    fig.text(.04,.04,'Solid pads are the only ground-bearing material. Dashed hulls are not solid plates.',fontsize=10)
    fig.savefig(out/'ground_feet.png');plt.close(fig)


def animate(name,domain,contacts,modules,order_ids,destination,label):
    if not order_ids:return
    lookup={r['candidate_id']:(r,a) for r,a in modules};selected=[lookup[cid] for cid in order_ids]
    framing=np.vstack([domain.mesh.vertices]+[a['union_vertices_m']+shift for entry,a in selected
        for shift in [np.zeros(3),np.asarray(entry['trajectory']['start_translation_m'])]])
    view=V.fit(framing,margin=1.10);ground=V.floor_triangles(framing,.04*float(domain.mesh.extents.max()))
    frames=[];installed=[];size=620
    for number,(entry,data) in enumerate(selected,1):
        static=V.layer(V.arrays(domain,contacts,installed,ground=ground),view,size)
        start=np.asarray(entry['trajectory']['start_translation_m'])
        for fraction in np.linspace(0,1,10):
            moving=V.layer(V.arrays(domain,contacts,[(entry,data)],translations=[(1-fraction)*start],show_object=False),view,size)
            page=Image.new('RGB',(680,790),R.PAPER);ink=ImageDraw.Draw(page)
            ink.text((25,18),f'{name} / {entry["candidate_id"]} / {number} of {len(selected)}',font=R.font(27),fill=R.INK)
            ink.text((25,62),label,font=R.font(20),fill=V.MUTED)
            page.paste(V.merge_layers(static,moving),(30,110))
            ink.text((25,752),'Object held fixed during installation.',font=R.font(20),fill=V.MUTED)
            frames.append(page)
        installed.append((entry,data))
    frames[0].save(destination,save_all=True,append_images=frames[1:],duration=150,loop=0)


# Current one-body entry; shared geometry and regression helpers stay available.
from step5_connect_support.belt_assembly import draw as draw_assembly


def run(name, static_only=False):
    out = C.OUTPUTS/name/pose_name()/C.STAGE
    for filename in ('bearing_failure.json', 'bearing_failure.npz', 'bearing_failure.png'):
        (out/filename).unlink(missing_ok=True)
    for filename in ('connectivity_failure.json', 'connectivity_failure.png',
                     'connectivity_union_mm.stl', 'connectivity_extra_shells_mm.stl'):
        (out/filename).unlink(missing_ok=True)
    draw_assembly(name, static_only)
    from step5_connect_support.bearing_failure import run as draw_bearing_failure
    draw_bearing_failure(name)
    from step5_connect_support.connectivity_failure import run as draw_connectivity_failure
    draw_connectivity_failure(name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects',nargs='*');parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args()
    for name in args.objects or C.OBJECTS:run(name,args.static_only)
