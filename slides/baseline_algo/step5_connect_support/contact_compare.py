"""Compare a Step 3 contact and its actual Step 5 support in the same camera."""
import argparse
from pathlib import Path
import sys
import numpy as np
import trimesh
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import connect as C,solids as S,ground as G,visual_details as V
from step5_connect_support.surface_check import surface_distances

from step1.cases import pose_name


def run(name,candidate_id):
    domain,contacts,schedule,_,_,depth,_=C.read_inputs(name)
    out=C.OUTPUTS/name/pose_name()/C.STAGE
    report=C.I.check_report(out/'connection.json')
    current=next(c for c in contacts if c['candidate_id']==candidate_id)
    round_number=next(r['round'] for r in schedule['rounds'] if r['candidate_id']==candidate_id)
    source=C.I.folder(name,'step3.3_optimize_contact',round_number)/'adjusted_contact.npz'
    original=C.I.read_contacts(source)[0]
    for key in original:np.testing.assert_array_equal(original[key],current[key])
    basis,reference=V.reference_camera(name,current,round_number,domain)
    if reference['source'] is None:raise ValueError('Generate the Step 3 round figure first for an exact camera comparison')
    view=(np.asarray(reference['focus_m']),reference['width_m'],basis)
    entry=next(s for s in report['supports'] if s['candidate_id']==candidate_id)
    path=out/entry['folder']/'geometry.npz';data=C.I.load_npz(path)
    joined=trimesh.Trimesh(data['union_vertices_m'],data['union_faces'],process=False)
    scale=float(domain.mesh.extents.max())
    points=np.vstack([current['triangles_m'].reshape(-1,3),current['triangles_m'].mean(axis=1)])
    distances=surface_distances(joined,points)
    assert distances.max()<scale*1e-9
    head,_=S.union_parts(C.D.Analyzer(domain.mesh,depth*entry.get('backing_depth_factor',1.)).heads(current),scale)
    origin=domain.mesh.bounds.mean(axis=0)
    difference=G.solid64(head,origin,scale)-G.solid64(joined,origin,scale)
    if difference.status()!=G.manifold.Error.NoError:raise RuntimeError('Head containment boolean failed')
    missing=abs(float(difference.volume()))*scale**3
    assert missing<=scale**3*1e-11

    size=960;page=Image.new('RGB',(2040,1750),V.R.PAPER);ink=ImageDraw.Draw(page)
    ink.text((40,24),f'{name} / {candidate_id} / Step 3 and Step 5',font=V.R.font(43),fill=V.R.INK)
    ink.text((43,87),'Same camera, scale and world coordinates. X-ray: the orange interface is shown through solids.',
             font=V.R.font(27),fill=V.MUTED)
    for i,(label,contact,modules) in enumerate([
            ('Step 3 / optimized contact',original,[]),
            ('Step 5 / contact + added connection',current,[(entry,data)])]):
        x=40+i*1000
        ink.text((x,157),label,font=V.R.font(30),fill=V.R.INK)
        picture=V.scene(domain,[contact],modules,size,view=view,xray=True,
                        show_interfaces=True,show_ground=False)
        page.paste(picture,(x,200))
        detail_view=(current['center_m'],2.8*current['radius_m'],basis)
        detail=V.scene(domain,[contact],modules,440,view=detail_view,xray=True,
                       show_interfaces=True,show_ground=False,labels=False)
        page.paste(detail,(x+260,1155))
    center=current['center_m']*1000
    ink.text((43,1612),f'Contact center in both stages (mm): ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})',
             font=V.R.font(27),fill=V.R.INK)
    ink.text((43,1658),f"Area: {current['triangle_areas_m2'].sum()*1e6:.6f} mm². Contact triangles match exactly; selected backing retained in the exported solid.",
             font=V.R.font(25),fill=V.R.INK)
    ink.text((43,1701),'Blue bars are added connection material. It is not the circular object-contact area.',
             font=V.R.font(25),fill=V.MUTED)
    filename=f'contact_comparison_{candidate_id}.png';page.save(out/filename)
    record=dict(object=name,candidate_id=candidate_id,complete=True,
        contact_arrays_exactly_equal=True,center_m=current['center_m'].tolist(),
        area_m2=float(current['triangle_areas_m2'].sum()),radius_m=float(current['radius_m']),
        max_contact_to_exported_surface_distance_m=float(distances.max()),
        selected_backing_missing_volume_m3=missing,backing_depth_factor=entry.get('backing_depth_factor',1.),volume_tolerance_m3=scale**3*1e-11,
        same_camera_and_scale=True,reference_camera=reference,
        contact_interfaces_shown_through_solids=True,
        provenance=dict(inputs=C.I.hashes([source,path,out/'connection.json',
            C.OUTPUTS/name/pose_name()/'step3_scheculer/final_contacts.npz',C.I.ROOT/reference['source']]),
            code=C.I.hashes([Path(__file__),Path(V.__file__),Path(S.__file__),Path(G.__file__)])),
        artifacts={filename:C.sha256(out/filename)})
    C.I.save(out/f'contact_comparison_{candidate_id}.json',record)
    print(name,candidate_id,'same coordinates; contact comparison written',flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object');parser.add_argument('candidate_id')
    args=parser.parse_args();run(args.object,args.candidate_id)
