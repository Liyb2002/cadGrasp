"""Build one connected rigid support from every selected head and a shared base.

Earlier independent-support helpers remain below for historical callers; the
current build/schema are imported from whole_assembly before main."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step1.needs import ContinuousNeeds,OUTPUTS,OBJECTS,sha256
from step1.cases import pose_name
from step2_local_support import insertion as D,work_volume as W
from step4_floor_contact import floor_contact as F
from step5_connect_support import fixed_feet as H,layout as L,solids as S,motion as M,ground as G,routing as T
from step5_connect_support import surface_check as U
from step5_connect_support import visual_details as V

STAGE='step5_connect_support'
SCHEMA='fixed_feet_individual_connections_v2'
COLORS=[[41,139,147],[210,128,44],[106,120,187],[162,90,142],[99,151,94],[164,126,89]]


def export_geometry(destination,module):
    plan=module['plan']
    np.savez_compressed(destination/'geometry.npz',**S.pack_parts(module['parts'],module['labels'],module['joined']),
        insertion_direction=plan['direction'],ground_corners_m=plan['ground_corners_m'],
        ground_polygon_offsets=np.cumsum([0]+list(map(len,plan['ground_polygons_xz_m']))))
    module['joined'].export(destination/'support.stl',file_type='stl_ascii')


def read_inputs(name):
    root=OUTPUTS/name/pose_name()
    schedule=I.check_report(root/'step3_scheculer/schedule.json')
    state=json.loads((root/'step3_scheculer/status.json').read_text())
    if not state.get('complete') or state.get('schedule_sha256')!=sha256(root/'step3_scheculer/schedule.json'):
        raise RuntimeError('Step 3 has no completed current search result')
    domain=ContinuousNeeds.read(root/'step_1_needs/needs.json')
    if not domain.mesh.is_watertight or not domain.mesh.is_winding_consistent or domain.mesh.volume<=0:
        raise ValueError('Step 5 needs a closed outward-oriented object mesh')
    contact_path=root/'step3_scheculer/final_contacts.npz'
    contacts=I.read_contacts(contact_path)
    directions=I.check_report(root/'step3_scheculer/insertion_directions.json')
    assert directions['mode']=='independent_contacts'
    assert directions['selected_ids']==schedule['selected_ids']==[c['candidate_id'] for c in contacts]
    assert len(directions['contacts'])==len(contacts)
    for contact,record in zip(contacts,directions['contacts']):
        assert record['geometry_signature']==D.signature(contact,directions['normal_depth_m'])
    floor_report=F.read(name)
    assert floor_report['foot_positions_frozen_for_step5'] and not floor_report['connectors_may_move_feet']
    assert floor_report['selected_ids']==schedule['selected_ids']
    floor=I.load_npz(root/F.STAGE/floor_report['arrays_file'])
    floor['report']=floor_report
    paths=[root/'step_1_needs/needs.json',root/'step_1_needs/samples.json',
        root/'step3_scheculer/schedule.json',root/'step3_scheculer/status.json',contact_path,
        root/'step3_scheculer/insertion_directions.json',root/F.STAGE/'floor_contact.json',root/F.STAGE/'floor_contact.npz',
        root/F.STAGE/'audit.json',root/W.STAGE/'work_volume.json',root/W.STAGE/'work_volume.npz',root/W.STAGE/'work_volume_audit.json']
    assert I.check_report(root/F.STAGE/'audit.json')['passed']
    return domain,contacts,schedule,floor,directions,float(directions['normal_depth_m']),paths


def build(name):
    started=time.monotonic()
    domain,contacts,schedule,floor,directions,depth,inputs=read_inputs(name)
    out=OUTPUTS/name/pose_name()/STAGE;out.mkdir(parents=True,exist_ok=True)
    for folder in ('supports','candidate_supports'):
        if (out/folder).exists():shutil.rmtree(out/folder)
    for filename in ('connection.json','audit.json','views.json','connection.png','contact_positions.png',
                     'ground_ring.png','ground_feet.png','insertion.gif','physical_check.json','progress.json'):
        (out/filename).unlink(missing_ok=True)
    I.save(out/'status.json',dict(complete=False,status='constructing_individual_fixed_feet'))
    input_hashes=I.hashes(inputs)
    volume_path=out.parent/W.STAGE/'work_volume.json'
    assert I.check_report(volume_path.parent/'work_volume_audit.json')['passed']
    work=W.WorkVolume.read(volume_path)
    from step5_connect_support import draw
    progress=[];previews=[]
    by_contact={c['candidate_id']:i for i,c in enumerate(contacts)}
    draw.progress_snapshot(name,domain,contacts,previews,floor,progress,out)
    def retain_one(record,module):
        cid=record['candidate_id'];destination=out/'supports'/cid
        destination.mkdir(parents=True,exist_ok=True)
        I.save(destination/'status.json',record)
        if module is not None:
            export_geometry(destination,module)
            entry=dict(candidate_id=cid,ground_color=COLORS[by_contact[cid]%len(COLORS)])
            arrays=S.pack_parts(module['parts'],module['labels'],module['joined'])
            previews.append((entry,arrays))
        progress.append(dict(candidate_id=cid,passed=record['passed'],status=record['status']))
        draw.progress_snapshot(name,domain,contacts,previews,floor,progress,out)
        I.save(out/'progress.json',dict(complete=False,processed_count=len(progress),
            contact_count=len(contacts),support_results=progress,
            scope='Individual artifacts available immediately; final assembly audit still pending.'))
    results,modules=H.search(domain.mesh,contacts,directions['contacts'],floor['report']['ground_footprints'],depth,work,on_result=retain_one)
    installation=L.check_assembly(domain.mesh,modules,work) if modules else None
    subset,order=H.compatible_subset(installation,len(modules)) if modules else ([],[])
    subset_ids=[modules[i]['plan']['candidate_id'] for i in subset]
    order_ids=[modules[i]['plan']['candidate_id'] for i in order]
    all_geometry=bool(contacts) and len(modules)==len(contacts) and bool(installation and installation['passed'])
    floor_verified=bool(floor['report']['continuous_domain_coverage_proved'])
    artifacts={};records=[]
    by_id={r['candidate_id']:r for r in results}
    for index,module in enumerate(modules):
        plan=module['plan'];cid=plan['candidate_id'];j=by_contact[cid]
        destination=out/'supports'/cid;destination.mkdir(parents=True,exist_ok=True)
        length=installation['lengths_m'][index]
        trajectory=dict(candidate_id=cid,direction=plan['direction'].tolist(),bearing_deg=plan['bearing_deg'],
            start_translation_m=(-length*plan['direction']).tolist(),end_translation_m=[0.,0.,0.],length_m=length,
            motion='support(t)=support_final-(1-t)*L*a, 0<=t<=1; fixed orientation',
            individual_insertion_verified=True,complete_assembly_verified=all_geometry,
            included_in_assemblable_subset=cid in subset_ids,
            installation_step=order_ids.index(cid)+1 if cid in order_ids else None,
            certified_head_directions=directions['contacts'][j]['certified_directions'])
        I.save(destination/'trajectory.json',trajectory)
        entry=dict(candidate_id=cid,folder=str(destination.relative_to(out)),ground_color=COLORS[j%len(COLORS)],
            ground_corners_m=plan['ground_corners_m'].tolist(),ground_area_m2=plan['ground_area_m2'],
            ground_polygons_xz_m=[p.tolist() for p in plan['ground_polygons_xz_m']],ground_height_m=plan['ground_height_m'],
            anchor_xz_m=plan['anchor_xz_m'].tolist(),bearing_deg=plan['bearing_deg'],direction=plan['direction'].tolist(),
            solid=module['solid'],construction=plan['construction'],routes=plan['routes'],
            backing_depth_factor=plan['backing_depth_factor'],trajectory=trajectory,
            individual_geometry_verified=True,footprint_unchanged=True)
        records.append(entry)
        by_id[cid]['folder']=entry['folder']
        for filename in ('geometry.npz','support.stl','trajectory.json'):
            artifacts[str((destination/filename).relative_to(out))]=sha256(destination/filename)
    for record in results:
        destination=out/'supports'/record['candidate_id'];destination.mkdir(parents=True,exist_ok=True)
        record['included_in_assemblable_subset']=record['candidate_id'] in subset_ids
        I.save(destination/'status.json',record)
        artifacts[str((destination/'status.json').relative_to(out))]=sha256(destination/'status.json')
    verified=all_geometry and floor_verified and bool(schedule['continuous_coverage_proved'])
    status=('fixed_supports_and_insertions_verified' if verified else
            'all_connections_verified_bearing_not_certified' if all_geometry else
            'partial_individual_supports_verified' if modules else 'no_individual_connection_verified')
    result=dict(object=name,pose=pose_name(),stage=STAGE,schema=SCHEMA,complete=True,passed=verified,status=status,
        selected_ids=[c['candidate_id'] for c in contacts],contact_count=len(contacts),
        support_count=len(modules),saved_geometry_count=len(modules),supports=records,support_results=results,
        geometric_assembly_verified=all_geometry,installation=installation,
        assemblable_subset_ids=subset_ids,assemblable_subset_order=order_ids,
        assemblable_subset_verified=bool(subset_ids),subset_load_coverage_claimed=False,
        foot_positions_frozen=True,footprint_source_sha256=sha256(out.parent/F.STAGE/'floor_contact.json'),
        work_volume_record='../step2_local_support/work_volume.json',
        final_work_volume_clearance_verified=all_geometry,
        contact_force_coverage_proved_by_step3=bool(schedule['continuous_coverage_proved']),
        step4_bearing_certificate_verified=floor_verified,
        independent_support_equilibrium_verified=verified,physical_supports_verified=False,
        scope='Fixed Step 4 pads; each support constructed and swept independently. Partial successes are retained. Pair collisions and assembly order are checked separately. Structural strength and compliance are not certified.',
        search_settings=dict(head_depth_factors=list(H.DEPTH_FACTORS),angular_step_deg=15.,
            candidate_order='direct_and_bent_across_all_angles_then_spatial_search',
            member_width_fraction=.022,route_edge_budget=800,feet_may_move=False,common_ground_ring=False),
        elapsed_seconds=time.monotonic()-started,artifacts=artifacts,
        provenance=dict(inputs=input_hashes,code={**D.code_hashes(),**I.hashes([Path(__file__),Path(H.__file__),Path(L.__file__),Path(S.__file__),Path(M.__file__),Path(G.__file__),Path(W.__file__),Path(T.__file__),Path(U.__file__),Path(V.__file__)])}))
    I.check_hashes(input_hashes)
    I.save(out/'connection.json',result)
    I.save(out/'status.json',dict(complete=True,status=status,connection_sha256=sha256(out/'connection.json')))
    I.save(out/'progress.json',dict(complete=True,processed_count=len(results),contact_count=len(contacts),
        connection_sha256=sha256(out/'connection.json')))
    print(name,pose_name(),'Step 5:',status,len(modules),'/',len(contacts),'individual supports',flush=True)
    return result


# Current one-body entry; earlier independent-body helpers remain for regressions.
from step5_connect_support.belt_assembly import build, SCHEMA


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    parser.add_argument('--edge-budget', type=int, default=2000)
    args=parser.parse_args()
    for name in args.objects or OBJECTS:build(name, edge_budget=args.edge_budget)
