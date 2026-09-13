"""Build one connected rigid support from every selected head and a shared base.

Earlier independent-support helpers remain below for historical callers; the
current build/schema are imported from belt_assembly before main."""
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
        ground_polygon_offsets=np.cumsum([0]+list(map(len,plan['ground_polygons_xy_m']))))
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


# Current one-body entry; shared geometry and regression helpers stay available.
from step5_connect_support.belt_assembly import build, SCHEMA


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    parser.add_argument('--edge-budget', type=int, default=2000)
    args=parser.parse_args()
    for name in args.objects or OBJECTS:build(name, edge_budget=args.edge_budget)
