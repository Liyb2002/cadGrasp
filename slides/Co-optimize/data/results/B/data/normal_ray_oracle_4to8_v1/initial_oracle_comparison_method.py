"""Initial-layout geometry-only Oracle comparison; no contact/force solve."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import sys, json
import numpy as np
import trimesh
ROOT=Path.cwd();COOPT=ROOT/'slides/Co-optimize'
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'step4.2')]
import _bootstrap
from base_oracle import BaseOracle
from whole_pipeline import load_layout

base=COOPT/'data/results/B';folder=base/'data/normal_ray_oracle_4to8_v1'
batch=json.loads((folder/'batch.json').read_text())
mesh=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
cache={};rows=[]
for label in batch['sets']:
    report=json.loads((base/label/'step4/step4.1/data/report.json').read_text())
    poses=report['poses'];tasks=[];native=[];works=[]
    for pose in poses:
        if pose not in cache:
            with np.load(ROOT/'objects/B/poses'/pose/'setup.npz') as z:
                cache[pose]=(z['T_world_mesh'].copy(),z['work_faces'].copy(),float(z['cone_half_deg']))
        T,ids,angle=cache[pose]
        if ids.dtype==bool:ids=np.flatnonzero(ids)
        native.append(T);works.append(SimpleNamespace(face_ids=ids,half_angle_deg=angle))
        tasks.append(SimpleNamespace(domain=SimpleNamespace(data={'load':{'cone_half_deg':angle}})))
    model=SimpleNamespace(mesh=mesh,extent=float(mesh.extents.max()),native=np.array(native),poses=poses,
        tasks=tasks,work_rays=works,inputs=[],name='B',length=max(.5,4*float(mesh.extents.max())),
        support_floor_clearance_m=0.)
    layout=load_layout(base/label/'step4/step4.1/layout.npz')
    answers={}
    for mode in ['angular','normal']:
        answer=BaseOracle(model,work_mode=mode).query(layout)
        answers[mode]={k:answer.get(k) for k in ['status','enclosure_passed','gap_m','work_only_enclosure_passed','query_seconds']}
    row=dict(id=label,pose_count=len(poses),answers=answers,
        initialized_layout_changed=False,force_solver_run=False,connection_query_run=False)
    rows.append(row)
    print(label,answers['angular']['status'],'->',answers['normal']['status'],flush=True)
result=dict(scope='same saved Step4.1 geometry; XYZ offsets zero and floor clearance zero in both modes',
    group_count=len(rows),cases=rows,force_solver_run=False,connection_query_run=False,
    angular_status_histogram=dict(Counter(r['answers']['angular']['status'] for r in rows)),
    normal_status_histogram=dict(Counter(r['answers']['normal']['status'] for r in rows)),
    floor_certificates_unchanged=sum(r['answers']['angular']['status']=='certified_floor_incompatible' and
        r['answers']['normal']['status']=='certified_floor_incompatible' for r in rows),
    enclosure_improved_count=sum(not r['answers']['angular']['enclosure_passed'] and
        r['answers']['normal']['enclosure_passed'] for r in rows))
(folder/'initial_oracle_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(folder/'initial_oracle_comparison_method.py').write_text(Path(__file__).read_text())
print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)
