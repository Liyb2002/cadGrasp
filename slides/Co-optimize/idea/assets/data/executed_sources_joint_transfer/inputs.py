"""Read the user's accepted thick-foot solids, preserving their saved layouts."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
COOPT=HERE.parent
ROOT=COOPT.parents[1]
SYSTEM=COOPT/'data/results/B/data/system_stocky_legs_all52_20261013_02'
sys.path[:0]=[str(COOPT/'helper_func'),str(ROOT)]
import stage5_shared as C

SETS=['pose1+2+3+4+5+6+7+8','pose2+20','pose1+2+3+4+5+6']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def transform(mesh,T):
    m=mesh.copy();m.apply_transform(T);return m

def load(label,selection=None,yaw_offset_deg=0.):
    if selection is None:
        selection_path=SYSTEM/'selection.json'
        case=next(r for r in json.loads(selection_path.read_text())['results'] if r['id']==label)
        evaluation=next(r for r in json.loads((SYSTEM/'evaluation_results.json').read_text())['cases'] if r['id']==label)
        folder=ROOT/evaluation['step53_source'];record=json.loads((folder/'data/report.json').read_text())
    else:
        selection_path=Path(selection).resolve()
        manifest=json.loads(selection_path.read_text())
        case=next(r for r in manifest['results'] if r['id']==label)
        if not case['passed']:raise ValueError('Video requires a certified complete support: '+label)
        path=case.get('output')or case.get('construction',{}).get('output')
        folder=ROOT/path;record=json.loads((folder/'data/report.json').read_text())
        evaluation=json.loads((folder/'evaluation.json').read_text())if(folder/'evaluation.json').exists()else{}
        evaluation.update(step53_source=str(folder.relative_to(ROOT)),step51_source=record['step51_source'])
        if not record['passed']or record['selected_step4_experiment']!=case['experiment']:
            raise ValueError('Video selection and saved layout differ: '+label)
        for name,expected in record['artifacts'].items():
            if sha(folder/name)!=expected:raise ValueError('Selected video solid/certificate changed: '+name)
    reader=C.Inputs('B',selection_path,Path(evaluation['step51_source']).name)
    data=reader.load(case)
    support=C.trimesh.load(folder/record['support'],force='mesh',process=False)
    source=ROOT/record['selected_step4_source']
    with np.load(source/'layout.npz') as a:layout={k:a[k].copy() for k in a.files}
    poses=record.get('poses',data['record']['poses']);groups={}
    for i,host in enumerate(layout['hosts']):groups.setdefault(int(host),[]).append(i)
    groups=sorted(groups.items(),key=lambda p:(-len(p[1]),p[0]))
    order=[i for host,guests in groups for i in [host]+[g for g in guests if g!=host]]
    rows=[];frames={}
    # Rotate the entire installation about gravity for a camera-facing approach.
    # Objects, work patches, fixture and withdrawal vector rotate together;
    # the stored object/fixture relationship is kept exactly.
    for host,guests in groups:
        base=data['rows'][host]['fixture_transform'];d=base[:3,:3]@layout['directions'][host]
        yaw=np.pi/2-np.arctan2(d[1],d[0])+np.radians(yaw_offset_deg)
        station=np.eye(4);station[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();station[:3,3]=[.43,0.,.24]
        frames[host]=station@base
        for i in guests:
            row=data['rows'][i];U=station@row['object_transform'];T=station@row['fixture_transform']
            assert np.allclose(np.linalg.inv(T)@U,layout['placements'][i],atol=1e-10)
            assert np.allclose(T,frames[host],atol=1e-10)
            pose_folder=ROOT/'objects/B/poses'/poses[i]
            with np.load(pose_folder/'setup.npz') as a:
                work=a['work_faces'].copy();native_pose=a['T_world_mesh'].copy();force_limit=float(a['K'])
            workmesh=trimesh.Trimesh(data['body'].vertices.copy(),data['body'].faces[work],process=False)
            workmesh.remove_unreferenced_vertices()
            direction=T[:3,:3]@layout['directions'][i]
            sv=trimesh.transform_points(support.vertices,T);ov=trimesh.transform_points(data['body'].vertices,U)
            travel=max(.10,float((sv@direction).max()-(ov@direction).min()+.022))
            rows.append(dict(index=i,pose=poses[i],host=host,role='reuse' if i==host else 'juxtapose',
                object=U,fixture=T,direction=direction,travel=travel,work_ids=work,work_mesh=workmesh,yaw=yaw,
                native_pose=native_pose,force_limit_mg=force_limit,load_samples=pose_folder/'angle_30/samples.npz'))
    rows={r['index']:r for r in rows}
    protected={str(ROOT/'objects/B/mesh.stl'):sha(ROOT/'objects/B/mesh.stl'),
        str(folder/record['support']):sha(folder/record['support']),str(source/'layout.npz'):sha(source/'layout.npz')}
    protected.update({str(ROOT/p):h for p,h in record['provenance']['inputs'].items()})
    protected.update({str(ROOT/p):h for p,h in reader.protected.items()})
    protected[str(selection_path)]=sha(selection_path)
    protected[str(folder/'data/report.json')]=sha(folder/'data/report.json')
    protected.update({str(folder/name):expected for name,expected in record['artifacts'].items()})
    for row in rows.values():protected.setdefault(str(row['load_samples']),sha(row['load_samples']))
    return dict(label=label,object=data['body'],fixture=support,rows=rows,order=order,frames=frames,
        record=record,protected=protected,evaluation=evaluation,fixture_source=str(folder.relative_to(ROOT)))
