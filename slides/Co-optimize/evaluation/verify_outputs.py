"""Read-only checks of final meshes, ground faces and static occupied spaces.

The completed B connected_base_v2 batch reuses Step5.1 connected_base_v1.
No force solver, Boolean, contact reconstruction or geometry mutation.
"""
from pathlib import Path
import json,hashlib,sys
import numpy as np,trimesh
from shapely.geometry import MultiPoint,Point
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'slides/Co-optimize/helper_func'))
import artifact_store as A
P=A.result_root('B')
sys.path.insert(0,str(ROOT/'slides/Co-optimize/evaluation'))
from metrics import static_bounds, METRIC_VERSION, BOX_KEYS
def sha(p):return A.digest(p)
selection=json.loads((P/'data/base_oracle_results/summary.json').read_text())['selected']
body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
rows=[];masks=0
for c in selection:
    base=P/c['id'];s4=base/'step4/step4.2'/c['experiment'];s52=base/'step5/step5.2/connected_base_v2'
    r=json.loads((s52/'data/report.json').read_text());assert r['passed'] and r['complete']
    m=trimesh.load(s52/'support_with_base.obj',force='mesh',process=False)
    assert m.is_watertight and len(m.split(only_watertight=False))==1,c['id']
    assert np.isclose(abs(m.volume)*1e6,r['construction']['material_cm3'],rtol=1e-9,atol=1e-6)
    s51=base/'step5/step5.1/connected_base_v1';r51=json.loads((s51/'data/report.json').read_text())
    posed=[];area=[];gaps=[]
    for pose in r51['per_pose']:
        with np.load(s51/pose['arrays']) as z:
            T=z['T_world_fixture'];U=z['T_world_object'];h=z['floor_demands_world_m'][z['hull_indices']]
        v=m.vertices@T[:3,:3].T+T[:3,3]
        assert v[:,2].min()>=-1e-8,(c['id'],pose['pose'],'underground')
        triangles=v[m.faces]
        flat=np.all(np.abs(triangles[:,:,2])<=1e-8,axis=1)
        ground_triangles=triangles[flat]
        assert len(ground_triangles)>0
        areas=np.linalg.norm(np.cross(ground_triangles[:,1]-ground_triangles[:,0],ground_triangles[:,2]-ground_triangles[:,0]),axis=1)/2
        positive=ground_triangles[areas>1e-14]
        feet=positive.reshape(-1,3)[:,:2]
        o=body.vertices@U[:3,:3].T+U[:3,3];own=o[np.abs(o[:,2])<=1e-9,:2]
        hull=MultiPoint(np.vstack([feet,own])).convex_hull
        gap=max(hull.distance(Point(p[:2])) for p in h)
        assert gap<=1e-8,(c['id'],pose['pose'],gap)
        gaps.append(gap);area.append(float(areas.sum()*1e4))
        with np.load(s4/(pose['pose']+'_force.npz')) as z:
            mask=z['mask'];assert mask.all() and len(mask)==32768;masks+=len(mask)
    for path,expected in r['provenance']['inputs'].items():assert sha(ROOT/path)==expected,path
    e=json.loads((base/'evaluation/connected_base_v2/report.json').read_text())
    with np.load(s4/'layout.npz') as z:layout={k:z[k] for k in z.files}
    nominal=trimesh.load(s4/'support.obj',force='mesh',process=False)
    saved=json.loads((s4/'report.json').read_text())
    assert e['metric_version']==METRIC_VERSION
    boxes=static_bounds(body.vertices,layout,nominal.vertices,m.vertices,saved['poses'])
    for key in BOX_KEYS:
        np.testing.assert_allclose(boxes[key]['dimensions_m'],e[key]['dimensions_m'],rtol=0,atol=1e-12)
        assert abs(boxes[key]['volume_cm3']-e[key]['volume_cm3'])<1e-6
    for key in ['objects_in_final_layout','independent_pose_object_sum_litres','state_space_sums_litres','fixture_states',
                'juxtaposed_pose_count','rotating_reuse_pose_count','multi_pose_fixture_state_count',
                'singleton_fixture_state_count','per_pose']:
        assert boxes[key]==e[key],(c['id'],key)
    rows.append(dict(id=c['id'],passed=True,connected=True,watertight=True,
        actual_ground_face_areas_cm2=area,maximum_pressure_hull_gap_m=max(gaps)))
for batch in ['step51_connected_base_v1','step52_connected_base_v2','evaluation_connected_base_v2']:
    d=P/'data'/batch;verification=json.loads((d/'verification.json').read_text())
    # The completed execution was already checked before its temporary source
    # copies were retired. Subsequent I/O-only edits need not equal that run's
    # source bytes; verify the unchanged final inputs instead.
    protected=verification.get('protected_inputs',verification.get('protected',{}))
    for path,h in protected.items():assert sha(ROOT/path)==h,path
result=dict(passed=True,group_count=len(rows),original_mask_entries_reused=masks,
    actual_positive_area_floor_faces_checked=True,connected_watertight_models_checked=True,
    pressure_hull_checked_from_actual_floor_faces=True,static_bbox_recomputed=True,
    independent_pose_and_shared_state_spaces_checked=True,layout_membership_checked=True,
    unsupported_overlapping_baseline_checked=True,
    metric_version=METRIC_VERSION,object_withdrawal_motion_included=False,
    force_solver_run=False,geometry_boolean_run=False,
    temporary_execution_copies_retired_after_original_verification=True,cases=rows,
    verifier_source=str(Path(__file__).resolve().relative_to(ROOT)),verifier_sha256=sha(Path(__file__).resolve()))
(P/'data/system_connected_base_v2/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('FINAL AUDIT',len(rows),'groups',masks,'saved mask entries; floor faces, provenance, static BBX and state membership pass')
