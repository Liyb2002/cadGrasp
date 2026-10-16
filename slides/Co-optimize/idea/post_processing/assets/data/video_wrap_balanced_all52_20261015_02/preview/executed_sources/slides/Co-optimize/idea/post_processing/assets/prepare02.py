"""Load actual accepted Step5 snapshots and the exact published final OBJ.

No geometry is constructed or changed. Connected construction and final
reinforcement remain separate from accepted deletion snapshots.
"""
from pathlib import Path
import json
import shutil
import render01 as R
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
DATA=HERE/'data/video02'
RUN=R.BASE/'step5/feasible/outside_in_all52_20261015_01'
FINAL=R.BASE/'step5/step5.3/outside_in_all52_20261015_03'
WORKER=R.BASE/'step5/feasible/outside_in_all52_20261015_03/connected_width'


def load_rows():
    first=json.loads((HERE/'data/manifest.json').read_text())
    report=json.loads((R.FLOOR/'data/report.json').read_text())
    rows=[]
    for record in report['per_pose']:
        with np.load(R.FLOOR/record['arrays'])as saved:
            cloud=saved['floor_demands_world_m'].copy();ids=saved['hull_indices'].copy()
            T=saved['T_world_fixture'].copy();U=saved['T_world_object'].copy()
        local=R.transform(cloud,np.linalg.inv(T))
        shown=np.unique(np.r_[np.linspace(0,len(cloud)-1,192,dtype=int),ids])
        rows.append(dict(pose=record['pose'],color=record['color'],T=T,U=U,
            hull_world=cloud[ids],hull=local[ids],dots=local[shown]))
    if [r['pose']for r in rows]!=R.POSES:raise ValueError('Wrong saved five-pose set')
    return first,rows


def bounds(points):return np.array([points.min(0),points.max(0)])


def box_l(box):return float(np.prod(box[1]-box[0])*1000)


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    _,rows=load_rows();ref=rows[0]['T']
    mesh_path=R.ROOT/'objects/B/mesh.stl'
    body=trimesh.load(mesh_path,force='mesh',process=False)
    objects=np.vstack([R.transform(body.vertices,r['U'])for r in rows])
    protected={R.relative(p):R.digest(p)for p in [mesh_path,R.MOVIE,R.REFERENCE,R.FLOOR/'data/report.json',
        R.STEP4/'layout.npz',FINAL/'data/report.json',WORKER/'report.json']}
    for row in rows:
        path=R.FLOOR/'data'/f"{row['pose']}.npz"
        protected[R.relative(path)]=R.digest(path)
    stages=[]
    sources=[('input',RUN/'baseline/support.obj',RUN/'baseline/report.json','Complete input')]
    for path in sorted((RUN/'outside_in/checkpoints').glob('*/support.obj')):
        kind='construction'if path.parent.name=='0001'else'pruning'
        sources.append((kind,path,path.with_name('report.json'),path.parent.name))
    sources.append(('final',FINAL/'support_with_base.obj',WORKER/'report.json','Step5 final'))
    if len(sources)!=21:raise ValueError('Expected baseline, 19 snapshots and actual final')
    for index,(kind,path,report_path,label)in enumerate(sources):
        report=json.loads(report_path.read_text());expected=report['support_sha256']
        if not report['passed']or R.digest(path)!=expected:
            raise ValueError('Saved accepted checkpoint mismatch: '+str(path))
        certificate=path.with_name('force_certificate.npz')
        if kind=='final':certificate=FINAL/'force_certificate.npz'
        with np.load(certificate)as saved:
            if len(saved.files)!=5 or any(saved[k].shape!=(32768,)or not saved[k].all()for k in saved.files):
                raise ValueError('Incomplete saved full-demand certificate')
        for p in [path,report_path,certificate]:protected[R.relative(p)]=R.digest(p)
        mesh=trimesh.load(path,force='mesh',process=False)
        if not mesh.is_watertight:raise ValueError('Saved geometry is not closed')
        shown_bounds=bounds(R.transform(mesh.vertices,ref))
        system=bounds(np.vstack([objects,*[R.transform(mesh.vertices,r['T'])for r in rows]]))
        stage=dict(index=index,kind=kind,label=label,support=R.relative(path),support_sha256=expected,
            source_report=R.relative(report_path),saved_force_certificate=R.relative(certificate),
            saved_demands_passed=5*32768,material_cm3=abs(float(mesh.volume))*1e6,
            shown_bounds_world_m=shown_bounds.tolist(),shown_bbx_l=box_l(shown_bounds),
            system_bounds_world_m=system.tolist(),system_bbx_l=box_l(system),
            geometry_is_actual_saved_obj=True,geometry_edited=False)
        if kind=='pruning':
            previous=stages[-1]
            if stage['material_cm3']>previous['material_cm3']+1e-6:raise ValueError('Pruning added material')
            if stage['shown_bbx_l']>previous['shown_bbx_l']+1e-8:raise ValueError('Pruning BBX increased')
            stage['shown_box_changed']=abs(stage['shown_bbx_l']-previous['shown_bbx_l'])>1e-8
            stage['system_box_changed']=abs(stage['system_bbx_l']-previous['system_bbx_l'])>1e-8
        stages.append(stage)
    public=json.loads((FINAL/'data/report.json').read_text())
    if public['artifacts']['support_with_base.obj']!=stages[-1]['support_sha256']:
        raise ValueError('Final frame is not the actual published Step5 model')
    shutil.copy2(FINAL/'support_with_base.obj',DATA/'actual_step5_final.obj')
    if R.digest(DATA/'actual_step5_final.obj')!=stages[-1]['support_sha256']:
        raise ValueError('Final model copy changed')
    worker=json.loads((WORKER/'report.json').read_text())
    if abs(worker['checks']['bbx_l']-stages[-1]['system_bbx_l'])>1e-8:
        raise ValueError('System BBX disagrees with saved final certificate')
    record=dict(complete=True,pose_set=R.CASE,stages=stages,protected_inputs=protected,
        reference_fixture_transform=ref.tolist(),final_obj_sha256=stages[-1]['support_sha256'],
        final_obj_copy=R.relative(DATA/'actual_step5_final.obj'),
        all_geometry_from_actual_saved_step5=True,illustrative_ring_geometry_used=False,
        original_numerical_artifacts_modified=False,force_solver_rerun=False,
        original_accepted_snapshot_order_preserved=True,
        reconstruction_and_final_reinforcement_distinguished_from_deletion=True,
        shown_bbx_definition='XYZ AABB of support at the fixed first absolute installation',
        system_bbx_definition='XYZ AABB of all actual object and support installations in original world coordinates',
        actual_system_bbx_reduction_percent=(1-stages[-1]['system_bbx_l']/stages[0]['system_bbx_l'])*100,
        pruning_shown_bbx_reduction_percent=(1-stages[-1]['shown_bbx_l']/stages[1]['shown_bbx_l'])*100,
        actual_material_reduction_percent=(1-stages[-1]['material_cm3']/stages[0]['material_cm3'])*100)
    for p,h in protected.items():
        if R.digest(R.ROOT/p)!=h:raise ValueError('Saved source changed')
    R.save(DATA/'stages.json',record)
    print(json.dumps(dict(actual_models=len(stages),final_obj_sha256=stages[-1]['support_sha256'],
        material_cm3=[stages[0]['material_cm3'],stages[-1]['material_cm3']],
        system_bbx_l=[stages[0]['system_bbx_l'],stages[-1]['system_bbx_l']],
        system_reduction_percent=record['actual_system_bbx_reduction_percent'])),flush=True)
    return record


if __name__=='__main__':prepare()
