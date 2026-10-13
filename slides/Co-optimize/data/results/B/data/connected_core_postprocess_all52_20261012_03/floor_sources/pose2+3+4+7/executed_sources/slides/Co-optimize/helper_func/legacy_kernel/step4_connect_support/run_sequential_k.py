"""Consume sequential K-pose contacts, register unique heads and build bodies."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step3_scheculer.run_sequential_k import folder
from step2_local_support import geometry as G, withdrawal as W
from step0_pose_selection.floor_points import pressure_centers
from step0_pose_selection.select_poses import accepted_tasks
from step4_connect_support import head_registration as H, build_local_bodies as L
from step4_connect_support.fixture_view import cells_for, local_to_world
from step4_connect_support.run_local_bodies import clip_floor, plain


def read_case(source):
    source = Path(source).resolve()
    summary = I.check_report(source/'schedule.json')
    poses, name = summary['poses'], summary['object']
    group = source.parents[2]
    selected = source/f'particle_{summary["winner"]:03d}'
    variant = source.parent.name
    result = json.loads((selected/'schedule.json').read_text())
    if result != summary['result']:
        raise ValueError('Selected particle and top-level result differ')
    for name_,digest in result['artifacts'].items():
        if I.sha256(selected/name_) != digest:
            raise ValueError('Step3 contact artifact changed')
    tasks = accepted_tasks(name, poses)
    groups = [I.read_contacts(selected/f'contacts_{p}.npz') for p in poses]
    frames = [np.asarray(t.domain.data['frame']['T_world_mesh']) for t in tasks]
    geometry = folder(name,poses,'step2_local_support',variant=variant)
    meta = {h['id']:h for h in summary['heads']}
    owners, paths, catalogues = {}, [source/'schedule.json',selected/'schedule.json'], []
    for k,p in enumerate(poses):
        path=geometry/f'candidates_{p}.npz'
        for contact in I.read_contacts(path):
            owners[contact['candidate_id']] = k,contact
        description=geometry/f'candidates_{p}.json'
        catalogues.append(np.asarray(json.loads(description.read_text())['direction_catalogue']['vectors']))
        paths.extend([path,description,selected/f'contacts_{p}.npz'])
    own_cells={}
    for ident,m in meta.items():
        k,contact=owners[ident]
        offsets=G.vertex_offsets(tasks[k].domain.mesh,m['normal_depth_m'])[0]
        own_cells[ident]=(k,cells_for(contact,tasks[k].domain,offsets))
    heads=[]
    for k,contacts in enumerate(groups):
        row=[]
        for contact in contacts:
            owner,pieces=own_cells[contact['candidate_id']]
            transform=frames[k]@np.linalg.inv(frames[owner])
            row.append([p@transform[:3,:3].T+transform[:3,3] for p in pieces])
        heads.append(row)
    demands=[]
    for t in tasks:
        path=group/'step0_pose_selection/data'/f'floor_contact_{t.pose}.npz'
        with np.load(path) as z:
            np.testing.assert_allclose(z['load_wrenches'],t.targets/t.scale,atol=1e-13,rtol=0)
            xy=z['floor_demands_xy_m'].copy()
        np.testing.assert_allclose(xy,pressure_centers(t.targets/t.scale,t.domain.com)[0],atol=1e-13,rtol=0)
        if len(xy)!=32768:
            raise ValueError('Original sample count changed')
        demands.append(xy); paths.extend([path]+list(t.inputs))
    return SimpleNamespace(name=name,poses=poses,pair=group,tasks=tasks,groups=groups,heads=heads,
        source=selected,summary_source=source,schedule=result,summary=summary,paths=paths,
        demands=demands,catalogues=catalogues)


def foot_menu(case,bases,offsets):
    for margin in (.025,.065,.11):
        feet=[]
        for k,xy in enumerate(case.demands):
            polygon=MultiPoint(xy).convex_hull.buffer(margin,join_style=2)
            rim=np.asarray(polygon.exterior.coords)[:-1]
            coefficients=[]
            for j in range(len(case.poses)):
                if j==k: continue
                normal=bases[j][2]
                co=np.r_[(bases[k]@normal)[:2],(offsets[k]-offsets[j])@normal]
                coefficients.append(co); rim=clip_floor(rim,co)
                if not len(rim):break
            pads=[]
            for point in rim:
                pad=point+np.array([[-1.,-1],[1,-1],[1,1],[-1,1]])*.004
                for co in coefficients:
                    pad=clip_floor(pad,co)
                    if not len(pad):break
                if len(pad)>=3 and MultiPoint(pad).convex_hull.area>1e-6:
                    pads.append(pad.tolist())
            feet.append(pads)
        if all(feet):yield feet


def directions(case,registered,bases,offsets):
    menus=[]
    local=[v for h in registered for v in h.cells]
    for k,task in enumerate(case.tasks):
        analyzer=W.Analyzer(task.domain.mesh,.01*task.domain.mesh.extents.max(),
                            dict(vectors=case.catalogues[k].tolist()))
        cells=[SimpleNamespace(vertices=local_to_world(v,bases[k],offsets[k])) for v in local]
        valid=[i for i in case.schedule['geometry']['per_pose'][k]['common_direction_ids']
               if analyzer.test(cells,case.catalogues[k][i])['clear']]
        menus.append(valid)
    for ids in itertools.islice(itertools.product(*menus),8):
        yield dict(bases=bases,offsets=offsets,direction_ids=list(ids),
                   directions=np.asarray([case.catalogues[k][i] for k,i in enumerate(ids)]))


def draw(case,report,out,mesh=None,bases=None,offsets=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    columns=len(case.poses)
    fig=plt.figure(figsize=(5*columns,6),facecolor='white')
    for k,task in enumerate(case.tasks):
        ax=fig.add_subplot(1,columns,k+1,projection='3d')
        obj=task.domain.mesh
        ax.add_collection3d(Poly3DCollection(obj.triangles,facecolors='#c6c9c9',alpha=.18,edgecolors='none'))
        if mesh is not None:
            world=local_to_world(mesh.vertices,bases[k],offsets[k])
            ax.add_collection3d(Poly3DCollection(world[mesh.faces],facecolors='#c9cdcb',edgecolors='none'))
        else:world=obj.vertices
        for contact in case.groups[k]:
            index=case.schedule['selected_ids'].index(contact['candidate_id'])
            ax.add_collection3d(Poly3DCollection(contact['triangles_m'],facecolors=plt.get_cmap('tab20')(index%20),edgecolors='none'))
        points=np.vstack([obj.vertices,world]); center=(points.min(0)+points.max(0))/2
        radius=np.ptp(points,axis=0).max()*.6
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect([1,1,1]); ax.view_init(elev=20,azim=-55); ax.set_axis_off()
        ax.set_title(f'{task.pose}: {len(case.groups[k])} active heads\n'
                     f'{case.schedule["covered_counts"][k]:,}/32,768 Step3 samples')
    label='Constructed shared fixture' if mesh is not None else 'NO FIXTURE CONSTRUCTED — selected contacts only'
    fig.suptitle(f'{case.pair.name} | {label}\n{report["status"]}',fontsize=14)
    if report.get('floor_compatibility') and not report['floor_compatibility']['passed']:
        worst=max(report['floor_compatibility']['per_pose'],key=lambda r:r['violating_sample_count'])
        text=f'Ground conflict: {worst["pose"]} vs {worst["other_pose"]}: {worst["violating_sample_count"]:,}/32,768 original demands'
    else:text='Bodies and feet are absent.' if mesh is None else 'One connected solid; final full audit remains disabled.'
    fig.text(.5,.025,text,ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.87)); fig.savefig(out/'overview.png',dpi=140); plt.close(fig)


def run(source):
    case=read_case(source)
    out=case.pair/'step4'; data=out/'data'; data.mkdir(parents=True,exist_ok=True)
    report=dict(schema='sequential_k_step4_v2',object=case.name,poses=case.poses,
        complete=False,attempt_complete=False,constructed=False,passed=False,
        step3_passed=case.schedule['passed'],step3_heads=case.schedule['heads'],
        global_head_exclusions=case.summary.get('global_head_exclusions',False),
        covered_counts=case.schedule['covered_counts'],active_ids_by_pose=case.schedule['active_ids_by_pose'],
        source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
        extra_loads_added=False,verification=dict(performed=False,independent_audit_performed=False),
        complete_fixture_verified=False,attempts=[],
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes([Path(__file__),Path(H.__file__),Path(L.__file__)])))
    mesh=bases=offsets=None
    if not case.schedule['passed']:
        report['status']='step3_incomplete'
    else:
        bases,offsets=H.fixed_placements(case.tasks)
        registered,registration=H.register(case.groups,case.heads,bases,offsets,general_layout=True)
        floor=H.floor_compatibility(case.tasks,case.demands,bases,offsets)
        report.update(registration=registration,floor_compatibility=floor,
                      placement=plain(dict(bases=bases,offsets=offsets)))
        if not floor['passed']:
            report['status']='fixed_shared_registration_floor_conflict'
        elif not registration['all_heads_above_all_floors']:
            report['status']='inactive_head_crosses_floor'
        else:
            report['status']='no_compatible_whole_head_withdrawal'
            for placement in directions(case,registered,bases,offsets):
                for feet in foot_menu(case,bases,offsets):
                    attempt=dict(direction_ids=placement['direction_ids'],feet=feet)
                    with tempfile.TemporaryDirectory(prefix='cadgrasp_sequential_k_body_') as tmp:
                        stage=Path(tmp)
                        try:
                            L.build(stage,feet,case=case,placement=placement,general_layout=True,
                                skip_unreachable=True,floor_policy='nearest',verify=False,cache_dir=data/'cache')
                        except RuntimeError as error:
                            attempt.update(constructed=False,error=str(error)); report['attempts'].append(attempt)
                            report['status']='body_construction_failed'
                            continue
                        attempt['constructed']=True; report['attempts'].append(attempt)
                        built=json.loads((stage/'report.json').read_text())
                        mesh=trimesh.load(stage/'fixture.obj',force='mesh',process=False)
                        for path in stage.iterdir():
                            shutil.copyfile(path,out/'shape.obj' if path.name=='fixture.obj' else data/path.name)
                        report.update(complete=True,constructed=True,passed=None,
                            status='connected_geometry_constructed_without_final_audit',
                            volume_cm3=built['volume_cm3'],solid=built['solid'],construction=built,
                            provenance=dict(inputs=I.hashes(case.paths),code=built['provenance']['code']))
                        report['provenance']['code'].update(I.hashes([Path(__file__)]))
                    break
                if mesh is not None:break
    report['attempt_complete']=True
    if mesh is None and (out/'shape.obj').exists():
        history=data/'history'; history.mkdir(exist_ok=True)
        shutil.move(out/'shape.obj',history/('previous_'+I.sha256(out/'shape.obj')[:12]+'.obj'))
    I.save(data/'report.json',report)
    draw(case,report,out,mesh,bases,offsets)
    report['artifacts']={'../overview.png':I.sha256(out/'overview.png')}
    if mesh is not None:report['artifacts']['../shape.obj']=I.sha256(out/'shape.obj')
    I.save(data/'report.json',report)
    print('STEP4 COMPLETE',case.poses,report['status'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    run(parser.parse_args().source)
