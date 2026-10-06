"""Reversible path initialization: per-pose native +z, continuous cuts, diagnostics."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from scipy.spatial import ConvexHull
from pathlib import Path
import argparse,time

def initialize_direction(T_pose):
    canonical=T_pose[:3,:3].T@np.array([0.,0.,1.])
    return canonical,T_pose[:3,:3]@canonical

def prepare(name):
    folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'objects'/name/'pose_sets.json').read_text())['sets']
    illegal=ROOT/'objects'/name/'illegal_pose_sets.json'
    if illegal.exists():manifest += [dict(g,id='illegal/'+g['id']) for g in json.loads(illegal.read_text())['sets']]
    poses=sorted({p for group in manifest for p in group['poses']});states={p:state(name,p) for p in poses}
    maximum=0.;inputs=[]
    for group in manifest:
        path=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3/step3.3/support_with_rings.obj'
        support=trimesh.load(path,force='mesh',process=False);inputs.append(path)
        for pose in group['poses']:
            task,T,mesh=states[pose];direction,_=initialize_direction(T)
            maximum=max(maximum,float((support.vertices@direction).max()-(mesh.vertices@direction).min()))
    length=max(.5,maximum+.02);paths={};artifacts={}
    for pose,(task,T,mesh) in states.items():
        direction,native=initialize_direction(T);path=folder/pose;path.mkdir(exist_ok=True)
        for kind,distance in [('full',length),('display',.10)]:
            sweep=S.swept_solid(mesh,distance*direction);D.export_exact_obj(sweep,path/f'{kind}_sweep.obj')
            artifacts[f'{pose}/{kind}_sweep.obj']=I.sha256(path/f'{kind}_sweep.obj')
        paths[pose]=dict(direction_fixture=direction.tolist(),direction_world=native.tolist());inputs+=task.inputs
        print('PREPARED OWN +Z',pose,flush=True)
    record=dict(complete=True,initialization_kind='Each pose withdraws along its own native world +z',paths=paths,full_length_m=length,display_length_m=.10,initialization_is_arbitrary=True,paths_independent_variables=True,initial_canonical_sweeps_identical=False,provenance=provenance(inputs,[HERE/'step4.1/step41.py',HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts)
    save(folder/'initialization.json',record)
    for old in ['full_sweep.obj','display_sweep.obj']:(folder/old).unlink(missing_ok=True)
    return record

def ground_coverage(mesh,T,points):
    world=transform_points(mesh.vertices,T);xy=world[np.abs(world[:,2])<=1e-9,:2]
    if len(xy)<3:return dict(passed=False,covered=0,total=len(points),ground_vertex_count=len(xy))
    try:
        hull=ConvexHull(xy);mask=np.all(points@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9,axis=1)
        return dict(passed=bool(mask.all()),covered=int(mask.sum()),total=len(mask),ground_vertex_count=len(xy))
    except Exception as error:return dict(passed=False,covered=0,total=len(points),error=str(error))

from step41_legacy_render import render

def run(name,group,initialization,render_images=True):
    began=time.monotonic();base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    seed_path=base/'step3/step3.3/support_with_rings.obj';seed_mesh=trimesh.load(seed_path,force='mesh',process=False);seed=S.solid(seed_mesh)
    folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41';length=initialization['full_length_m']
    cached_full=[];display=[];directions=[];states=[]
    for pose in group['poses']:
        task,T,mesh=state(name,pose);states.append((task,T));directions.append(np.array(initialization['paths'][pose]['direction_fixture']))
        cached_full.append(S.solid(trimesh.load(folder/pose/'full_sweep.obj',force='mesh',process=False)))
        display.append(trimesh.load(folder/pose/'display_sweep.obj',force='mesh',process=False))
    policy=ExitClearance(mesh)
    with np.load(base/'step3/step3.2/data/contacts.npz') as z:allowed=z['allowed_faces']
    allowed=allowed[np.max(mesh.face_normals[allowed]@np.asarray(directions).T,axis=1)<=1e-9]
    chosen_fan=8;best=None
    for fan in [8,2,16,64]:
        individual=cached_full if fan==8 else [S.solid(S.swept_solid(mesh,length*d,fan_in=fan)) for d in directions]
        sweep_candidate=union(individual)
        construction=policy.construct(seed,individual,[policy.sweep(length*d,fan) for d in directions],allowed)
        candidate_remaining=construction['remaining'];candidate_removed=construction['removed']
        partition_error=abs(material_volume(seed)-material_volume(candidate_removed)-material_volume(candidate_remaining))
        overlap=max(material_volume(candidate_remaining^part) for part in individual)
        score=max(partition_error,overlap,construction['diagnostics']['padded_sweep_overlap_outside_contact_cores_m3'])
        if not construction['diagnostics']['contact_area_preserved']:score=max(score,1.)
        if best is None or score<best[0]:best=(score,fan,sweep_candidate,candidate_remaining,candidate_removed,partition_error,overlap,individual,construction)
        if score<1e-10:break
    _,chosen_fan,full,remaining,removed,_,residual_sweep_overlap,individual,construction=best
    removed_mesh=S.unpack(removed);remaining_mesh=S.unpack(remaining)
    raw_intersection_volume=material_volume(seed^full)
    D.export_exact_obj(S.unpack(full),out/'data/applied_sweep.obj')
    D.export_exact_obj(removed_mesh,out/'removed_support.obj');D.export_exact_obj(remaining_mesh,out/'remaining_support.obj')
    _,_,mesh=state(name,group['poses'][0]);triangles,sources=contact_boundary(mesh,remaining_mesh,allowed) if len(remaining_mesh.faces) else (np.empty((0,3,3)),np.empty(0,int))
    np.savez_compressed(out/'data/remaining_contacts.npz',triangles_mesh_m=triangles,source_faces=sources)
    rows=[];inputs=[seed_path,base/'step3/step3.3/data/report.json',base/'step3/step3.2/data/contacts.npz',folder/'initialization.json']
    for k,pose in enumerate(group['poses']):
        task,T=states[k];direction=directions[k];inputs+=task.inputs+[folder/pose/'full_sweep.obj',folder/pose/'display_sweep.obj']
        full_supply=supply(task,T,triangles,sources);error=None
        try:mask,info=J.classify(full_supply,task.targets)
        except (RuntimeError,AssertionError,np.linalg.LinAlgError) as e:mask=np.zeros(len(task.targets),bool);info={};error=str(e)
        floor_path=ROOT/'objects'/name/'poses'/pose/'floor_contact.npz';points=np.load(floor_path)['floor_demands_xy_m'];inputs.append(floor_path)
        coverage=ground_coverage(remaining_mesh,T,points)
        native=T[:3,:3]@direction;end=mesh.vertices+length*direction
        separated=bool((end@direction).min()>(seed_mesh.vertices@direction).max()+1e-9)
        row=dict(pose=pose,direction_fixture=direction.tolist(),direction_world=native.tolist(),path_kind='Straight translation along this pose native world +z, independent path variable',path_fixture_m=[[0.,0.,0.],(direction*length).tolist()],full_length_m=length,endpoint_completely_separated=separated,minimum_object_world_z_along_path_m=float(min(task.domain.mesh.vertices[:,2].min(),task.domain.mesh.vertices[:,2].min()+length*native[2])),own_removed_volume_cm3=material_volume(seed-(seed-individual[k]))*1e6,force_covered=int(mask.sum()),load_count=len(mask),force_passed=bool(mask.all()) if error is None else None,force_error=error,classifier=info,ground_coverage=coverage)
        rows.append(row);np.savez_compressed(out/'data'/f'{pose}.npz',force_mask=mask,supply_7d=full_supply,T_fixture_to_world=T,direction_world=native)
        print('INITIAL EXIT',group['id'],pose,row['force_covered'],flush=True)
    from step41_render import render as render_current
    if render_images:render_current(name,group)
    conserved=abs(material_volume(seed)-material_volume(removed)-material_volume(remaining))
    partition_tolerance=max(1e-10,material_volume(seed)*1e-5)
    partition_resolved=best[0]<1e-10 and conserved<partition_tolerance and residual_sweep_overlap<1e-10
    force_path_gate=partition_resolved and all(row['force_passed'] and row['minimum_object_world_z_along_path_m']>=-1e-9 for row in rows)
    gate=force_path_gate and all(row['ground_coverage']['passed'] for row in rows)
    report=dict(clearance_diagnostics=construction['diagnostics'],exit_clearance=policy.metadata,complete=True,stage='step4.1_initialize',pose_set=group['id'],initialization=initialization,state_results=rows,remaining_force_and_ground_gate=gate,force_and_path_initialization_gate=force_path_gate,ground_coverage_is_diagnostic_only=True,status='geometry_unresolved' if not partition_resolved else 'initialization_meets_force_and_path' if force_path_gate else 'needs_path_optimization',full_fixture_accepted=False,optimized=False,seed_volume_cm3=material_volume(seed)*1e6,removed_volume_cm3=material_volume(removed)*1e6,remaining_volume_cm3=material_volume(remaining)*1e6,volume_partition_error_m3=conserved,removed_geometry_policy='S0 minus computed remaining material; complement partition avoids unstable direct-intersection coplanar triangulation',direct_intersection_discrepancy_m3=abs(material_volume(removed)-raw_intersection_volume),remaining_sweep_overlap_m3=residual_sweep_overlap,volume_partition_tolerance_m3=partition_tolerance,geometry_partition_resolved=partition_resolved,sweep_boolean_fan_in=chosen_fan,remaining_component_count=len(remaining.decompose()),remaining_contact_triangle_count=len(triangles),remaining_contact_area_cm2=float(trimesh.triangles.area(triangles).sum())*1e4,cut_policy=policy.metadata['policy'],sweep_policy='Full continuous nonconvex object translation; no convex-hull approximation or sampled-frame cutting',initial_sweeps_identical=False,ground_model='Original object-floor reaction model retained for force diagnostic; actual remaining support ground coverage reported separately',validation_policy='One construction computation; no exported-model replay. Object exit stays above its own floor; installed support floor legality, connectivity and strength are not accepted here.',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step4.1/step41.py',HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py'),HERE/'helper_func/co_common.py',Path(J.__file__)]),artifacts={**{f'../{f}':I.sha256(out/f) for f in (['exit_directions.png','final_result.png','exit_sweeps.png'] if render_images else [])+['removed_support.obj','remaining_support.obj']},'remaining_contacts.npz':I.sha256(out/'data/remaining_contacts.npz'),'applied_sweep.obj':I.sha256(out/'data/applied_sweep.obj')})
    save(out/'data/report.json',report)
    lines=[f"# {group['id']}：Step4.1 退出初始化",'', '每个状态沿自己的原生世界 +z 竖直向上退出，分别变换到共同支撑坐标。这是思路二的初始化，没有优化。', '', '三张图片：exit_directions.png、final_result.png、exit_sweeps.png；灰色物体、蓝色支撑、红色切除材料。图只显示前 100 mm；切除使用完整退出距离。', '', f"实际切除 {report['removed_volume_cm3']:.2f} cm³，剩余 {report['remaining_volume_cm3']:.2f} cm³。",'', '| Pose | 剩余接触承载需求通过数 | 剩余实际接地覆盖 |','|---|---:|---:|']
    for row in rows:lines.append(f"| {row['pose']} | {row['force_covered']}/{row['load_count']} | {row['ground_coverage']['covered']}/{row['ground_coverage']['total']} |")
    lines+=['','各 pose 注册后的初始路径不同；共同切除是所有路径扫掠的并集。每格青色仅显示当前 pose 的扫掠，红色显示所有 pose 共同要求切掉的材料。接地圈覆盖仅作诊断，不要求保住固定环；本阶段尚未重建接地材料。每次从未修改的 Step3.3 材料重新计算，允许恢复先前切掉的材料。', '', '这些是初始化诊断，不是完整夹具接受，也不把当前失败说成总体无解。退出路径优化尚未运行。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');return dict(id=group['id'],force_and_ground_gate=gate,removed_volume_cm3=report['removed_volume_cm3'],remaining_volume_cm3=report['remaining_volume_cm3'],seconds=report['seconds'])

def main():
    import sys
    if '--render-only' in sys.argv:
        sys.argv.remove('--render-only')
        from step41_render import main as render_main
        return render_main()
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];initialization=prepare('B')
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    selected=[g for g in groups if not args.sets or g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run,'B',g,initialization) for g in selected];rows=[f.result() for f in futures]
    save(HERE/'output/B/data/step41_batch.json',dict(complete=True,results=rows,sets=len(rows),optimized=False))
    print('INITIALIZATION COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
