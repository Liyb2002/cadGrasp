"""Saved ground demands -> adaptively expanded convex hulls -> unconnected ring material seeds."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from work_access import RegisteredWorkVolumes
from scipy.spatial import ConvexHull
import time,argparse

def demand_hull(points):
    points=np.asarray(points,float)
    hull=ConvexHull(points)
    return points[hull.vertices],hull.vertices,float(hull.volume)


def perimeter(polygon,width=.005,height=.005,expansion=0.):
    """Keep a fixed-width ring around the optionally expanded demand hull."""
    outer=F.md.CrossSection([np.asarray(polygon)/S.SCALE])
    if expansion:
        outer=outer.offset(expansion/S.SCALE,join_type=F.md.JoinType.Miter,miter_limit=1000.)
    inner=outer.offset(-width/S.SCALE,join_type=F.md.JoinType.Miter,miter_limit=1000.)
    return (outer-inner).extrude(height/S.SCALE)


def ground_hull_coverage(mesh,T,points):
    world=transform_points(mesh.vertices,T)
    xy=world[np.abs(world[:,2])<1e-9,:2]
    # Union can erase a source ring's coplanar boundary vertices when another
    # part crosses its floor. Include actual boundary/plane intersections.
    edges=world[mesh.edges_unique]
    z0,z1=edges[:,0,2],edges[:,1,2]
    crossing=((z0<0)&(z1>0))|((z0>0)&(z1<0))
    if crossing.any():
        selected=edges[crossing];t=-z0[crossing]/(z1[crossing]-z0[crossing])
        intersections=selected[:,0,:2]+t[:,None]*(selected[:,1,:2]-selected[:,0,:2])
        xy=np.vstack([xy,intersections])
    if len(xy)<3:return [],False,0
    try:hull=ConvexHull(xy)
    except Exception:return [],False,0
    mask=np.all(points@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9,axis=1)
    return xy[hull.vertices].tolist(),bool(mask.all()),int(mask.sum())


def rod(a,b,r=.003):
    ball=trimesh.creation.icosphere(subdivisions=1,radius=r)
    return S.solid(G.hull_mesh(np.vstack([ball.vertices+a,ball.vertices+b])))

from step33_legacy_render import render

def run(name,group):
    began=time.monotonic();base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3';out=base/'step3.3';(out/'data').mkdir(parents=True,exist_ok=True)
    wrap_path=initialized_wrap(base)
    shellmesh=trimesh.load(wrap_path,force='mesh',process=False);shell=S.solid(shellmesh)
    _,_,mesh=state(name,group['poses'][0]);work=np.unique(np.concatenate([state(name,p)[0].domain.work_ids for p in group['poses']]))
    offsets=wrap_offsets(mesh,.005);obstacle=union([S.solid(mesh)]+[S.solid(G.hull_mesh(G.head_cell(mesh,mesh.triangles[i],i,offsets))) for i in work])
    initialization=json.loads((wrap_path.parent/'data/report.json').read_text())
    access=RegisteredWorkVolumes(mesh,{p:state(name,p) for p in group['poses']}) if initialization.get('schema')=='whole_registered_work_cones_v1' else None
    parts=[];rows=[];points=[];inputs=[wrap_path,wrap_path.parent/'data/report.json'];combined=shell
    for pose in group['poses']:
        task,T,_=state(name,pose);source=ROOT/'objects'/name/'poses'/pose/'floor_contact.npz'
        p=np.load(source)['floor_demands_xy_m'];polygon,indices,area=demand_hull(p)
        trials=[];max_expansion=max(.064,2.*float(np.linalg.norm(mesh.extents)))
        expansion=0.
        while True:
            ringmesh=S.unpack(perimeter(polygon,expansion=expansion));ringmesh.apply_transform(np.linalg.inv(T))
            if access is not None:
                forbidden,_=access.solid_for_points(np.vstack([shellmesh.vertices,ringmesh.vertices]))
                ring=S.solid(ringmesh)-obstacle-forbidden
            else:ring=S.solid(ringmesh)-obstacle
            actual_hull,coverage,covered=ground_hull_coverage(S.unpack(ring),T,p)
            trials.append(dict(expansion_m=expansion,covered=covered,total=len(p)))
            if coverage or expansion>=max_expansion:break
            expansion=min(max_expansion,.001 if expansion==0. else expansion*2.)
        parts.append(ring);combined=union([combined,ring])
        row=dict(pose=pose,saved_demand_count=len(p),boundary_kind='expanded_convex_hull',
                 demand_hull_world_xy_m=polygon.tolist(),minimum_hull_area_cm2=area*1e4,
                 boundary_sample_indices=indices.tolist(),ring_width_m=.005,ring_height_m=.005,
                 thickness_direction='inward from expanded hull boundary',outer_expansion_m=expansion,
                 expansion_trials=trials,expansion_limit_m=max_expansion,own_ring_demands_covered=coverage,
                 actual_ground_hull_world_xy_m=actual_hull,all_saved_demands_covered=coverage,
                 connection_required=False)
        rows.append(row);points.append(p);inputs.append(source)
        np.savez_compressed(out/'data'/f'{pose}.npz',saved_demands_world_xy_m=p,
                            demand_hull_world_xy_m=polygon,T_fixture_to_world=T)
        D.export_exact_obj(S.unpack(ring),out/'data'/f'{pose}_perimeter.obj')
    final=S.unpack(combined);checks=[]
    for pose in group['poses']:
        task,T,_=state(name,pose);actual_hull,covered,count=ground_hull_coverage(final,T,points[len(checks)]);rows[len(checks)].update(actual_ground_hull_world_xy_m=actual_hull,all_saved_demands_covered=covered,combined_support_covered_count=count,coverage_scope='combined_support');world=final.copy();world.apply_transform(T);check=WORK.check(world,task);checks.append(dict(pose=pose,working_surface_clear=check['passed'],min_world_z_m=float(world.vertices[:,2].min())))
    D.export_exact_obj(final,out/'support_with_rings.obj');render(mesh,shellmesh,parts,rows,out,points,name=name,group=group)
    if access is not None:
        forbidden,_=access.solid_for_points(final.vertices)
        work_overlap=material_volume(combined^forbidden)
        if work_overlap>1e-10:raise RuntimeError('Ground ring material violates a full working cone')
    else:work_overlap=None
    diagnostics_passed=all(r['all_saved_demands_covered'] for r in rows) and all(r['working_surface_clear'] for r in checks)
    # Step3.3 constructs an initialization seed; these are not rejection gates.
    passed=True
    report=dict(complete=True,passed=passed,status='pass',construction_completed=True,step4_ready=True,diagnostics_passed=diagnostics_passed,diagnostics_are_acceptance_gates=False,pose_set=group['id'],state_results=rows,state_geometry=checks,component_count=len(combined.decompose()),full_fixture_accepted=False,exit_checked=False,connection_policy='No connecting rods; connectivity is not a Step3.3 requirement',floor_policy='Accept floor penetration during Step3.3 seed construction; record native minimum z and defer floor carving to Step4',minimum_scope='Start at exact demand hull; deterministically expand fixed-width perimeter until obstacle-cut contact hull covers all demands or bounded search ends; final coverage checks combined support. No minimum-size or infeasibility claim',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step3.3/step33.py']),artifacts={f'../{f}':I.sha256(out/f) for f in ['support_with_rings.obj','overview.png']})
    save(out/'data/report.json',report)
    if access is not None:
        report.update(full_work_cones_preserved=True,continuous_work_union_overlap_m3=work_overlap,
                      work_access_definition=access.definition(),source_initialization_schema=initialization['schema'])
        report['provenance']=provenance(inputs,[HERE/'step3.3/step33.py',HERE/'helper_func/work_access.py',HERE/'helper_func/co_common.py'])
        save(out/'data/report.json',report)
    (out/'README.md').write_text(f"# Step3.3：各 pose 下的壳子与凸包围边\n\n{report['status'].upper()}。唯一图片 `overview.png` 按 pose 分格：将同一个壳子和全部凸包围边放回各 pose 的原生世界坐标，显示地面与原始撒点。灰色是壳子，围边的颜色标识来源 pose。\n\n本阶段不生成连接杆，也不要求连成整体。允许向外扩大 5 mm 宽的环，再扣除物体和工作面禁占体；接地覆盖、工作面和最低地面高度只作诊断，不作为本阶段拒绝条件。允许环插入其他 pose 的地面，先接受初始化支撑；跨状态地面冲突、退出路径和最终连接留给后续步骤。\n")
    (out/'ground_demands.png').unlink(missing_ok=True)
    print(group['id'],report['status'],report['component_count'],round(report['seconds'],2),flush=True)
    return dict(id=group['id'],passed=passed,component_count=report['component_count'],seconds=report['seconds'])

def main():
    from codes.precompute_objects.dataset import read_selected_pose_groups
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    parser=argparse.ArgumentParser();parser.add_argument('object',nargs='?',default='B');parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--scope',choices=['all','legal','illegal'],default='all');args=parser.parse_args()
    groups=read_selected_pose_groups(args.object,scope=args.scope)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    root=HERE/'output'/args.object if args.object=='B' else HERE/'data/object_inputs'/args.object
    ready=[];skipped=[]
    for group in groups:
        path=initialized_wrap(root/group['id']).parent/'data/report.json'
        if path.exists() and json.loads(path.read_text()).get('step4_ready',False):ready.append(group)
        else:skipped.append(dict(id=group['id'],passed=False,status='step3_force_gate_stopped'))
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(run,[args.object]*len(ready),ready))
    save(root/'data/step33_batch.json',dict(complete=True,object=args.object,results=rows,skipped=skipped,passed_sets=sum(r['passed'] for r in rows),boundary_kind='expanded_convex_hull'))
    print('STEP3.3 CONVEX HULL TOTAL',len(rows),'sets',sum(r['passed'] for r in rows),'passed',len(skipped),'gate-stopped',flush=True)
if __name__=='__main__':main()
