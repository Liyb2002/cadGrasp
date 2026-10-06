"""Saved ground demands -> minimum convex hulls -> unconnected ring material seeds."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from scipy.spatial import ConvexHull
import time,argparse

def demand_hull(points):
    points=np.asarray(points,float)
    hull=ConvexHull(points)
    return points[hull.vertices],hull.vertices,float(hull.volume)


def perimeter(polygon,width=.005,height=.005):
    """Keep the exact demand hull as the outer edge; material grows inward."""
    outer=F.md.CrossSection([np.asarray(polygon)/S.SCALE])
    inner=outer.offset(-width/S.SCALE,join_type=F.md.JoinType.Miter,miter_limit=1000.)
    return (outer-inner).extrude(height/S.SCALE)


def rod(a,b,r=.003):
    ball=trimesh.creation.icosphere(subdivisions=1,radius=r)
    return S.solid(G.hull_mesh(np.vstack([ball.vertices+a,ball.vertices+b])))

from step33_legacy_render import render

def run(name,group):
    began=time.monotonic();base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3';out=base/'step3.3';(out/'data').mkdir(parents=True,exist_ok=True)
    shellmesh=trimesh.load(base/'step3.2/wrapped_support.obj',force='mesh',process=False);shell=S.solid(shellmesh)
    _,_,mesh=state(name,group['poses'][0]);work=np.unique(np.concatenate([state(name,p)[0].domain.work_ids for p in group['poses']]))
    offsets=wrap_offsets(mesh,.005);obstacle=union([S.solid(mesh)]+[S.solid(G.hull_mesh(G.head_cell(mesh,mesh.triangles[i],i,offsets))) for i in work])
    parts=[];rows=[];points=[];inputs=[base/'step3.2/wrapped_support.obj',base/'step3.2/data/report.json'];combined=shell
    for pose in group['poses']:
        task,T,_=state(name,pose);source=ROOT/'objects'/name/'poses'/pose/'floor_contact.npz'
        p=np.load(source)['floor_demands_xy_m'];polygon,indices,area=demand_hull(p)
        ringmesh=S.unpack(perimeter(polygon));ringmesh.apply_transform(np.linalg.inv(T))
        ring=S.solid(ringmesh)-obstacle
        ground=S.unpack(ring);world=transform_points(ground.vertices,T)
        xy=world[np.abs(world[:,2])<1e-9,:2]
        actual_hull=[];coverage=False
        if len(xy)>=3:
            hull=ConvexHull(xy);actual_hull=xy[hull.vertices].tolist()
            coverage=bool(np.all(p@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9))
        parts.append(ring);combined=union([combined,ring])
        row=dict(pose=pose,saved_demand_count=len(p),boundary_kind='minimum_convex_hull',
                 demand_hull_world_xy_m=polygon.tolist(),minimum_hull_area_cm2=area*1e4,
                 boundary_sample_indices=indices.tolist(),ring_width_m=.005,ring_height_m=.005,
                 thickness_direction='inward from exact hull boundary',outer_expansion_m=0.,
                 actual_ground_hull_world_xy_m=actual_hull,all_saved_demands_covered=coverage,
                 connection_required=False)
        rows.append(row);points.append(p);inputs.append(source)
        np.savez_compressed(out/'data'/f'{pose}.npz',saved_demands_world_xy_m=p,
                            demand_hull_world_xy_m=polygon,T_fixture_to_world=T)
        D.export_exact_obj(S.unpack(ring),out/'data'/f'{pose}_perimeter.obj')
    final=S.unpack(combined);checks=[]
    for pose in group['poses']:
        task,T,_=state(name,pose);world=final.copy();world.apply_transform(T);check=WORK.check(world,task);checks.append(dict(pose=pose,working_surface_clear=check['passed'],min_world_z_m=float(world.vertices[:,2].min())))
    D.export_exact_obj(final,out/'support_with_rings.obj');render(mesh,shellmesh,parts,rows,out,points)
    passed=all(r['all_saved_demands_covered'] for r in rows) and all(r['working_surface_clear'] for r in checks)
    report=dict(complete=True,passed=passed,status='pass' if passed else 'unresolved',pose_set=group['id'],state_results=rows,state_geometry=checks,component_count=len(combined.decompose()),full_fixture_accepted=False,exit_checked=False,connection_policy='No connecting rods; connectivity is not a Step3.3 requirement',floor_policy='Source ring sits on its native floor; cross-state floor intersections remain Step4 carving constraints',minimum_scope='Exact convex hull of all original sampled ground demands; 5 mm perimeter grows inward; no circular fit or outer enlargement; obstacle-cut material must still cover all original demands',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step3.3/step33.py']),artifacts={f'../{f}':I.sha256(out/f) for f in ['support_with_rings.obj','overview.png']})
    save(out/'data/report.json',report)
    (out/'README.md').write_text(f"# Step3.3：各 pose 下的壳子与凸包围边\n\n{report['status'].upper()}。唯一图片 `overview.png` 按 pose 分格：将同一个壳子和全部凸包围边放回各 pose 的原生世界坐标，显示地面与原始撒点。灰色是壳子，围边的颜色标识来源 pose。\n\n本阶段不生成连接杆，也不要求连成整体。检查全部保存撒点的真实接地凸包覆盖与工作面自由。跨状态地面冲突、退出路径和最终连接留给后续步骤。\n")
    (out/'ground_demands.png').unlink(missing_ok=True)
    print(group['id'],report['status'],report['component_count'],round(report['seconds'],2),flush=True)
    return dict(id=group['id'],passed=passed,component_count=report['component_count'],seconds=report['seconds'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--scope',choices=['all','legal','illegal'],default='all');args=parser.parse_args()
    groups=[]
    if args.scope in ('all','legal'):groups+=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    if args.scope in ('all','illegal'):groups+=[dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    rows=[run('B',group) for group in groups]
    save(HERE/'output/B/data/step33_batch.json',dict(complete=True,results=rows,passed_sets=sum(r['passed'] for r in rows),boundary_kind='minimum_convex_hull'))
    print('STEP3.3 CONVEX HULL TOTAL',len(rows),'sets',sum(r['passed'] for r in rows),'passed',flush=True)
    if not all(r['passed'] for r in rows):raise SystemExit(2)
if __name__=='__main__':main()
