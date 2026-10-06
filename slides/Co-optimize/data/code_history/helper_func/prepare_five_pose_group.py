"""Compute-only prerequisites for a single immutable five-pose experiment."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
for folder in ['helper_func','step3.1','step3.2','step3.3','step4.1']:
    sys.path.insert(0,str(HERE/folder))
import _bootstrap
from co_common import *
from scipy.spatial import ConvexHull
import time
import step31,step32,step41
from step33 import demand_hull,perimeter

class CommonSurfaceInfeasible(RuntimeError):
    """Certified failure of the full common-surface force upper bound."""


def build_rings(name,group):
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
    D.export_exact_obj(final,out/'support_with_rings.obj')
    passed=all(r['all_saved_demands_covered'] for r in rows) and all(r['working_surface_clear'] for r in checks)
    report=dict(complete=True,passed=passed,status='pass' if passed else 'unresolved',pose_set=group['id'],state_results=rows,state_geometry=checks,component_count=len(combined.decompose()),full_fixture_accepted=False,exit_checked=False,connection_policy='No connecting rods; connectivity is not a Step3.3 requirement',floor_policy='Source ring sits on its native floor; cross-state floor intersections remain Step4 carving constraints',minimum_scope='Exact convex hull of all original sampled ground demands; 5 mm perimeter grows inward; no circular fit or outer enlargement; obstacle-cut material must still cover all original demands',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step3.3/step33.py',Path(__file__)]),artifacts={f'../{f}':I.sha256(out/f) for f in ['support_with_rings.obj']})
    save(out/'data/report.json',report)
    print(group['id'],report['status'],report['component_count'],round(report['seconds'],2),flush=True)
    return dict(id=group['id'],passed=passed,component_count=report['component_count'],seconds=report['seconds'])

def prepare_group(name,group):
    """Only this group; retain original poses, loads and geometry policies."""
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']
    if (base/'step4/step4.1/data/report.json').exists():
        raise FileExistsError('Refusing to overwrite an existing initialization')
    states,mesh,_=step31.run(name,group,base/'step3/step3.1')
    step32.run(name,group,states,mesh,base/'step3/step3.2')
    gate=json.loads((base/'step3/step3.2/data/report.json').read_text())
    if gate['status']=='fail':
        raise CommonSurfaceInfeasible('Step3.2 proves a saved load impossible even with every common non-work source triangle; see failure_proof')
    build_rings(name,group)
    seed_path=base/'step3/step3.3/support_with_rings.obj'
    support=trimesh.load(seed_path,force='mesh',process=False)
    # The diagonal bound covers projection spans for EVERY optimized direction,
    # not merely the native starting exits; endpoint separation is checked later.
    combined=np.vstack([support.vertices,mesh.vertices])
    length=max(.5,float(np.linalg.norm(np.ptp(combined,axis=0)))+.02)
    folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41';folder.mkdir(parents=True,exist_ok=True)
    paths={};artifacts={};inputs=[seed_path,ROOT/'objects'/name/'pose_sets.json']
    for pose,(task,T,m) in states.items():
        direction,native=step41.initialize_direction(T)
        path=folder/pose;path.mkdir(parents=True,exist_ok=True)
        for kind,distance in [('full',length),('display',.10)]:
            D.export_exact_obj(S.swept_solid(m,distance*direction),path/f'{kind}_sweep.obj')
            artifacts[f'{pose}/{kind}_sweep.obj']=I.sha256(path/f'{kind}_sweep.obj')
        paths[pose]=dict(direction_fixture=direction.tolist(),direction_world=native.tolist())
        inputs+=task.inputs
    initialization=dict(complete=True,initialization_kind='Each pose withdraws along its own native world +z',
        paths=paths,full_length_m=length,display_length_m=.10,initialization_is_arbitrary=True,
        paths_independent_variables=True,initial_canonical_sweeps_identical=False,
        length_policy='Combined seed/object AABB diagonal plus 20 mm; at least 500 mm',
        provenance=provenance(inputs,[Path(__file__),HERE/'step4.1/step41.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts)
    save(folder/'initialization.json',initialization)
    step41.run(name,group,initialization,render_images=False)
    directions=base/'step4/step4.1/data/native_start_directions.npz'
    np.savez_compressed(directions,directions=np.array([paths[p]['direction_fixture'] for p in group['poses']]))
    return directions
