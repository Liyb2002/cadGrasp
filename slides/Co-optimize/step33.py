"""Saved ground demands -> minimum circles -> unconnected ring material seeds."""
from co_common import *
from scipy.spatial import ConvexHull,cKDTree
import time,argparse

def circle(points):
    p=np.asarray(points,float); hull=ConvexHull(p).vertices if len(p)>2 else np.arange(len(p))
    order=np.random.default_rng(0).permutation(hull);c=p[order[0]].copy();r=0.;support=[int(order[0])]
    def outside(q):return np.linalg.norm(q-c)>r+1e-12
    for ii,i in enumerate(order):
        if not outside(p[i]):continue
        c=p[i].copy();r=0.;support=[int(i)]
        for jj,j in enumerate(order[:ii]):
            if not outside(p[j]):continue
            c=(p[i]+p[j])/2;r=np.linalg.norm(p[i]-c);support=[int(i),int(j)]
            for k in order[:jj]:
                if not outside(p[k]):continue
                a,b,d=p[i],p[j],p[k];mat=2*np.array([b-a,d-a])
                if abs(np.linalg.det(mat))<1e-20:
                    pairs=[(i,j),(i,k),(j,k)];u,v=max(pairs,key=lambda uv:np.linalg.norm(p[uv[0]]-p[uv[1]]));c=(p[u]+p[v])/2;r=np.linalg.norm(p[u]-c);support=[int(u),int(v)]
                else:c=np.linalg.solve(mat,np.array([b@b-a@a,d@d-a@a]));r=np.linalg.norm(a-c);support=[int(i),int(j),int(k)]
    assert np.max(np.linalg.norm(p-c,axis=1))<=r+2e-11
    return c,float(r),support

def cylinder(center,radius,height):
    return F.md.Manifold.cylinder(height/S.SCALE,radius/S.SCALE,circular_segments=128).translate((center[0]/S.SCALE,center[1]/S.SCALE,center[2]/S.SCALE))

def rod(a,b,r=.003):
    ball=trimesh.creation.icosphere(subdivisions=1,radius=r)
    return S.solid(G.hull_mesh(np.vstack([ball.vertices+a,ball.vertices+b])))

def render(mesh,shell,parts,rows,out,points):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    colors=plt.get_cmap('tab10').colors;n=len(rows);cols=min(3,n);nr=(n+cols-1)//cols
    fig=plt.figure(figsize=(5*cols,5*nr))
    states=[np.load(out/'data'/f"{row['pose']}.npz")['T_fixture_to_world'] for row in rows]
    canonical=[shell]+[S.unpack(part) for part in parts]
    extent=max(np.ptp(np.vstack([transform_points(m.vertices,T) for m in canonical]),axis=0).max() for T in states)*1000
    for i,(row,T,p) in enumerate(zip(rows,states,points)):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d')
        obj=mesh.copy();obj.apply_transform(T)
        ax.add_collection3d(Poly3DCollection(obj.triangles*1000,facecolor='#77b7dd',alpha=.22,edgecolor='none'))
        vertices=[]
        for j,m in enumerate(canonical):
            world=m.copy();world.apply_transform(T);vertices.append(world.vertices*1000)
            color='#999999' if j==0 else colors[j-1]
            ax.add_collection3d(Poly3DCollection(world.triangles*1000,facecolor=color,alpha=.6 if j==0 else .9,edgecolor='none'))
        v=np.vstack(vertices);center=(v.min(0)+v.max(0))/2;r=extent*.56
        floor=np.array([[center[0]-r,center[1]-r,0],[center[0]+r,center[1]-r,0],[center[0]+r,center[1]+r,0],[center[0]-r,center[1]+r,0]])
        ax.add_collection3d(Poly3DCollection([floor],facecolor='#eeeeee',edgecolor='#bbbbbb',alpha=.2))
        ax.scatter(p[:,0]*1000,p[:,1]*1000,np.zeros(len(p)),s=.1,alpha=.12,color=colors[i],rasterized=True)
        for axis,x in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(x-r,x+r)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.set_axis_off();ax.set_title(row['pose'].replace('_',' ').title()+f" — ground ring {i+1}")
    fig.suptitle('Step3.3: the same shell and ground rings in each pose\nGrey = shell; colored rings = pose ground demands; no connecting rods',fontsize=13)
    fig.legend(handles=[Patch(color=colors[i],label=row['pose']) for i,row in enumerate(rows)],loc='lower center',ncol=n)
    fig.subplots_adjust(left=0,right=1,bottom=.06,top=.78,wspace=0,hspace=.05)
    fig.savefig(out/'overview.png',dpi=150,bbox_inches='tight');plt.close(fig)

def run(name,group):
    began=time.monotonic();base=HERE/'output'/name/group['id']/'step3';out=base/'step3.3';(out/'data').mkdir(parents=True,exist_ok=True)
    shellmesh=trimesh.load(base/'step3.2/wrapped_support.obj',force='mesh',process=False);shell=S.solid(shellmesh)
    _,_,mesh=state(name,group['poses'][0]);work=np.unique(np.concatenate([state(name,p)[0].domain.work_ids for p in group['poses']]))
    offsets=wrap_offsets(mesh,.005);obstacle=union([S.solid(mesh)]+[S.solid(G.hull_mesh(G.head_cell(mesh,mesh.triangles[i],i,offsets))) for i in work])
    parts=[];rows=[];points=[];inputs=[base/'step3.2/wrapped_support.obj',base/'step3.2/data/report.json'];combined=shell
    for pose in group['poses']:
        task,T,_=state(name,pose);source=ROOT/'objects'/name/'poses'/pose/'floor_contact.npz';p=np.load(source)['floor_demands_xy_m'];c,r,indices=circle(p);inv=np.linalg.inv(T)
        # Circumscribed outer polygon, not an inscribed circle approximation.
        physical_r=r/np.cos(np.pi/128)+1e-7
        for enlargement in [0.,.0005,.001,.002,.004,.008,.016,.032]:
            actual_radius=physical_r+enlargement
            native=cylinder([*c,0],actual_radius,.005)-cylinder([*c,-.001],max(actual_radius-.005,1e-5),.007)
            ringmesh=S.unpack(native);ringmesh.apply_transform(inv);ring=S.solid(ringmesh)-obstacle
            ground=S.unpack(ring);world=transform_points(ground.vertices,T);xy=world[np.abs(world[:,2])<1e-9,:2]
            hull=ConvexHull(xy);coverage=bool(np.all(p@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9))
            if coverage:break
        parts.append(ring);combined=union([combined,ring])
        row=dict(pose=pose,saved_demand_count=len(p),center_world_xy_m=c.tolist(),minimum_radius_m=r,minimum_circle_area_cm2=np.pi*r*r*1e4,boundary_sample_indices=indices,actual_outer_radius_m=actual_radius,clearance_enlargement_m=enlargement,ring_width_m=.005,ring_height_m=.005,actual_ground_hull_world_xy_m=xy[hull.vertices].tolist(),all_saved_demands_covered=coverage,connection_required=False)
        rows.append(row);points.append(p);inputs.append(source);np.savez_compressed(out/'data'/f'{pose}.npz',saved_demands_world_xy_m=p,circle_center_world_xy_m=c,circle_radius_m=r,T_fixture_to_world=T)
    final=S.unpack(combined);checks=[]
    for pose in group['poses']:
        task,T,_=state(name,pose);world=final.copy();world.apply_transform(T);check=WORK.check(world,task);checks.append(dict(pose=pose,working_surface_clear=check['passed'],min_world_z_m=float(world.vertices[:,2].min())))
    D.export_exact_obj(final,out/'support_with_rings.obj');render(mesh,shellmesh,parts,rows,out,points)
    passed=all(r['all_saved_demands_covered'] for r in rows) and all(r['working_surface_clear'] for r in checks)
    report=dict(complete=True,passed=passed,status='pass' if passed else 'unresolved',pose_set=group['id'],state_results=rows,state_geometry=checks,component_count=len(combined.decompose()),full_fixture_accepted=False,exit_checked=False,connection_policy='No connecting rods; connectivity is not a Step3.3 requirement',floor_policy='Source ring sits on its native floor; cross-state floor intersections remain Step4 carving constraints',minimum_scope='Minimum enclosing circular demand region, not globally minimum arbitrary footprint or material',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step33.py']),artifacts={f'../{f}':I.sha256(out/f) for f in ['support_with_rings.obj','overview.png']})
    save(out/'data/report.json',report)
    (out/'README.md').write_text(f"# Step3.3：各 pose 下的壳子与接地圈\n\n{report['status'].upper()}。唯一图片 `overview.png` 按 pose 分格：将同一个壳子和全部接地圈放回各 pose 的原生世界坐标，显示地面与原始撒点。灰色是壳子，圈的颜色标识来源 pose。\n\n本阶段不生成连接杆，也不要求连成整体。检查全部保存撒点的真实接地凸包覆盖与工作面自由。跨状态地面冲突、退出路径和最终连接留给后续步骤。\n")
    (out/'ground_demands.png').unlink(missing_ok=True)
    print(group['id'],report['status'],report['component_count'],round(report['seconds'],2),flush=True)
    return dict(id=group['id'],passed=passed,component_count=report['component_count'],seconds=report['seconds'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];rows=[]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    selected=[group for group in groups if not args.sets or group['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run,'B',group) for group in selected]
        rows=[future.result() for future in futures]
    save(HERE/'output/B/data/step33_batch.json',dict(complete=True,results=rows,passed_sets=sum(r['passed'] for r in rows)))
if __name__=='__main__':main()
