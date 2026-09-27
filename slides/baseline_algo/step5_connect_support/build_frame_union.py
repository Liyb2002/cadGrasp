"""Fit hollow floor frames separately, connect each active triple, then union.

B / pose1+3 specimen. Reuses the previous admissible placements. Geometry is
routed inside both tasks' free space, so this is not a collision-blind union.
Default replays fitted offsets; --refit repeats the finite footprint search.
"""
import argparse,sys,json,time,tempfile,os
from pathlib import Path
import numpy as np,trimesh,manifold3d as md
from scipy.spatial import ConvexHull,cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import MultiPoint,Polygon,Point
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import build_coupled_saddle as S
from step5_base.bearing import seed_probes
from step2_local_support import geometry as G
from step4_floor_contact import equilibrium as Q
from step4_floor_contact.whole_assembly import pressure_centers
from step5_base.bearing import bearing_rays,grounded_matrix
from step5_connect_support.build_shared_geometry import export_viewer


from step5_connect_support.surface_check import surface_distances


def build(output,work,refit=False):
    OUT=Path(work)
    tasks,groups,heads,directions,source,schedule,paths=S.inputs()
    bases=[np.eye(3),S.ROTATION];offsets=[np.zeros(3),-S.TRANSLATION@S.ROTATION]
    WIDTH=.006;HEIGHT=.004;RADIUS=.004;STEP=.005

    def clipped_foot(k,margin):
     p=tasks[k];cop=pressure_centers(p.targets/p.scale,p.domain.com)[0]
     poly=MultiPoint(cop).convex_hull.buffer(margin,quad_segs=6)
     xy=np.asarray(poly.exterior.coords)[:-1]
     v=np.c_[xy,np.zeros(len(xy))]@bases[k]+offsets[k]
     normal=bases[1-k][2];v=G.clip_plane(v,np.r_[-normal,normal@offsets[1-k]])
     return Polygon(S.local_to_world(v,bases[k],offsets[k])[:,:2]).convex_hull

    feet=[]
    for k,p in enumerate(tasks):
     cache=OUT/f'foot{k}.json'
     if not refit:
      margin=[.0385625,.0048125][k]
      poly=clipped_foot(k,margin);feet.append(poly)
      cache.write_text(json.dumps(dict(margin=margin,xy=np.asarray(poly.exterior.coords)[:-1].tolist(),fit='replayed case parameters; final body revalidated'),indent=2))
      continue
     points,normals,owners=bearing_rays(p.domain,groups[k],p.floor,64.)
     targets=Q.padded_targets(p.targets/p.scale,p.scale,12);witness=set(seed_probes(targets,128));trials=[]
     def check(m):
      poly=clipped_foot(k,m);xy=np.asarray(poly.exterior.coords)[:-1]
      a,_=grounded_matrix(points,normals,owners,p.domain.com,p.scale,[dict(pads_xy_m=[xy])],64.)
      solver=Q.BatchSolver(a);ids=sorted(witness)
      probe=solver.solve(targets[ids]);ok=False
      if probe['passed']:
       result=solver.solve(targets);ok=bool(result['passed'])
       witness.update(int(d['index']) for d in result['diagnostics'])
      else:witness.update(ids[int(d['index'])] for d in probe['diagnostics'])
      trials.append(dict(margin=m,passed=ok));print('Foot trial',k,m,ok,flush=True)
      return ok,poly
     lo=0;hi=.06
     if check(0)[0]:hi=0
     else:
      assert check(hi)[0]
      while hi-lo>.001:
       mid=(lo+hi)/2
       if check(mid)[0]:hi=mid
       else:lo=mid
     # 2mm geometric reserve; source hull is the least-area convex enclosure,
     # this uniform offset family is not a globally minimal support design.
     margin=hi+.002;ok,poly=check(margin);assert ok
     cache.write_text(json.dumps(dict(margin=margin,xy=np.asarray(poly.exterior.coords)[:-1].tolist(),trials=trials),indent=2))
     feet.append(poly)

    sweeps=[]
    for k,p in enumerate(tasks):
     cache=OUT/f'sweep{k}.npz'
     if cache.exists():
      z=np.load(cache);m=trimesh.Trimesh(z['v'],z['f'],process=False)
     else:
      obj=trimesh.Trimesh(p.domain.mesh.vertices@bases[k]+offsets[k],p.domain.mesh.faces,process=False)
      m=S.swept_solid(obj,-.5*directions[k]@bases[k]);np.savez_compressed(cache,v=m.vertices,f=m.faces)
     sweeps.append(m)
    print('Sweeps ready',flush=True)
    box=md.Manifold.cube([.0008/S.SCALE]*3).translate([-.0004/S.SCALE]*3)
    forbidden=md.Manifold.batch_boolean([S.solid(m).minkowski_sum(box) for m in sweeps],md.OpType.Add)

    def carve(value):
     for b,o in zip(bases,offsets):
      n=b[2];value=value.trim_by_plane(n.tolist(),float(n@o/S.SCALE))
     return value-forbidden

    rings=[];anchors=[]
    for k,poly in enumerate(feet):
     inner=poly.buffer(-WIDTH)
     assert inner.geom_type=='Polygon' and not inner.is_empty
     cs=md.CrossSection([np.asarray(poly.exterior.coords)[:-1]/S.SCALE],md.FillRule.EvenOdd)-md.CrossSection([np.asarray(inner.exterior.coords)[:-1]/S.SCALE],md.FillRule.EvenOdd)
     mesh=S.unpack(cs.extrude(HEIGHT/S.SCALE));mesh.vertices=mesh.vertices@bases[k]+offsets[k]
     rings.append(carve(S.solid(mesh)))
     centerline=poly.buffer(-WIDTH/2).exterior
     points=np.array([centerline.interpolate(t).coords[0] for t in np.arange(0,centerline.length,.008)])
     anchors.append(np.c_[points,np.full(len(points),HEIGHT/2)]@bases[k]+offsets[k])
     print('Ring',k,rings[-1].volume()*S.SCALE**3*1e6,flush=True)

    # Conservative common free-space lattice is only a routing aid. Every final
    # member is carved and the entire final solid gets continuous sweep validation.
    allheads=[v@b+o for hs,b,o in zip(heads,bases,offsets) for cs in hs for v in cs]
    v=np.vstack(allheads+anchors);low=np.floor((v.min(0)-.025)/STEP)*STEP;high=np.ceil((v.max(0)+.025)/STEP)*STEP
    axes=[np.arange(a,b+STEP/2,STEP) for a,b in zip(low,high)];shape=tuple(map(len,axes))
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    clearance=RADIUS+np.sqrt(3)*STEP/2+.001
    valid=np.ones(len(grid),bool)
    for b,o in zip(bases,offsets):valid&=(grid@b[2]-b[2]@o)>clearance
    ids=np.flatnonzero(valid)
    for m in sweeps:
     for start in range(0,len(ids),512):
      ix=ids[start:start+512];q=grid[ix]
      _,dist,_=trimesh.proximity.closest_point(m,q)
      inside=m.contains(q)
      valid[ix]&=(dist>clearance)&(~inside)
     ids=np.flatnonzero(valid)
     print('Free nodes',len(ids),flush=True)
    coords=np.array(np.unravel_index(ids,shape)).T;nodes=grid[ids];lookup=np.full(len(grid),-1,int);lookup[ids]=np.arange(len(ids));lookup=lookup.reshape(shape)
    rows=[];cols=[];weights=[]
    for delta in [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)>(0,0,0)]:
     nxt=coords+delta;ok=((nxt>=0)&(nxt<shape)).all(1);r=np.flatnonzero(ok);c=lookup[tuple(nxt[ok].T)];keep=c>=0;r=r[keep];c=c[keep]
     rows.extend([r,c]);cols.extend([c,r]);weights.extend([np.full(len(r),STEP*np.linalg.norm(delta))]*2)
    graph=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(len(nodes),len(nodes))).tocsr();tree=cKDTree(nodes)

    def line_clear(a,b):
     q=np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.002)+2));ix=np.rint((q-low)/STEP).astype(int)
     return bool(((ix>=0)&(ix<shape)).all() and np.all(lookup[tuple(ix.T)]>=0))

    def smooth(path):
     result=[path[0]];i=0
     while i<len(path)-1:
      j=len(path)-1
      while j>i+1 and not line_clear(path[i],path[j]):j-=1
      result.append(path[j]);i=j
     return np.asarray(result)

    sphere=trimesh.creation.icosphere(subdivisions=1,radius=RADIUS).vertices

    def tube(path):return [S.solid(G.hull_mesh(np.vstack([a+sphere,b+sphere]))) for a,b in zip(path[:-1],path[1:])]

    fixtures=[];routes=[]
    for k in range(2):
     # Each pose connects its own three heads to its own floor frame.
     _,near=tree.query(anchors[k]);unique=np.unique(near);anchor_for={int(n):anchors[k][np.argmin(np.linalg.norm(anchors[k]-nodes[n],axis=1))] for n in unique}
     dist,prev,src=dijkstra(graph,directed=False,indices=unique,min_only=True,return_predecessors=True)
     parts=[rings[k]];original=[]
     for c,cs in zip(groups[k],heads[k]):
      local=[v@bases[k]+offsets[k] for v in cs];direction=directions[k]@bases[k];center=c['center_m']@bases[k]+offsets[k]
      choices=[]
      for length in [.010,.014,.018,.024,.032,.045,.060]:
       root=center+length*direction;gap,index=tree.query(root);choices.append((gap+length*.15+dist[index]*.1,length,root,index))
      score,length,root,index=min(choices,key=lambda x:x[0]);path=[nodes[index]];end=index
      while prev[end]>=0:end=int(prev[end]);path.append(nodes[end])
      route=np.vstack([root,smooth(np.array(path)),anchor_for[end]])
      parts.extend(tube(route))
      for v in local:
       parts.append(S.solid(G.hull_mesh(np.vstack([v,v+length*direction]))))
       original.append(S.solid(G.hull_mesh(v)))
      routes.append(dict(pose=tasks[k].pose,id=c['candidate_id'],path_m=route.tolist(),head_extension_m=length))
      print('Route',k,c['candidate_id'],length,len(route),float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),flush=True)
     body=carve(md.Manifold.batch_boolean(parts,md.OpType.Add));body=md.Manifold.batch_boolean([body]+original,md.OpType.Add)
     fixtures.append(body);print('Fixture',k,[(x.volume()*S.SCALE**3*1e6) for x in body.decompose()],flush=True)
    merged=md.Manifold.batch_boolean(fixtures,md.OpType.Add)
    # Keep the Boolean's Mesh64 coordinates in the same normalized frame used
    # by the head containment check; avoid an unnecessary recentering round trip.
    mesh=S.unpack(merged)
    count=len(merged.decompose())
    record=dict(component_count=count,boundary_component_count=len(mesh.split(only_watertight=False)),
        watertight=bool(mesh.is_watertight),consistently_wound=bool(mesh.is_winding_consistent),volume_m3=float(mesh.volume),
        one_solid=bool(count==1 and mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0))
    print('Merged',record,flush=True)
    np.savez_compressed(OUT/'geometry.npz',vertices_m=mesh.vertices,faces=mesh.faces,rotations=bases,local_offsets_m=offsets)
    if not record['one_solid']:raise RuntimeError('The merged frames did not form one solid')
    (OUT/'design.json').write_text(json.dumps(dict(routes=routes,solid=record,volume_cm3=mesh.volume*1e6,dimensions_mm=(mesh.extents*1000).tolist()),indent=2))

    checks,cert=S.verify(tasks,groups,heads,directions,bases,offsets,mesh,sweeps)
    np.savez_compressed(OUT/'equilibrium.npz',**cert)
    mesh.export(OUT/'fixture.obj',file_type='obj',digits=17,include_normals=False)
    mm=mesh.copy();mm.apply_scale(1000)
    (OUT/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
    reloaded=trimesh.load(OUT/'fixture_mm.stl',force='mesh')
    if not reloaded.is_watertight or not reloaded.is_winding_consistent:raise RuntimeError('STL reload failed')
    np.testing.assert_allclose(reloaded.extents,mm.extents,atol=1e-10,rtol=0)
    part_meshes=[S.unpack(v) for v in fixtures]
    part_checks=[];part_certificates={}
    for k,part in enumerate(part_meshes):
     if not part.is_watertight or len(part.split())!=1:raise RuntimeError('A task-specific structure is disconnected')
     filename=f'pose{[1,3][k]}_structure.obj'
     part.export(OUT/filename,file_type='obj',digits=17,include_normals=False)
     own_checks,own_cert=S.verify([tasks[k]],[groups[k]],[heads[k]],[directions[k]],[bases[k]],[offsets[k]],part,[sweeps[k]])
     part_certificates.update(own_cert)
     part_checks.append(dict(pose=tasks[k].pose,volume_cm3=float(part.volume*1e6),connected=True,watertight=True,file=filename,own_task_check=own_checks[0]))
    np.savez_compressed(OUT/'per_pose_equilibrium.npz',**part_certificates)
    artifacts=['fixture.obj','fixture_mm.stl','geometry.npz','equilibrium.npz','pose1_structure.obj','pose3_structure.obj','design.json','per_pose_equilibrium.npz']
    code=[Path(__file__),Path(S.__file__),Path(S.swept_solid.__code__.co_filename),Path(Q.__file__),
          Path(S.W.__file__),Path(G.__file__),Path(S.SOL.__file__),Path(S.FLOOR.__file__),
          Path(bearing_rays.__code__.co_filename),Path(surface_distances.__code__.co_filename),
          Path(__file__).with_name('shared_geometry_viewer.html'),Path(__file__).with_name('export_shared_geometry.cjs')]
    report=dict(object='B',poses=S.POSES,particle=9,complete=True,status='geometry_and_fixed_sample_equilibrium_passed',
     design='separate_floor_frames_and_head_connections_then_union',same_rigid_solid_in_both_poses=True,
     contact_groups=5,contact_patch_count=6,original_task_poses_changed=False,original_active_contacts_preserved=True,
     shared_branch_definition='two distinct orange contact patches, six patches total; same successful relative placement as the saddle specimen',
     support_mass_ignored=True,extra_loads_added=False,structural_strength_verified=False,
     dimensions_mm=(mesh.extents*1000).tolist(),volume_cm3=float(mesh.volume*1e6),solid=record,checks=checks,
     source_schedule=str((source/'schedule.json').relative_to(S.ROOT)),
     task_fixture_transforms=[dict(pose=p.pose,rotation=b.tolist(),translation_m=(-b@o).tolist()) for p,b,o in zip(tasks,bases,offsets)],
     body_design=dict(method='per-pose hollow floor boundary and three head connections; union after common-space clearance',
      separately_constructed_parts=part_checks,frame_width_m=WIDTH,frame_height_m=HEIGHT,nominal_member_diameter_m=2*RADIUS,
      floor_fits=[json.loads((OUT/f'foot{k}.json').read_text()) for k in range(2)],routes=routes,
      optimization_scope='least-area convex cloud enclosure as initializer, then bounded uniform-offset fitting; not globally minimal geometry',
      common_constraints='fixed previously feasible placements; opposite floor and both complete sweeps constrain every frame and connection',
      frame_openings='small portions removed where another pose needs withdrawal clearance; interior remains empty',
      merge_overlap_cm3=float(sum(m.volume for m in part_meshes)*1e6-mesh.volume*1e6)),
     comparison=dict(previous_saddle_volume_cm3=575.1613127639359,volume_reduction_fraction=float(1-mesh.volume*1e6/575.1613127639359),
      structural_equivalence_verified=False,interpretation='geometric volume comparison under the same ideal rigid-body load checks; no equal-stiffness or strength claim'),
     presentation_description='每个 pose 分别用窄地框连接自己的三个头，再合并成一件。框内留空，装卸冲突处留开口。可切换查看各 pose 单独结构；橙色仍有两个不同接触面。',
     exports=dict(obj_units='m',stl_units='mm',stl_encoding='ASCII',reloaded_stl_watertight=True),
     artifacts={name:S.I.sha256(OUT/name) for name in artifacts},
     provenance=dict(inputs=S.I.hashes(paths),code=S.I.hashes(code)))
    S.I.save(OUT/'report.json',report)
    colors={schedule['shared_head']['selected_id']:'#dc9d47'}
    colors.update(zip([i for i in schedule['selected_ids'] if i not in colors],['#ac7098','#7196c0','#50a59b','#77a76a']))
    visual=[(c['candidate_id'],G.hull_mesh(np.concatenate([v@b+o for v in cs]))) for gs,hs,b,o in zip(groups,heads,bases,offsets) for c,cs in zip(gs,hs)]
    export_viewer(OUT,mesh,visual,report,tasks,bases,offsets,colors)
    # Same physical component meshes in the optional per-pose presentation.
    from step5_connect_support.refresh_shared_geometry_view import write_viewer
    from step5_connect_support.build_shared_geometry import pack
    html=(OUT/'index.html').read_text();data,_=json.JSONDecoder().raw_decode(html.split('const DATA=',1)[1])
    data['design_parts']=[pack(m) for m in part_meshes];write_viewer(OUT,data)
    output=Path(output) if output else S.pair_folder('B',S.POSES,'step3_scheculer').parent/'step5'
    output.mkdir(parents=True,exist_ok=True)
    for name in artifacts+['report.json','index.html']:os.replace(OUT/name,output/name)
    print('EXPORTED',output,report['volume_cm3'],flush=True)
    return output


def run(output=None,refit=False):
    with tempfile.TemporaryDirectory(prefix='cadgrasp_frame_union_') as work:
        return build(output,work,refit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--refit',action='store_true',help='Refit the per-pose uniform footprint offsets against original samples')
    args=parser.parse_args();run(args.output,args.refit)
