"""200 centers allocated by surface area; target 1%, total wrap <=90 degrees."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

HERE=Path(__file__).resolve().parent
BASELINE=HERE.parent
sys.path.insert(0,str(BASELINE))
from step1.needs import ContinuousNeeds, OUTPUTS, OBJECTS, sha256
from step1.cases import pose_name
from step2_local_support import surface as S, geometry as G
from step2_local_support import bearing_direction as H
from step2_local_support import support_policy as POLICY
from step2_local_support import work_volume as W, work_clearance as WC, audit_work_volume as WA
from step2_local_support.surface import fan, area, areas, eligible_polygons, surface_centers

CENTER_COUNT=S.CENTER_COUNT
AREA_FRACTION=.01
AREA_REL_TOL=1e-4
DEPTH_FRACTION=.01
CIRCLE_SIDES=128
MAX_NORMAL_ANGLE_DEG=90.
MAX_WRAP_ANGLE_DEG=90.
NORMAL_DOT_TOL=1e-10
FLOOR_CLEARANCE_M=S.FLOOR_CLEARANCE_M


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def geometry_signature(domain):
    """Geometry and process-access inputs that determine candidate eligibility."""
    data=domain.data
    payload=dict(vertices_m=data['geometry']['vertices_m'],faces=data['geometry']['faces'],
                 work_face_ids=data['geometry']['work_face_ids'],
                 moment_origin_m=data['frame']['moment_origin_m'],
                 cone_half_deg=data['load']['cone_half_deg'],reachability=data['reachability'])
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def normal_spread(mesh,faces):
    """Greatest angle between any two actual source-face normals."""
    ids=np.asarray(sorted(faces),int)
    if not len(ids):
        return dict(wrap_angle_degrees=0.,wrap_limit_satisfied=False,wrap_extreme_faces=[])
    normals=mesh.face_normals[ids]
    minimum=1.;pair=(0,0)
    for start in range(0,len(ids),256):
        dots=normals[start:start+256]@normals.T
        i,j=np.unravel_index(np.argmin(dots),dots.shape)
        if dots[i,j]<minimum:minimum=float(dots[i,j]);pair=(start+i,j)
    first,second=normals[list(pair)]
    angle=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first,second)),first@second)))
    return dict(wrap_angle_degrees=angle,
                wrap_limit_satisfied=bool(minimum>=np.cos(np.deg2rad(MAX_WRAP_ANGLE_DEG))-NORMAL_DOT_TOL),
                wrap_extreme_faces=ids[list(pair)].tolist())


def edge_interval(poly, normal, segment, tolerance):
    """Parameter interval of an actual shared edge inside a convex polygon."""
    edges=np.roll(poly,-1,axis=0)-poly
    value=np.cross(edges,segment[0]-poly)@normal
    slope=np.cross(edges,segment[1]-segment[0])@normal
    threshold=tolerance*np.linalg.norm(edges,axis=1)
    zero=np.abs(slope)<=threshold*1e-3
    if np.any(zero & (value < -threshold)):
        return 1.,0.
    lo=max(0.,float(np.max(-value[slope>threshold*1e-3]/slope[slope>threshold*1e-3],initial=0.)))
    hi=min(1.,float(np.min(-value[slope<-threshold*1e-3]/slope[slope<-threshold*1e-3],initial=1.)))
    return lo,hi


class SurfaceCircles:
    def __init__(self,mesh,polygons,floor_height=FLOOR_CLEARANCE_M):
        self.mesh,self.polygons=mesh,polygons
        self.tol=mesh.extents.max()*1e-9
        self.allowed=np.zeros(len(mesh.faces),bool)
        self.allowed[list(polygons)]=True
        self.source_area=np.zeros(len(mesh.faces))
        self.frames={};self.flat={}
        for f,p in polygons.items():
            x=p[-1]-p[0];x=x/np.linalg.norm(x)
            basis=np.array([x,-np.cross(mesh.face_normals[f],x)])
            self.frames[f]=(p[0],basis)
            self.flat[f]=Polygon((p-p[0])@basis.T)
            self.source_area[f]=self.flat[f].area
        self.graph=[[] for _ in mesh.faces]
        for (a,b),edge in zip(mesh.face_adjacency,mesh.face_adjacency_edges):
            if not (self.allowed[a] and self.allowed[b]):continue
            segment=mesh.vertices[edge].copy()
            if segment[:,1].max()<=floor_height:continue
            if segment[:,1].min()<floor_height:
                i=int(segment[:,1].argmin());j=1-i
                segment[i]+=(floor_height-segment[i,1])/(segment[j,1]-segment[i,1])*(segment[j]-segment[i])
            if np.linalg.norm(segment[1]-segment[0])<=16*self.tol:continue
            self.graph[a].append((int(b),segment));self.graph[b].append((int(a),segment))
        theta=np.arange(CIRCLE_SIDES)*2*np.pi/CIRCLE_SIDES
        self.unit_disk=np.c_[np.cos(theta),np.sin(theta)]
        self.wrap_cache={}

    def pool(self,seed):
        # This is only a necessary prefilter. The final circle must also pass
        # the pairwise bound between all its faces, including nonneighbors.
        keep=self.allowed & (self.mesh.face_normals@self.mesh.face_normals[seed]>=np.cos(np.deg2rad(MAX_NORMAL_ANGLE_DEG))-NORMAL_DOT_TOL)
        seen={int(seed)};todo=[int(seed)]
        while todo:
            f=todo.pop()
            for other,_ in self.graph[f]:
                if keep[other] and other not in seen:seen.add(other);todo.append(other)
        return np.array(sorted(seen),int)

    def at_radius(self,center,seed,radius,pool=None):
        pool=self.pool(seed) if pool is None else pool
        tri=self.mesh.triangles[pool]
        ids=pool[((tri.min(axis=1)<=center+radius).all(axis=1)) & ((tri.max(axis=1)>=center-radius).all(axis=1))]
        clipped={};full=set();measure={}
        for f in ids:
            p=self.polygons[f]
            if (np.linalg.norm(p-center,axis=1)<=radius).all():
                clipped[int(f)]=p;full.add(int(f));measure[int(f)]=self.source_area[f];continue
            origin,basis=self.frames[f]
            height=(center-origin)@self.mesh.face_normals[f]
            if abs(height)>=radius:continue
            disk_radius=np.sqrt(radius*radius-height*height)
            disk=Polygon((center-origin)@basis.T+disk_radius*self.unit_disk)
            hit=self.flat[f].intersection(disk)
            if hit.is_empty or hit.area<=radius**2*1e-14:continue
            assert hit.geom_type=='Polygon'
            clipped[int(f)]=origin+np.asarray(orient(hit,sign=-1.).exterior.coords)[:-1]@basis
            measure[int(f)]=float(hit.area)
        if seed not in clipped:return {},0.
        selected={int(seed)};todo=[int(seed)]
        while todo:
            f=todo.pop()
            for other,segment in self.graph[f]:
                if other not in clipped or other in selected:continue
                first=(0.,1.) if f in full else edge_interval(clipped[f],self.mesh.face_normals[f],segment,self.tol)
                second=(0.,1.) if other in full else edge_interval(clipped[other],self.mesh.face_normals[other],segment,self.tol)
                shared=(min(first[1],second[1])-max(first[0],second[0]))*np.linalg.norm(segment[1]-segment[0])
                if shared>16*self.tol:selected.add(other);todo.append(other)
        return {f:clipped[f] for f in sorted(selected)},float(sum(measure[f] for f in selected))

    def fit_area(self,center,seed,target):
        pool=self.pool(seed)
        upper=float(self.source_area[pool].sum())
        cap=float(np.linalg.norm(np.concatenate([self.polygons[f] for f in pool])-center,axis=1).max())*(1+1e-9)

        def result(polygons,total,radius,reason):
            reached=abs(total-target)<=target*AREA_REL_TOL
            return polygons,dict(status='area_fitted' if reached else 'area_smaller',
                                 area_status='target_reached' if reached else reason,
                                 area_m2=total,radius_m=radius,
                                 relative_area_error=float(abs(total-target)/target))

        if upper<target*(1-AREA_REL_TOL):
            polygons,total=self.at_radius(center,seed,cap,pool)
            return result(polygons,total,cap,'smaller_connected_area')
        low=0.;high=min(cap,np.sqrt(target/np.pi));below=({},0.,0.)
        for _ in range(30):
            polygons,total=self.at_radius(center,seed,high,pool)
            if total>=target or high>=cap:break
            low=high;below=(polygons,total,high);high=min(cap,high*1.4)
        if total<target*(1-AREA_REL_TOL):
            return result(polygons,total,high,'smaller_circular_area')
        best=(abs(total-target),polygons,total,high)
        if total<=target:below=(polygons,total,high)
        for _ in range(30):
            if best[0]<=target*AREA_REL_TOL:break
            radius=(low+high)/2
            polygons,total=self.at_radius(center,seed,radius,pool)
            if abs(total-target)<best[0]:best=(abs(total-target),polygons,total,radius)
            if total<target:low=radius;below=(polygons,total,radius)
            else:high=radius
        error,polygons,total,radius=best
        if error>target*AREA_REL_TOL:
            # A connected component can gain a distant lobe suddenly. Keep
            # the circle just before that join, never the oversized result.
            polygons,total,radius=below
            return result(polygons,total,radius,'smaller_before_component_join')
        return result(polygons,total,radius,'target_reached')

    def spread(self,polygons):
        key=tuple(sorted(polygons))
        if key not in self.wrap_cache:self.wrap_cache[key]=normal_spread(self.mesh,key)
        return self.wrap_cache[key]

    def fit(self,center,seed,target):
        polygons,report=self.fit_area(center,seed,target)
        before=self.spread(polygons)
        report.update(radius_before_wrap_m=report['radius_m'],area_before_wrap_m2=report['area_m2'],
                      wrap_angle_before_degrees=before['wrap_angle_degrees'],radius_shrunk_for_wrap=False)
        if not polygons or before['wrap_limit_satisfied']:
            return polygons,dict(report,**before)
        pool=self.pool(seed);low=0.;high=report['radius_m']
        best=({},0.,0.);iterations=0
        tolerance=max(32*self.tol,high*1e-7)
        # Ball intersections and their center-connected components are nested.
        # Keep the largest passing radius in the bracket; never prune faces
        # asymmetrically or move the center to satisfy the angle limit.
        for iterations in range(1,41):
            radius=(low+high)/2
            patch,total=self.at_radius(center,seed,radius,pool)
            if patch and self.spread(patch)['wrap_limit_satisfied']:
                low=radius;best=(patch,total,radius)
            else:high=radius
            if high-low<=tolerance:break
        polygons,total,radius=best
        reached=abs(total-target)<=target*AREA_REL_TOL
        report.update(status='area_fitted' if reached else 'area_smaller',
                      area_status='target_reached' if reached else 'smaller_wrap_limit',
                      area_m2=total,radius_m=radius,relative_area_error=float(abs(total-target)/target),
                      radius_shrunk_for_wrap=True,wrap_radius_bracket_m=[low,high],
                      wrap_search_iterations=iterations)
        return polygons,dict(report,**self.spread(polygons))


class LocalClearance:
    def __init__(self,mesh,depth):
        self.mesh,self.depth=mesh,depth
        self.tol=mesh.extents.max()*1e-9
        self.collision=G.Clearance(mesh,self.tol)
        self.offsets,self.valid=G.vertex_offsets(mesh,depth)
        self.cache={}

    def check(self,polygons):
        ids=np.array(list(polygons),int)
        bearing=H.check(self.mesh,ids)
        if not len(ids):return dict(valid=False,status='empty_circle',bearing_direction_check=bearing)
        spread=normal_spread(self.mesh,ids)
        if not spread['wrap_limit_satisfied']:
            return dict(valid=False,status='wrap_angle_exceeded',bearing_direction_check=bearing,**spread)
        if not bearing['passed']:
            return dict(valid=False,status='bearing_direction_rejected',bearing_direction_check=bearing)
        for face in ids:
            p=polygons[face]
            key=(int(face),p.tobytes())
            if key in self.cache:
                result=self.cache[key]
            else:
                normal=np.vstack([p,p+self.depth*self.mesh.face_normals[face]])
                result=dict(valid=True,status='valid')
                if normal[:,1].min()<-self.tol:
                    result=dict(valid=False,status='normal_hits_floor',source_face=int(face))
                else:
                    hit=self.collision.obstruction(normal)
                    if hit!=-1:result=dict(valid=False,status='normal_hits_object',source_face=int(face),obstacle_face=int(hit))
                if result['valid']:
                    if not self.valid[self.mesh.faces[face]].all():
                        result=dict(valid=False,status='no_joined_offset',source_face=int(face))
                    else:
                        head=G.head_cell(self.mesh,p,int(face),self.offsets)
                        hit=self.collision.obstruction(head)
                        if head[:,1].min()<-self.tol:result=dict(valid=False,status='head_hits_floor',source_face=int(face))
                        elif hit!=-1:result=dict(valid=False,status='head_hits_object',source_face=int(face),obstacle_face=int(hit))
                self.cache[key]=result
            if not result['valid']:return dict(result,bearing_direction_check=bearing)
        return dict(valid=True,status='valid',checked_source_faces=len(ids),normal_depth_m=self.depth,
                    bearing_direction_check=bearing)


def build(name,reuse_work_volume=False):
    needs=OUTPUTS/name/pose_name()/'step_1_needs/needs.json'
    domain=ContinuousNeeds.read(needs)
    mesh=domain.mesh
    folder=OUTPUTS/name/pose_name()/W.STAGE;folder.mkdir(parents=True,exist_ok=True)
    work=None
    if POLICY.ENFORCE_PROCESS_ACCESS and reuse_work_volume:
        work=W.WorkVolume.read(folder/'work_volume.json')
        assert W.I.check_report(folder/'work_volume_audit.json')['passed']
        np.testing.assert_array_equal(work.object_mesh.vertices,mesh.vertices)
        np.testing.assert_array_equal(work.object_mesh.faces,mesh.faces)
        np.testing.assert_array_equal(work.original_face_ids,domain.work_ids)
        assert work.half_angle_deg==domain.data['load']['cone_half_deg'] and work.ray_offset==domain.ray_offset
        print(name,'reused unchanged, hash-validated Step 2 work volume and ray audit',flush=True)
    elif POLICY.ENFORCE_PROCESS_ACCESS:
        work=W.WorkVolume.from_domain(domain)
        work_path=work.export(folder,needs)
        WA.run(work_path)
    _,polygons=eligible_polygons(domain)
    centers,center_faces,sampling=surface_centers(polygons,mesh=mesh,with_report=True)
    print(name,'area allocation:',len(centers),'centers in',sampling['chart_count'],'surface charts',flush=True)
    surface=SurfaceCircles(mesh,polygons)
    target=float(mesh.area)*AREA_FRACTION
    depth=float(mesh.extents.max())*DEPTH_FRACTION
    clearance=LocalClearance(mesh,depth)
    work_clearance=WC.ContactClearance(mesh,depth,work,clearance.offsets)
    triangles=[];sources=[];offsets=[0];records=[]
    start=time.monotonic()
    for i,(center,face) in enumerate(zip(centers,center_faces)):
        patch,fitted=surface.fit(center,int(face),target)
        checked=clearance.check(patch)
        if checked['valid']:
            checked=WC.apply_constraint(checked,work_clearance.check(patch))
        else:
            checked['work_volume_check']=dict(passed=False,classification='not_checked',reason='Local geometry already rejected')
        pieces=[fan(p) for p in patch.values()]
        for f,piece in zip(patch,pieces):triangles.extend(piece);sources.extend([f]*len(piece))
        offsets.append(len(triangles))
        records.append(dict(id=f'C{i+1:03d}',index=i,kind='candidate_circle',center_m=center.tolist(),
                            center_face=int(face),target_area_m2=target,
                            area_fraction=float(fitted['area_m2']/mesh.area),**fitted)|checked)
        if (i+1)%10==0:print(name,'target-1% circles',i+1,f'/{len(centers)}; valid',sum(r['valid'] for r in records),
                             'seconds',round(time.monotonic()-start,1),flush=True)
    folder=OUTPUTS/name/pose_name()/'step2_local_support';folder.mkdir(parents=True,exist_ok=True)
    triangles=np.asarray(triangles).reshape(-1,3,3)
    np.savez_compressed(folder/'circles.npz',triangles=triangles,source_faces=np.asarray(sources,int),
                        offsets=np.asarray(offsets,np.int64),triangle_areas=areas(triangles),
                        centers_m=centers,center_faces=center_faces,
                        radius_m=np.array([r['radius_m'] if r['radius_m'] is not None else 0. for r in records]),
                        valid=np.array([r['valid'] for r in records]),needs_sha256=np.array(sha256(needs)))
    report=dict(object=name,geometry='one_sided_center_connected_surface_intersection_with_ball',
                center_count=CENTER_COUNT,valid_count=sum(r['valid'] for r in records),
                smaller_valid_count=sum(r['valid'] and r['area_status']!='target_reached' for r in records),
                wrap_shrunk_count=sum(r['radius_shrunk_for_wrap'] for r in records),
                total_object_area_m2=float(mesh.area),target_area_fraction=AREA_FRACTION,
                max_wrap_angle_degrees=MAX_WRAP_ANGLE_DEG,normal_dot_tolerance=NORMAL_DOT_TOL,
                bearing_direction_constraint=H.definition(),
                bearing_direction_rejected_count=sum(not r['bearing_direction_check']['passed'] for r in records),
                relative_area_tolerance=AREA_REL_TOL,circle_sides=CIRCLE_SIDES,normal_depth_m=depth,
                center_method=sampling['method'],sampling=sampling,
                radius_rule='Target 1% of true surface area. Allow smaller connected circles; shrink the same-center radius until the maximum pairwise normal angle is at most 90 degrees.',
                candidate_rule='Sample non-working surface, fit connected circles with whole-patch wrap <=90 degrees, allow every contact orientation, and check actual head clearance against object and floor. Process-access volume is omitted by the current model.',
                process_access_enforced=POLICY.ENFORCE_PROCESS_ACCESS,
                work_volume_record='work_volume.json' if work is not None else None,
                work_volume_reused=bool(work is not None and reuse_work_volume),
                work_volume_rejected_count=sum(r['status'].startswith('work_volume_') for r in records),
                patches=records,provenance=dict(needs_sha256=sha256(needs),geometry_signature=geometry_signature(domain),
                circles_sha256=sha256(folder/'circles.npz'),
                inputs=W.I.hashes([folder/f for f in ['work_volume.json','work_volume.npz','work_volume_audit.json']] if work is not None else []),
                code={p.name:sha256(p) for p in [Path(__file__),Path(POLICY.__file__),Path(H.__file__),Path(S.__file__),Path(G.__file__),Path(W.__file__),Path(W.V.__file__),Path(WC.__file__),Path(WC.D.__file__)]},
                clearance_code=WC.D.code_hashes()))
    save(folder/'circles.json',report)
    save(folder/'centers.json',dict(object=name,count=len(centers),method=sampling['method'],points=[dict(id=r['id'],position_m=r['center_m'],source_face=r['center_face'],valid=r['valid'],status=r['status']) for r in records]))
    print(name,'DONE:',report['valid_count'],'valid circular candidates',flush=True)
    return report


def rebind(name):
    """Only already-current eligibility can be reused; changed inputs need a rebuild."""
    try:
        _,_,report=read(name)
    except (KeyError,AssertionError,RuntimeError,OSError) as error:
        raise ValueError(f'{name}: Step 2 validity now depends on the access volume; rebuild Step 2 after input or code changes') from error
    print(name,'Step 2 eligibility is already current',flush=True)
    return report


def read(name):
    folder=OUTPUTS/name/pose_name()/'step2_local_support';needs=OUTPUTS/name/pose_name()/'step_1_needs/needs.json'
    report=json.loads((folder/'circles.json').read_text())
    assert report['provenance']['needs_sha256']==sha256(needs)
    assert report['provenance']['geometry_signature']==geometry_signature(ContinuousNeeds.read(needs))
    assert report['provenance']['circles_sha256']==sha256(folder/'circles.npz')
    for filename,digest in report['provenance']['code'].items():assert sha256(HERE/filename)==digest
    W.I.check_hashes(report['provenance']['inputs'])
    W.I.check_hashes(report['provenance']['clearance_code'])
    assert report['process_access_enforced']==POLICY.ENFORCE_PROCESS_ACCESS
    if POLICY.ENFORCE_PROCESS_ACCESS:
        W.I.check_report(folder/'work_volume.json')
        assert W.I.check_report(folder/'work_volume_audit.json')['passed']
    with np.load(folder/'circles.npz') as z:data={key:z[key] for key in z.files}
    expected=np.array([r['valid'] for r in report['patches']],bool)
    np.testing.assert_array_equal(data['valid'],expected)
    assert all(r['work_volume_check']['passed'] for r in report['patches'] if r['valid'])
    assert report['bearing_direction_constraint']==H.definition()
    assert all(r['bearing_direction_check']['passed'] for r in report['patches'] if r['valid'])
    return ContinuousNeeds.read(needs),SimpleNamespace(**data),report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    parser.add_argument('--rebind',action='store_true',help='Validate already-current geometry; changed inputs require rebuilding Step 2')
    parser.add_argument('--reuse-work-volume',action='store_true',help='Reuse only unchanged, hash-validated process-access geometry and its existing ray audit')
    args=parser.parse_args()
    for name in args.objects or OBJECTS:
        if args.rebind:rebind(name)
        else:build(name,reuse_work_volume=args.reuse_work_volume)
