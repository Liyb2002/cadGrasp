"""Deterministic material construction inside Step4's saved legal component.

No placement search, force solver, or obstacle reconstruction. Ground slabs
provide enclosing contact patches; clipped local bridges connect components.
All coordinates are metres, Manifold coordinates use the original 0.2m scale.
"""
from itertools import product
import time
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import MultiPoint, Point

SCALE=.2
VOLUME_TOL=1e-13
WIDTH=.005

def solid(mesh):
    result=md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices/SCALE),
                               np.ascontiguousarray(mesh.faces,dtype=np.uint32)))
    if result.status()!=md.Error.NoError: raise ValueError(f'Invalid input solid: {result.status()}')
    return result

def unpack(value):
    m=value.to_mesh64()
    return trimesh.Trimesh(np.array(m.vert_properties[:,:3])*SCALE,np.array(m.tri_verts),process=False)

def volume(value): return max(0.,float(value.volume())*SCALE**3)


def added_material_statistics(material_m3,legal_nominal_m3,retained_loss_m3,outside_allowed_m3):
    """Report added volume without constructing a redundant Boolean boundary.

    N is the legal nominal material (or the full original when its illegal
    volume is below the original tolerance). Thus N is a subset of original,
    and original minus N lies outside allowed. For the constructed material A,
    |A minus N| = |A| - |N| + |N minus A|. Subtracting its possible overlap
    with original minus N gives the bounds below. These are statistics only;
    all geometry, acceptance tests and total-material costs remain unchanged.
    """
    upper=max(0.,material_m3-legal_nominal_m3+retained_loss_m3)
    lower=max(0.,upper-outside_allowed_m3)
    return dict(added_material_cm3=upper*1e6,
        added_material_bounds_cm3=[lower*1e6,upper*1e6],
        added_material_measure=('set_volume_identity' if outside_allowed_m3==0.
                                else 'set_volume_identity_upper_bound'),
        added_material_statistics_only=True)

def union(values):
    return md.Manifold.batch_boolean(values,md.OpType.Add) if values else md.Manifold()

def box(low,high):
    return md.Manifold.cube(tuple((np.asarray(high)-low)/SCALE)).translate(tuple(np.asarray(low)/SCALE))

def transform(points,T): return np.asarray(points)@T[:3,:3].T+T[:3,3]

def moved(value,T):
    matrix=np.asarray(T[:3]).copy();matrix[:,3]/=SCALE
    return value.transform(matrix)

def enclosure(mesh,rows):
    records=[]
    for r in rows:
        world=transform(mesh.vertices,r['fixture_transform'])
        feet=world[np.abs(world[:,2])<=1e-8,:2]
        all_points=np.vstack([feet,r['object_ground']])
        boundary=MultiPoint(all_points).convex_hull if len(all_points) else MultiPoint([])
        gap=max(boundary.distance(Point(p[:2])) for p in r['hull']) if not boundary.is_empty else 1e3
        records.append(dict(pose=r['pose'],passed=bool(gap<=1e-8),gap_m=float(gap),
            foot_vertex_count=len(feet),ground_contact_hull_xy_m=(np.asarray(boundary.exterior.coords)[:-1].tolist()
                if boundary.geom_type=='Polygon' else [])))
    return records

def components(value):
    return [p for p in value.decompose() if volume(p)>VOLUME_TOL]

def closest(a,b):
    av,bv=unpack(a).vertices,unpack(b).vertices
    dist,ids=cKDTree(bv).query(av)
    k=int(np.argmin(dist))
    return float(dist[k]),av[k],bv[ids[k]]

def bridge(a,b,width):
    offsets=np.array(list(product([-.5,.5],repeat=3)))*width
    cloud=np.vstack([a+offsets,b+offsets])
    h=ConvexHull(cloud)
    return solid(trimesh.Trimesh(cloud,h.simplices,process=True).convex_hull)

class Builder:
    def added_material_record(self,outside,retained):
        return added_material_statistics(volume(self.body),volume(self.legal_nominal),retained,outside)

    def __init__(self,fixture,witness,rows,width=WIDTH):
        self.original=solid(fixture);self.allowed=solid(witness);self.rows=rows;self.width=width
        self.history=[];self.began=time.monotonic()
        # Only already-illegal presentation material is removed, never a pose.
        self.removed=volume(self.original-self.allowed)
        # Avoid recutting an already legal solid along coincident boundaries:
        # a redundant intersection can create numerical disconnected slivers.
        self.legal_nominal=(self.original if self.removed<=VOLUME_TOL
                            else self.original^self.allowed)
        self.body=self.legal_nominal
        clouds=[fixture.vertices]
        clouds.extend(transform(r['hull'],np.linalg.inv(r['fixture_transform'])) for r in rows)
        joined=np.vstack(clouds);self.low=joined.min(0);self.high=joined.max(0)
        self.witness_bounds=witness.bounds

    def ground(self):
        distinct={np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in self.rows}
        for margin in [self.width,2*self.width,4*self.width,8*self.width,16*self.width,32*self.width,64*self.width,128*self.width]:
            domain=box(self.low-margin,self.high+margin)^self.allowed
            slabs=[]
            for T in distinct.values():
                # The saved witness already lies above EVERY installed floor.
                slab=domain.trim_by_plane(tuple(-T[2,:3]),float((T[2,3]-self.width)/SCALE))
                # A common-coordinate box can have distant corners when
                # rotated onto this floor. Bound feet around this state's
                # actual pressure hulls, enlarging only if enclosure needs it.
                required=np.vstack([r['hull'][:,:2] for r in self.rows
                    if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)])
                low,high=required.min(0)-margin,required.max(0)+margin
                for axis in range(2):
                    slab=slab.trim_by_plane(tuple(T[axis,:3]),float((low[axis]-T[axis,3])/SCALE))
                    slab=slab.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-high[axis])/SCALE))
                slabs.append(slab)
            feet=union(slabs);candidate=self.body+feet
            cover=enclosure(unpack(candidate),self.rows)
            if all(r['passed'] for r in cover):
                # The enclosing polygon is virtual: its interior need not be
                # filled. Retain only essential vertices as small physical feet.
                patches=[];locations=[]
                body_mesh=unpack(self.body)
                for T,slab in zip(distinct.values(),slabs):
                    world=transform(unpack(slab).vertices,T)
                    xy=np.unique(np.round(world[np.abs(world[:,2])<=1e-8,:2],12),axis=0)
                    if len(xy)<3:continue
                    hull=ConvexHull(xy);selected=list(xy[hull.vertices])
                    relevant=[r for r in self.rows if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)]
                    ground=transform(body_mesh.vertices,T)
                    existing=ground[np.abs(ground[:,2])<=1e-8,:2]
                    # Remove far feet first, keeping all task hulls enclosed.
                    order=sorted(range(len(selected)),key=lambda k:(-float(cKDTree(transform(body_mesh.vertices,T)[:,:2]).query(selected[k])[0]),k))
                    active=set(range(len(selected)))
                    for k in order:
                        remaining=np.array([selected[j] for j in sorted(active-{k})]).reshape(-1,2)
                        ok=True
                        for row in relevant:
                            combined=np.vstack([remaining,existing,row['object_ground']])
                            region=MultiPoint(combined).convex_hull if len(combined) else MultiPoint([])
                            if region.is_empty or any(region.distance(Point(p[:2]))>1e-8 for p in row['hull']):ok=False;break
                        if ok:active.remove(k)
                    for k in sorted(active):
                        p=selected[k]
                        foot=moved(box(np.array([p[0]-self.width/2,p[1]-self.width/2,0]),
                            np.array([p[0]+self.width/2,p[1]+self.width/2,self.width])),np.linalg.inv(T))^self.allowed
                        if volume(foot)>VOLUME_TOL:patches.append(foot);locations.append(dict(world_xy_m=p.tolist(),fixture_transform=T.tolist()))
                compact=self.body+union(patches)
                compact_cover=enclosure(unpack(compact),self.rows)
                use_compact=all(r['passed'] for r in compact_cover)
                self.body=compact if use_compact else candidate
                self.history.append(dict(operation='ground_patches',margin_m=margin,
                    compact_feet_used=use_compact,foot_locations=locations,foot_count=len(patches),
                    volume_cm3=volume(union(patches) if use_compact else feet)*1e6,
                    component_count=len(components(self.body))))
                return
        raise ValueError('Saved legal witness cannot construct demand-enclosing ground patches')

    def clip_proposal(self,proposal):
        return proposal^self.allowed

    def regularize_material(self,value):
        return union(components(value))

    def connect(self):
        for iteration in range(256):
            parts=components(self.body)
            if len(parts)<=1:return
            pairs=sorted(((*closest(parts[i],parts[j]),i,j) for i in range(len(parts)) for j in range(i)),
                         key=lambda x:(x[0],x[-2],x[-1]))
            accepted=None
            # Deterministic local tube / box family; no physical variables move.
            for multiplier in [1,2,4,8,16,32,64,128,256]:
                width=self.width*multiplier
                for distance,p,q,i,j in pairs[:12]:
                    proposal=bridge(p,q,width) if multiplier==1 else box(np.minimum(p,q)-width/2,np.maximum(p,q)+width/2)
                    clipped=self.clip_proposal(proposal)
                    for piece in components(clipped):
                        if volume(piece^parts[i])<=VOLUME_TOL or volume(piece^parts[j])<=VOLUME_TOL:continue
                        candidate=self.body+piece
                        if len(components(candidate))>=len(parts):continue
                        cost=volume(candidate)-volume(self.body)
                        if accepted is None or cost<accepted[0]:
                            accepted=(cost,candidate,distance,width)
                if accepted is not None:break
            if accepted is None:
                raise ValueError('Cannot connect material inside the saved common component')
            cost,self.body,distance,width=accepted
            self.history.append(dict(operation='clipped_local_bridge',iteration=iteration,
                width_m=width,span_m=distance,added_cm3=cost*1e6,component_count=len(components(self.body))))
        raise ValueError('Connection construction exceeded its deterministic action bound')

    def run(self):
        self.ground();self.connect()
        # Boolean boundary coincidences can emit essentially zero-volume
        # disconnected shells. Remove exactly the same sub-tolerance pieces
        # already excluded from component counting; do not move any geometry.
        regularized=self.regularize_material(self.body)
        discarded=volume(self.body-regularized)
        self.body=regularized
        mesh=unpack(self.body)
        cover=enclosure(mesh,self.rows)
        outside=volume(self.body-self.allowed)
        retained_loss=volume(self.legal_nominal-self.body)
        passed=all(r['passed'] for r in cover) and len(components(self.body))==1 and outside<=1e-10 and retained_loss<=1e-10
        record=dict(passed=passed,connected=len(components(self.body))==1,
            per_pose=cover,material_cm3=volume(self.body)*1e6,
            **self.added_material_record(outside,retained_loss),
            removed_illegal_nominal_material_cm3=self.removed*1e6,
            discarded_zero_volume_shells_cm3=discarded*1e6,
            outside_saved_legal_component_cm3=outside*1e6,
            retained_legal_step4_material_loss_cm3=retained_loss*1e6,
            floor_pressure_enclosure_passed=all(r['passed'] for r in cover),
            watertight=bool(mesh.is_watertight),history=self.history,
            deterministic=True,step4_layout_changed=False,force_solver_run=False,
            legal_component_reconstructed=False,seconds=time.monotonic()-self.began)
        return mesh,record
