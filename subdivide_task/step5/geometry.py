"""Current-pose adapter for Co-optimize's actual-solid Step5 guards.

The ground polygon is virtual. Only small pads and their legal connectors are
material. No perimeter ring/slab fallback is constructed.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
from scipy.spatial import cKDTree
from subdivide_task.step3.legacy import kernel as K, COOPT
for folder in ('step5.2','step5.3','step5'):
    sys.path.insert(0,str(COOPT/folder))
from subdivide_task.step4.export_geometry import exclusions
from subdivide_task.common.wrenches import endpoint_demands
from subdivide_task.step1.common import write_json
from deterministic_base import (Builder, solid, unpack, volume, union, enclosure,
                                transform, closest, components, box, SCALE)
from continuous_legs import capsule
from prune_support import ForceCertificate

RETAINED_VOLUME_FRACTION=.55


def physical_component_count(value):
    return sum(float(p.volume())>0 for p in value.decompose())


def save_mesh(mesh,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    K.D.export_exact_obj(mesh,path)
    return K.trimesh.load(path,force='mesh',process=False)


def execution_rows(model,layout):
    rows=[]
    for k,raw in zip(layout.active,model.base_oracle.rows(layout)):
        state=model.execution(layout,k)
        T=np.asarray(state['world_from_support']);U=np.asarray(state['world_from_object'])
        body=transform(model.mesh.vertices,U)
        D=np.asarray(state['center_translation_m'])
        rows.append(dict(pose=model.poses[k],fixture_transform=T,object_transform=U,
            hull=np.c_[raw['hull']+D[:2],np.zeros(len(raw['hull']))],
            object_ground=body[np.abs(body[:,2])<=1e-9,:2]))
    return rows


def construction_rows(model,layout):
    """Build in search gauge: horizontal recentering must not duplicate feet.

    Execution rows are retained for acceptance/evaluation. Translating the
    complete object, fixture and demand hull preserves relative floor demands.
    """
    return [dict(pose=r['pose'],fixture_transform=r['T'],object_transform=r['U'],
        object_ground=r['own'],hull=np.c_[r['hull'],np.zeros(len(r['hull']))])
        for r in model.base_oracle.rows(layout)]


def occupied_bounds(body,mesh,rows):
    points=np.vstack([p for row in rows for p in
        (transform(body.vertices,row['object_transform']),transform(mesh.vertices,row['fixture_transform']))])
    return np.array([points.min(0),points.max(0)])


class SparseFeet(Builder):
    """Port deterministic_base ground hull-corner pads, with rounded links.

    Compact pads must enclose every task demand. The original broad slab
    fallback is forbidden; inability to construct pads is an explicit failure.
    """
    def ground(self):
        super().ground()
        record=self.history[-1]
        if not record['compact_feet_used']:
            raise ValueError('Sparse feet cannot enclose all pressure hulls; slab fallback disabled')
        record.update(perimeter_ring_generated=False,full_slab_fallback_allowed=False)

    def regularize_material(self,value):
        # Removing tiny detached material must preserve inward cavity shells.
        for part in value.decompose():
            if 0.<float(part.volume())*SCALE**3<=1e-13:value=value-part
        return value

    def connect(self):
        routes=None
        for iteration in range(256):
            parts=components(self.body)
            if len(parts)<=1:return
            pairs=sorted(((*closest(parts[i],parts[j]),i,j) for i in range(len(parts)) for j in range(i)),key=lambda x:x[0])
            accepted=None
            for radius in (.004,.006,.008,.012,.016):
                for distance,a,b,i,j in pairs[:12]:
                    proposal=capsule(a,b,radius);link=proposal^self.allowed
                    for piece in components(link):
                        if volume(piece-proposal)>1e-10 or volume(piece-self.allowed)>1e-10:continue
                        if volume(piece^parts[i])<=1e-13 or volume(piece^parts[j])<=1e-13:continue
                        candidate=self.body+piece
                        if len(components(candidate))>=len(parts):continue
                        cost=volume(candidate)-volume(self.body)
                        if accepted is None or cost<accepted[0]:accepted=(cost,candidate,distance,radius)
                if accepted is not None:break
            if accepted is None:
                # Route thin links around obstacles instead of expanding a
                # bounding box until it fills a large fraction of free space.
                from .surface_route import SurfaceRoutes
                if routes is None:routes=SurfaceRoutes(unpack(self.allowed))
                for length,i,j,path in routes.paths([unpack(p) for p in parts]):
                    for radius in (.004,.006,.008):
                        segments=[capsule(a,b,radius) for a,b in zip(path,path[1:]) if np.linalg.norm(a-b)>1e-10]
                        proposal=union(segments)
                        link=proposal^self.allowed
                        # Keep holes and all signed boundary components. A
                        # detached positive outer shell can fill a cavity.
                        if volume(link-proposal)>1e-10 or volume(link-self.allowed)>1e-10:continue
                        candidate=self.body+link
                        if len(components(candidate))>=len(parts):continue
                        cost=volume(candidate)-volume(self.body)
                        accepted=(cost,candidate,length,radius)
                        self.history.append(dict(operation='boundary_surface_route',
                            path_m=path.tolist(),radius_m=radius,added_cm3=cost*1e6))
                        break
                    if accepted is not None:break
                if accepted is None:raise ValueError('Cannot route sparse feet through certified free space')
            cost,self.body,distance,radius=accepted
            self.history.append(dict(operation='rounded_local_leg',radius_m=radius,
                span_m=distance,added_cm3=cost*1e6,component_count=len(components(self.body))))
        raise ValueError('Sparse-foot connectors exhausted')


class Validator:
    """Exact exported geometry + all samples + continuous outer-cone cover."""
    def __init__(self,model,layout,witness,reference):
        self.model=model;self.layout=layout;self.rows=execution_rows(model,layout)
        self.data=dict(body=model.mesh)
        self.witness=solid(witness)
        work,bodies,exits=exclusions(model,layout,reference)
        self.forbidden=dict(work=work,object=bodies,exit=exits)
        self.checker=ForceCertificate(model.poses,layout,model.mesh,self.rows,model.name,states=model.states)
        self.endpoints={k:endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale for k in layout.active}
        self.endpoint_cache={}

    def check(self,mesh,previous=None,bounds=None,**kwargs):
        if not np.isfinite(mesh.vertices).all() or not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Invalid actual exported solid')
        value=solid(mesh);cover=enclosure(mesh,self.rows)
        actual=occupied_bounds(self.model.mesh,mesh,self.rows)
        floors=[float(transform(mesh.vertices,r['fixture_transform'])[:,2].min()) for r in self.rows]
        if min(floors)<-1e-8:raise ValueError('Actual support crosses an installed floor')
        expansion=0. if bounds is None else float(max(0,np.max(bounds[0]-actual[0]),np.max(actual[1]-bounds[1])))
        checks=dict(components=physical_component_count(value),material_cm3=abs(float(mesh.volume))*1e6,
            outside_m3=volume(value-self.witness),added_m3=volume(value-previous) if previous is not None else 0.,
            bbx_l=float(np.prod(actual[1]-actual[0])*1000),bounds_world_m=actual.tolist(),
            minimum_support_height_m=floors,bbx_expansion_m=expansion,floor_passed=all(r['passed'] for r in cover),pressure_enclosure=cover)
        if checks['components']!=1 or checks['outside_m3']>1e-10 or checks['added_m3']>1e-10 or expansion>1e-8 or not checks['floor_passed']:
            raise ValueError('Solid/floor/subset guard: '+str({k:v for k,v in checks.items() if k not in ['pressure_enclosure','bounds_world_m']}))
        if bounds is not None and checks['bbx_l']>np.prod(bounds[1]-bounds[0])*1000+1e-8:
            raise ValueError('Occupied BBX increased')
        # Exclusions are rebuilt from final stock once, then all cuts are
        # subsets. Checking the same finite swept solids preserves this guard.
        overlaps={name:[volume(value^x) for x in pieces] for name,pieces in self.forbidden.items()}
        if max(v for group in overlaps.values() for v in group)>1e-10:
            raise ValueError('Work/object/complete-exit exclusion guard: '+str(overlaps))
        proof=self.checker.evaluate_candidate(mesh)
        if not proof['passed'] or len(proof['masks'])!=len(self.rows) or any(len(v)!=len(self.model.tasks[k].targets) or not v.all() for k,v in proof['masks'].items()):
            raise ValueError('Original load guard: '+str(proof['counts']))
        masks={}
        for k in self.layout.active:
            rays=np.ascontiguousarray(proof['supplies'][k]);key=(int(k),hashlib.sha256(rays.tobytes()).digest())
            if key not in self.endpoint_cache:self.endpoint_cache[key]=K.J.classify(rays,self.endpoints[k])[0]
            masks[k]=self.endpoint_cache[key]
            if not masks[k].all():raise ValueError('Continuous cone guard: '+self.model.poses[k])
        proof['endpoint_masks']=masks
        checks.update(passed=True,watertight=True,winding_consistent=True,serialized_geometry_checked=True,
            counts=proof['counts'],continuous_endpoint_counts={self.model.poses[k]:dict(passed=int(m.sum()),total=len(m)) for k,m in masks.items()},
            physical_exclusion_overlap_m3=overlaps)
        return value,proof,checks


def ground_contact_metrics(mesh,rows):
    """Measure real coplanar contact patches separately from their virtual hull."""
    from shapely.geometry import Polygon,MultiPoint
    from shapely.ops import unary_union
    result=[]
    for row in rows:
        world=transform(mesh.vertices,row['fixture_transform'])
        triangles=world[mesh.faces]
        mask=np.max(np.abs(triangles[:,:,2]),axis=1)<=1e-8
        polygons=[Polygon(t[:,:2]) for t in triangles[mask]]
        contact=unary_union([p for p in polygons if p.area>1e-16])
        regions=list(contact.geoms) if hasattr(contact,'geoms') else [contact]
        feet=world[np.abs(world[:,2])<=1e-8,:2]
        hull=MultiPoint(np.vstack([feet,row['object_ground']])).convex_hull
        result.append(dict(pose=row['pose'],support_contact_area_cm2=float(contact.area*1e4),
            ground_hull_area_cm2=float(hull.area*1e4),
            support_contact_to_ground_hull_area_percent=float(contact.area/hull.area*100),
            coplanar_contact_region_count=sum(p.area>1e-12 for p in regions),
            contact_region_count_area_threshold_m2=1e-12,
            virtual_convex_hull_is_not_material=True))
    return result
