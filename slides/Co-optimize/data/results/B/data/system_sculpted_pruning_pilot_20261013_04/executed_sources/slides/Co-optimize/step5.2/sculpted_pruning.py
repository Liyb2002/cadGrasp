"""Round real convex edges by deletion, protecting contacts and occupied BBX.

Unlike rounded stock retention, the old shape is not a mandatory scaffold.
Actual surface contacts, essential core and BBX extrema are protected locally.
All proposed geometry is intersected with its original material, never inflated.
"""
import time
import numpy as np
import manifold3d as md

from prune_support import Pruner, material_components
from deterministic_base import solid, unpack, volume, union, SCALE, transform, enclosure
from whole_search.common import C, transform_mesh


def sharp_edges(mesh, angle=35):
    mesh=mesh.copy();mesh.merge_vertices(digits_vertex=12)
    angles=mesh.face_adjacency_angles
    edges=mesh.face_adjacency_edges
    lengths=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    selected=angles>np.deg2rad(angle)
    return dict(threshold_degrees=angle,sharp_edge_length_m=float(lengths[selected].sum()),
                sharp_edge_count=int(selected.sum()),mesh_triangles=len(mesh.faces))


def contact_guard(mesh,certificate,depth=.00045,pose_ids=None):
    prisms=[];count=0
    for k in certificate.layout.active:
        if pose_ids is not None and k not in pose_ids:continue
        Q=certificate.layout.placements[k]
        direction=Q[:3,:3].T@certificate.layout.directions[k]
        ids=certificate.allowed[k][certificate.mesh.face_normals[certificate.allowed[k]]@direction<=1e-9]
        triangles,sources=C.contact_boundary(transform_mesh(certificate.mesh,Q),mesh,ids)
        normals=certificate.mesh.face_normals[sources]@Q[:3,:3].T
        for triangle,normal in zip(triangles,normals):
            prism=md.Manifold.hull_points(np.vstack([triangle-depth*.1*normal,triangle+depth*normal])/SCALE)
            if not prism.is_empty():prisms.append(prism);count+=1
    return union(prisms),count


def bbox_guard(mesh,frames,object_cloud,bounds,radius=.001):
    centers=[]
    for T in frames:
        world=transform(mesh.vertices,T)
        for axis in range(3):
            for side in [0,1]:
                edge=bounds[side,axis]
                object_edge=object_cloud[:,axis].min()if side==0 else object_cloud[:,axis].max()
                if abs(edge-object_edge)<=1e-9:continue
                extreme=world[:,axis].min()if side==0 else world[:,axis].max()
                if abs(extreme-edge)>1e-9:continue
                ids=np.flatnonzero(abs(world[:,axis]-edge)<=1e-10)
                # One actual extremum suffices; choose its broadest local region.
                if len(ids):centers.append(mesh.vertices[ids[len(ids)//2]])
    unique=np.unique(np.round(centers,13),axis=0)if centers else []
    sphere=md.Manifold.sphere(radius/SCALE,16)
    return union([sphere.translate(tuple(p/SCALE))for p in unique]),len(unique)


def edge_fillet_knife(a,b,n1,n2,radius):
    direction=b-a;length=float(np.linalg.norm(direction));direction/=length
    bisector=n1+n2;bisector/=np.linalg.norm(bisector)
    cosine=float(n1@bisector)
    if cosine<.15:return md.Manifold()
    midpoint=(a+b)/2;center=midpoint-radius/cosine*bisector
    tangent1=center+radius*n1;tangent2=center+radius*n2
    span=length/2+.00005
    points=np.vstack([np.array([midpoint,tangent1,tangent2])+sign*span*direction for sign in [-1,1]])
    brush=md.Manifold.hull_points(points/SCALE)
    across=np.cross(direction,bisector);across/=np.linalg.norm(across)
    matrix=np.c_[np.column_stack([bisector,across,direction]),center/SCALE]
    cylinder=md.Manifold.cylinder(2*span/SCALE,radius/SCALE,circular_segments=48,center=True).transform(matrix)
    return brush-cylinder


class SculptedPruner(Pruner):
    def __init__(self,*args,mandatory=None,compaction_guard=None,final_support=False,future_legs=None,
                 guide_reference=None,radii=(.0012,.0008,.00055),**kwargs):
        kwargs['grids']=()
        super().__init__(*args,**kwargs)
        self.mandatory=solid(mandatory)if mandatory is not None else md.Manifold()
        self.coarse_guard=solid(compaction_guard)if compaction_guard is not None else md.Manifold()
        self.final_support=final_support;self.radii=radii
        self.future_legs=solid(future_legs)if future_legs is not None else None
        self.guide_reference=guide_reference

    def bounds(self,mesh):
        cloud=np.vstack([self.object_cloud]+[transform(mesh.vertices,T)for T in self.frames])
        return np.array([cloud.min(0),cloud.max(0)])

    def run(self):
        self.proof=self.certificate.evaluate(self.fixture);self.best_mesh=self.fixture
        if not self.proof['passed']:raise ValueError('Frozen input fails actual original demands')
        target=self.bounds(self.fixture)
        extremes,extreme_count=bbox_guard(self.fixture,self.frames,self.object_cloud,target)
        extreme_stock=extremes^self.original
        baseline_quality=sharp_edges(self.fixture);selected=[];compaction_radius=None
        def geometry_ok(value):
            from material_builder_closed import regularize_external_boundaries
            value,_=regularize_external_boundaries(value)
            if value.is_empty()or len(material_components(value))!=1:return False
            mesh=unpack(value)
            if volume(self.mandatory-value)>1e-10 or not np.allclose(self.bounds(mesh),target,atol=1e-8,rtol=0):return False
            if self.final_support and not all(r['passed']for r in enclosure(mesh,self.rows)):return False
            if self.future_legs is not None:
                future=value+self.future_legs
                if len(material_components(future))!=1:return False
                if not all(r['passed']for r in enclosure(unpack(future),self.rows)):return False
            return True
        if self.guide_reference is not None:
            anchor=solid(self.guide_reference);shape=anchor.simplify(.00015/SCALE)
            for padding in [.003,.0045]:
                expanded=shape.minkowski_sum(md.Manifold.sphere(padding/SCALE,24))
                candidate=(anchor+expanded+extreme_stock+self.coarse_guard)^self.body
                if geometry_ok(candidate)and self.accept(candidate,'compact_rounded_block_'+str(padding)):
                    compaction_radius=padding;break
        # Work on actual convex edges; avoid a global opening of thin walls.
        reference=self.best_mesh.copy();reference.merge_vertices(digits_vertex=12)
        pairs=reference.face_adjacency;edges=reference.face_adjacency_edges
        angles=reference.face_adjacency_angles;convex=reference.face_adjacency_convex
        endpoints=reference.vertices[edges];lengths=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1)
        ids=np.flatnonzero(convex&(angles>np.deg2rad(40))&(angles<np.deg2rad(150))&(lengths>.002))
        ids=sorted(ids,key=lambda k:float(lengths[k]*angles[k]),reverse=True)[:80]
        for k in ids:
            if self.attempts>=self.budget:break
            a,b=endpoints[k];n1,n2=reference.face_normals[pairs[k]]
            radius=min(self.radii[0],lengths[k]*.18)
            for actual_radius in [radius,radius*.6]:
                if actual_radius<.0002:continue
                knife=edge_fillet_knife(a,b,n1,n2,actual_radius)
                if knife.is_empty():continue
                candidate=(self.body-knife)+(extreme_stock^self.body)
                if not geometry_ok(candidate):continue
                before=sharp_edges(self.best_mesh)['sharp_edge_length_m']
                quality=sharp_edges(unpack(candidate))
                if quality['sharp_edge_length_m']>before+1e-6:continue
                if self.accept(candidate,'local_edge_fillet_'+str(round(float(actual_radius),7))):
                    selected.append(float(actual_radius));break
        actual=solid(self.best_mesh);bounds=self.bounds(self.best_mesh)
        if (volume(actual-self.original)>1e-10 or volume(self.mandatory-actual)>1e-10 or
                not np.allclose(bounds,target,atol=1e-8,rtol=0)):
            raise ValueError('Sculpted output violates original-stock, core or locked BBX')
        record=dict(passed=True,stage_type='support_pruning',pruning_style='sculpted_fillet',
            bbx_first=True,input_bbx_l=self.bbx(self.original),retained_bbx_l=self.bbx(actual),
            locked_bbx_bounds_m=target.tolist(),locked_bbx_unchanged=True,
            input_material_cm3=volume(self.original)*1e6,retained_material_cm3=volume(actual)*1e6,
            removed_material_cm3=volume(self.original-actual)*1e6,added_material_cm3=volume(actual-self.original)*1e6,
            component_count=len(material_components(actual)),connected=True,watertight=bool(self.best_mesh.is_watertight),
            all_original_demands_passed=True,counts=self.proof['counts'],contact_triangles=self.proof['contact_triangles'],
            force_required_isolated_components=0,grid_cell_deletion_used=False,reference_scaffold_used=False,
            rounding_radius_m=max(selected)if selected else None,requested_radii_m=list(self.radii),
            accepted_edge_fillets=len(selected),candidate_convex_edges=len(ids),edge_fillet_radii_m=selected,
            compaction_radius_m=compaction_radius,protected_bbox_extrema=extreme_count,
            sharp_edges_before=baseline_quality,sharp_edges_after=sharp_edges(self.best_mesh),
            mandatory_core_loss_cm3=volume(self.mandatory-actual)*1e6,attempts=self.attempts,history=self.history,
            force_solver_run=True,layout_changed=False,demand_resampling=False,global_minimum_claimed=False,
            seconds=time.monotonic()-self.began)
        return self.best_mesh,record,self.proof
