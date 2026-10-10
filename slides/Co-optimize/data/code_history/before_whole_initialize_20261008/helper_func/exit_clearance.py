"""Persistent exit clearance with explicit, physically available seating exceptions."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from collections import OrderedDict
from shapely.geometry import Polygon
from shapely.ops import unary_union

FRACTION=.01
CONTACT_DEPTH_M=.005
_PRISM_FACES=np.array([[2,1,0],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])

class ExitClearance:
    def __init__(self,mesh,fraction=FRACTION):
        self._sweep_cache=OrderedDict();self.mesh=mesh;self.radius=float(fraction*np.max(mesh.extents));self.fraction=fraction
        kernel=trimesh.creation.icosphere(subdivisions=1,radius=1.)
        inradius=float(np.einsum('ij,ij->i',kernel.triangles[:,0],kernel.face_normals).min())
        kernel.vertices*=self.radius/inradius
        self.kernel=S.solid(kernel)
        self.expanded=S.unpack(S.solid(mesh).minkowski_sum(self.kernel)) if self.radius else mesh.copy()
        self.metadata=dict(fraction_of_object_max_extent=fraction,object_max_extent_m=float(np.max(mesh.extents)),minimum_radial_clearance_m=self.radius,maximum_kernel_radius_m=self.radius/inradius,kernel='circumscribed icosphere, subdivision 1; contains the requested Euclidean ball',contact_core_depth_m=CONTACT_DEPTH_M,contact_exception='Only positive-volume cores under actual current nominal-sweep contact patches; never material intersecting an unpadded object sweep',policy='Rebuild from Step3.3: remove padded sweeps, retain current collision-free contact cores; clearance is mandatory outside these explicit cores')

    def sweep(self,displacement,fan=8,padded=True):
        key=(bool(padded),int(fan),tuple(np.asarray(displacement,float)))
        if key not in self._sweep_cache:
            try:
                result=S.solid(S.swept_solid(self.expanded if padded else self.mesh,displacement,fan_in=fan))
            except RuntimeError:
                if not padded:raise
                # Equivalent Minkowski order avoids numerically collapsed
                # leading prisms on tiny faces created by object dilation.
                result=self.sweep(displacement,fan,padded=False).minkowski_sum(self.kernel)
            material_volume(result)
            self._sweep_cache[key]=result
        self._sweep_cache.move_to_end(key)
        if len(self._sweep_cache)>16:self._sweep_cache.popitem(last=False)
        return self._sweep_cache[key]

    def construct(self,seed,nominal_sweeps,padded_sweeps,allowed,boundary=contact_boundary,check_contacts=True):
        nominal_cut=union(nominal_sweeps);padded_cut=union(padded_sweeps)
        nominal=seed-nominal_cut;nominal_mesh=S.unpack(nominal)
        triangles,sources=boundary(self.mesh,nominal_mesh,allowed) if len(nominal_mesh.faces) else (np.empty((0,3,3)),np.empty(0,int))
        parts=[];bearing_triangles=[];bearing_sources=[]
        for tri,src in zip(triangles,sources):
            normal=self.mesh.face_normals[src]
            prism=trimesh.Trimesh(np.vstack([tri,tri+CONTACT_DEPTH_M*normal]),_PRISM_FACES,process=False)
            if prism.volume<0:prism.invert()
            part=S.solid(prism)
            if material_volume(part^nominal)>1e-16:
                parts.append(part);bearing_triangles.append(tri);bearing_sources.append(src)
        core=union(parts)^nominal if parts else F.md.Manifold()
        remaining=((seed-padded_cut)+core)-nominal_cut
        remaining=remaining-(padded_cut-core)
        removed=seed-remaining
        overlap=max((material_volume(remaining^s) for s in nominal_sweeps),default=0.)
        margin_overlap=material_volume((remaining-core)^padded_cut)
        partition=abs(material_volume(seed)-material_volume(remaining)-material_volume(removed))
        if check_contacts:
            kept_mesh=S.unpack(remaining)
            kept_tri,kept_src=boundary(self.mesh,kept_mesh,allowed) if len(kept_mesh.faces) else (np.empty((0,3,3)),np.empty(0,int))
            def contact_area(tris,srcs):
                area=0.
                for src in np.unique(srcs):
                    original=self.mesh.triangles[src];u=original[1]-original[0];u/=np.linalg.norm(u);v=np.cross(self.mesh.face_normals[src],u)
                    polygons=[Polygon(np.c_[(t-original[0])@u,(t-original[0])@v]) for t in tris[srcs==src]]
                    area+=unary_union(polygons).area
                return float(area)
            before=contact_area(np.asarray(bearing_triangles).reshape(-1,3,3),np.asarray(bearing_sources,int));after=contact_area(kept_tri,kept_src)
            preserved=after>=before-max(1e-12,before*1e-6)
        else:
            kept_tri=np.empty((0,3,3));kept_src=np.empty(0,int)
            before=after=preserved=None
        resolved=max(overlap,margin_overlap,partition)<1e-10 and (not check_contacts or preserved)
        diagnostics=dict(contact_check_performed=check_contacts,nominal_sweep_overlap_m3=overlap,padded_sweep_overlap_outside_contact_cores_m3=margin_overlap,partition_error_m3=partition,nominal_contact_area_m2=before,remaining_contact_area_m2=after,contact_area_preserved=preserved,protected_contact_core_volume_cm3=material_volume(core)*1e6,geometry_resolved=resolved)
        return dict(remaining=remaining,removed=removed,nominal_remaining=nominal,protected_contact_core=core,nominal_cut=nominal_cut,padded_cut=padded_cut,triangles=kept_tri,sources=kept_src,diagnostics=diagnostics)
