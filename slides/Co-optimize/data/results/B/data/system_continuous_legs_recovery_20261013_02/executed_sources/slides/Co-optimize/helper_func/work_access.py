"""Unbounded working-face access cones shared by contacts, material and meshes.

Each whole triangle is Minkowski-added to a circumscribed polygonal cone.
No ray length or directional sampling truncates the search exclusion.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/co_optimize_work_access_numba')
import numpy as np
from numba import njit
from scipy.spatial import ConvexHull
from co_common import *

def unpack_solid(solid):
    data=solid.to_mesh64()
    return trimesh.Trimesh(np.array(data.vert_properties[:,:3],copy=True)*S.SCALE,
                           np.array(data.tri_verts,copy=True),process=False)

def tangent_frame(normal):
    normal=np.asarray(normal)/np.linalg.norm(normal)
    axis=np.eye(3)[np.argmin(abs(normal))];u=np.cross(normal,axis);u/=np.linalg.norm(u)
    return np.column_stack([u,np.cross(normal,u)])

SIDES=32


@njit(cache=True)
def _contains(points,planes,counts):
    result=np.zeros(len(points),np.bool_)
    for i in range(len(points)):
        x,y,z=points[i]
        for face in range(len(counts)):
            inside=True
            for p in range(counts[face]):
                a,b,c,d=planes[face,p]
                if a*x+b*y+c*z+d>1e-10:
                    inside=False
                    break
            if inside:
                result[i]=True
                break
    return result


class WorkAccess:
    def __init__(self,mesh,face_ids,half_angle_deg):
        self.face_ids=np.asarray(face_ids,int)
        self.triangles=mesh.triangles[self.face_ids].copy()
        self.normals=mesh.face_normals[self.face_ids].copy()
        self.half_angle_deg=float(half_angle_deg)
        if not 0<self.half_angle_deg<90:raise ValueError('Invalid saved work cone angle')
        self.clearance=float(mesh.extents.max())*1e-6
        self.radial_slope=np.tan(np.radians(self.half_angle_deg))/np.cos(np.pi/SIDES)
        angles=(np.arange(SIDES)+.5)*2*np.pi/SIDES
        self.rings=[];families=[]
        for triangle,normal in zip(self.triangles,self.normals):
            frame=tangent_frame(normal)
            ring=self.radial_slope*(np.cos(angles)[:,None]*frame[:,0]+np.sin(angles)[:,None]*frame[:,1])
            self.rings.append(ring)
            hull=ConvexHull(self.cell_points(triangle,normal,ring,1.))
            equations=hull.equations[hull.equations[:,:3] @ normal<1e-7]
            # Remove only the far cap: the remaining halfspaces are unbounded.
            _,unique=np.unique(np.round(equations,10),axis=0,return_index=True)
            equations=equations[np.sort(unique)]
            base=int(np.argmin(equations[:,:3] @ normal))
            equations=np.concatenate([equations[base:base+1],np.delete(equations,base,axis=0)])
            families.append(equations)
        self.rings=np.asarray(self.rings)
        self.counts=np.array([len(p) for p in families],np.int64)
        self.planes=np.zeros((len(families),int(self.counts.max()),4))
        for i,p in enumerate(families):self.planes[i,:len(p)]=p
        self._solids={}

    def cell_points(self,triangle,normal,ring,length):
        base=triangle-self.clearance*normal
        top=(base[:,None,:]+length*(normal+ring)[None,:,:]).reshape(-1,3)
        return np.vstack([base,top])

    def contains_points(self,points):
        return _contains(np.ascontiguousarray(points,dtype=np.float64),self.planes,self.counts)

    def required_length(self,points,minimum=.5):
        # All possible support is inside the fitted-wrap seed. Cap geometry
        # beyond that seed, never at a physical/tool or force-dependent length.
        height=np.asarray(points)[None]-self.triangles[:,0,None]
        maximum=float(np.einsum('fpc,fc->fp',height,self.normals).max())
        return max(float(minimum),maximum+2*self.clearance+.001)

    def solid(self,length):
        key=round(float(length),9)
        if key not in self._solids:
            parts=[S.solid(G.hull_mesh(self.cell_points(t,n,r,key)))
                   for t,n,r in zip(self.triangles,self.normals,self.rings)]
            self._solids[key]=union(parts)
        return self._solids[key]

    def mesh(self,length):
        return unpack_solid(self.solid(length))

    def definition(self):
        return dict(source='original task.domain.work_ids and task.domain.data.load.cone_half_deg',
            half_angle_deg=self.half_angle_deg,total_opening_deg=2*self.half_angle_deg,
            equation='union over ALL work-triangle points p of p + outward cone(alpha)',
            work_face_count=len(self.face_ids),unbounded_during_search=True,
            whole_work_triangles=True,full_angle_range=True,cone_sides=SIDES,
            circumscribed=True,maximum_radial_overestimate_fraction=float(1/np.cos(np.pi/SIDES)-1),
            clearance_m=self.clearance,self_occluded_directions_also_reserved=True,
            finite_mesh_cap_only_beyond_entire_candidate_seed=True)


class RegisteredWorkVolumes:
    """Union all original work cones; deduplicate nested cones on a face."""
    def __init__(self,mesh,states):
        self.mesh=mesh;self.poses={};angles={}
        for pose,(task,transform,body) in states.items():
            ids=np.asarray(task.domain.work_ids,int)
            angle=float(task.domain.data['load']['cone_half_deg'])
            if not 0<angle<90:raise ValueError('Invalid original work-cone half-angle')
            self.poses[pose]=dict(face_ids=ids.tolist(),half_angle_deg=angle,total_opening_deg=2*angle)
            for face in ids:angles[int(face)]=max(angle,angles.get(int(face),0.))
        self.ids=np.array(sorted(angles),int)
        self.families=[WorkAccess(mesh,[face for face in self.ids if angles[face]==angle],angle)
                       for angle in sorted(set(angles.values()))]

    def solid_for_points(self,points):
        lengths=[work.required_length(points,minimum=.001) for work in self.families]
        return union([work.solid(length) for work,length in zip(self.families,lengths)]),lengths

    def contains_points(self,points):
        result=np.zeros(len(points),bool)
        for work in self.families:result|=work.contains_points(points)
        return result

    def definition(self):
        return dict(poses=self.poses,face_ids=self.ids.tolist(),
            families=[work.definition() for work in self.families],
            registration='all original work triangles coincide with the reference object mesh',
            unbounded_exclusion=True,full_working_areas=True,
            full_original_angle_ranges=True,same_face_nested_cones_deduplicated=True)

    def rounded_excerpt(self):
        center=np.average(self.mesh.triangles_center[self.ids],axis=0,weights=self.mesh.area_faces[self.ids])
        radius=float(self.mesh.extents.max())*.65
        sphere=trimesh.creation.icosphere(subdivisions=5,radius=radius);sphere.apply_translation(center)
        volume,lengths=self.solid_for_points(sphere.vertices)
        region=unpack_solid(volume^S.solid(sphere))
        assert self.contains_points(region.vertices).all()
        assert np.linalg.norm(region.vertices-center,axis=1).max()<=radius+1e-10
        cap_minimum=float(np.min(np.einsum('fc,fc->f',sphere.face_normals,sphere.triangles_center-center)))
        return region,dict(truncation='original union of work exclusions intersected with one sphere',
            center_object_m=center.tolist(),radius_m=radius,spherical_cap_subdivisions=5,
            maximum_cap_radial_error_m=radius-cap_minimum,source_cone_definition=self.definition(),
            subset_of_original_exclusion=True,axial_cap_lengths_m=lengths,
            axial_caps_beyond_display_sphere=True,unbounded_exclusion_unchanged=True)
