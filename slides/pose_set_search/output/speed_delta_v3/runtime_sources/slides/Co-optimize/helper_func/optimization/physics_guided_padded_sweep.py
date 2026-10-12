"""Conservative continuous padded sweeps with explicit collapsed-prism repair.

Only the padded exclusion uses this representation. The original nominal
object/sweep and all contact/force inputs remain unchanged. Each repaired
prism contains its original six vertices, so no positive leading prism is
omitted. Repair adds a recorded tiny cube to that prism, never subtracts it.
"""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull,QhullError
from step4_connect_support.translation_sweep import _solid,_union,_PRISM_FACES
from exit_clearance import ExitClearance
from co_common import S,material_volume


def repaired_prism(vertices,label):
    vertices=np.asarray(vertices,float)
    corners=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
    for epsilon in [1e-12,1e-11,1e-10,1e-9]:
        points=(vertices[:,None,:]+epsilon*corners[None,:,:]).reshape(-1,3)
        try:
            hull=ConvexHull(points)
            # Verify containment before constructing the closed convex body.
            if np.max(vertices@hull.equations[:,:3].T+hull.equations[:,3])>0:continue
            faces=hull.simplices.copy();triangles=points[faces]
            flip=np.einsum('ij,ij->i',np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),hull.equations[:,:3])<0
            faces[flip]=faces[flip][:,::-1]
            value=_solid(points,faces,label+' conservative repair')
            if not value.is_empty() and value.volume()>0:
                actual=value.to_mesh64();actual_vertices=np.asarray(actual.vert_properties[:,:3])
                actual_triangles=actual_vertices[np.asarray(actual.tri_verts)]
                normal=np.cross(actual_triangles[:,1]-actual_triangles[:,0],actual_triangles[:,2]-actual_triangles[:,0])
                lengths=np.linalg.norm(normal,axis=1)
                if np.any(lengths==0):continue
                normal/=lengths[:,None]
                if np.max(vertices@normal.T-np.einsum('ij,ij->i',actual_triangles[:,0],normal))>0:continue
                return value,epsilon
        except (ValueError,QhullError):continue
    raise RuntimeError(label+': conservative prism repair unresolved')


def padded_swept_solid(mesh,displacement,*,fan_in=8):
    displacement=np.asarray(displacement,float)
    if displacement.shape!=(3,) or not np.isfinite(displacement).all():raise ValueError('finite displacement required')
    if not isinstance(fan_in,int) or fan_in<2:raise ValueError('Boolean fan-in >=2 required')
    if not len(mesh.faces) or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('closed outward positive-volume mesh required')
    if np.all(displacement==0):
        result=mesh.copy();result.metadata['padded_sweep_repair']=dict(leading_faces=0,repaired_prisms=0,
            maximum_extra_cube_half_extent_m=0.,repairs=[])
        return result
    origin=np.asarray(mesh.bounds).mean(axis=0);scale=max(float(mesh.extents.max()),float(np.linalg.norm(displacement)))
    vertices=(np.asarray(mesh.vertices)-origin)/scale;delta=displacement/scale
    triangles=vertices[mesh.faces];extended=triangles.astype(np.longdouble)
    signed=np.cross(extended[:,1]-extended[:,0],extended[:,2]-extended[:,0])@delta.astype(np.longdouble)
    leading=np.flatnonzero(signed>0);pieces=[_solid(vertices,mesh.faces,'Original padded object')]
    repairs=[]
    for index in leading:
        points=np.vstack([triangles[index],triangles[index]+delta]);label=f'Padded face prism {index}'
        try:
            value=_solid(points,_PRISM_FACES,label)
            valid=not value.is_empty() and value.volume()>0
        except ValueError:valid=False
        if not valid:
            value,epsilon=repaired_prism(points,label);repairs.append(dict(face=int(index),extra_cube_half_extent_m=epsilon*scale))
        pieces.append(value)
    joined=_union(pieces,fan_in);data=joined.to_mesh64()
    result=trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])*scale+origin,np.asarray(data.tri_verts),process=False)
    if not result.is_watertight or not result.is_winding_consistent or result.volume<=0:raise RuntimeError('Padded sweep output unresolved')
    result.metadata['padded_sweep_repair']=dict(leading_faces=len(leading),repaired_prisms=len(repairs),
        maximum_extra_cube_half_extent_m=max((r['extra_cube_half_extent_m'] for r in repairs),default=0.),
        policy='every positive leading prism retained; convex repairs contain original six vertices; conservative additional exclusion only',
        repairs=repairs)
    return result


class ConservativeExitClearance(ExitClearance):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.padded_repairs=[]

    def sweep(self,displacement,fan=8,padded=True):
        if not padded:return super().sweep(displacement,fan,padded=False)
        key=(True,int(fan),tuple(np.asarray(displacement,float)))
        if key not in self._sweep_cache:
            try:
                mesh=padded_swept_solid(self.expanded,displacement,fan_in=fan)
                value=S.solid(mesh);record=mesh.metadata['padded_sweep_repair']
                self.padded_repairs.append(dict(displacement=np.asarray(displacement).tolist(),fan=fan,**record))
            except RuntimeError:
                # Keep the original exact Minkowski-order fallback if repair fails.
                value=self.sweep(displacement,fan,padded=False).minkowski_sum(self.kernel)
                self.padded_repairs.append(dict(displacement=np.asarray(displacement).tolist(),fan=fan,
                    fallback='original nominal sweep Minkowski sum with the unchanged kernel'))
            material_volume(value);self._sweep_cache[key]=value
        self._sweep_cache.move_to_end(key)
        if len(self._sweep_cache)>16:self._sweep_cache.popitem(last=False)
        return self._sweep_cache[key]

    def construct(self,*args,**kwargs):
        result=super().construct(*args,**kwargs)
        result['padded_sweep_repairs']=list(self.padded_repairs)
        result['diagnostics']['maximum_padded_prism_extra_cube_half_extent_m']=max((r.get('maximum_extra_cube_half_extent_m',0.) for r in self.padded_repairs),default=0.)
        return result
