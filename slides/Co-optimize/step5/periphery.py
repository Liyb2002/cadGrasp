"""Peel a shared solid with an enclosing convex boundary, never local pockets.

The retained solid is original material intersected with a convex envelope.
Moving that envelope inward can remove contact patches too. Every committed
mesh is checked against the original complete force and ground conditions.
"""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import manifold3d as md
import trimesh
from scipy import ndimage
from prune_support import Pruner, material_components
from deterministic_base import solid, unpack, volume, box, enclosure, SCALE
from whole_search.common import C, transform_mesh
from whole_search.classify import classify


def clip_triangles(triangles,sources,normal,limit):
    """Clip actual contact polygons; a new vertex contributes its true moment."""
    if not len(triangles):return triangles,sources
    distances=triangles@normal-limit
    inside=distances<=0
    count=inside.sum(1)
    output=[triangles[count==3]];ids=[sources[count==3]]
    for selected in [1,2]:
        for triangle,source,mask,ds in zip(triangles[count==selected],sources[count==selected],
                                          inside[count==selected],distances[count==selected]):
            polygon=[]
            for a in range(3):
                b=(a+1)%3
                if mask[a]:polygon.append(triangle[a])
                if mask[a]!=mask[b]:
                    polygon.append(triangle[a]+ds[a]/(ds[a]-ds[b])*(triangle[b]-triangle[a]))
            for j in range(1,len(polygon)-1):
                output.append(np.array([[polygon[0],polygon[j],polygon[j+1]]]))
                ids.append(np.array([source]))
    return np.concatenate(output),np.concatenate(ids)


class PlaneForceFilter:
    """Reject impossible cuts before a costly solid Boolean; acceptance is later."""
    def __init__(self,certificate,mesh,fixed_mesh=None):
        self.certificate=certificate;self.contacts={};self.cache={}
        self.fixed_contacts={}
        self.bases={k:[]for k in certificate.layout.active}
        for k in certificate.layout.active:
            Q=certificate.layout.placements[k]
            direction=Q[:3,:3].T@certificate.layout.directions[k]
            ids=certificate.allowed[k][certificate.mesh.face_normals[certificate.allowed[k]]@direction<=1e-9]
            self.contacts[k]=C.contact_boundary(transform_mesh(certificate.mesh,Q),mesh,ids)
            if fixed_mesh is not None:
                self.fixed_contacts[k]=C.contact_boundary(transform_mesh(certificate.mesh,Q),fixed_mesh,ids)

    def passes(self,planes):
        cert=self.certificate
        for k in cert.layout.active:
            triangles,sources=self.contacts[k]
            for normal,limit in planes:
                triangles,sources=clip_triangles(triangles,sources,normal,limit)
            if k in self.fixed_contacts:
                ft,fs=self.fixed_contacts[k]
                triangles=np.concatenate([triangles,ft]);sources=np.concatenate([sources,fs])
            Q=cert.layout.placements[k]
            points=C.transform_points(triangles.reshape(-1,3),cert.native[k]@np.linalg.inv(Q))
            normals=np.repeat(-cert.tasks[k].domain.mesh.face_normals[sources],3,axis=0)
            raw=np.c_[normals,np.cross(points-cert.tasks[k].domain.com,normals)]
            floor=cert.floors[k]if cert.grounded[k]else cert.floors[k][-1:]
            full=C.I.merge_columns(floor,C.U.heads(raw,cert.tasks[k].scale))
            key=(k,hashlib.sha256(full.tobytes()).hexdigest())
            if key not in self.cache:
                columns=[int.from_bytes(hashlib.blake2b(row.tobytes(),digest_size=8).digest(),'little')for row in full]
                mask,_=classify(full,cert.tasks[k].targets,basis_cache=self.bases[k],column_ids=columns,
                                stop_on_counterexample=True)
                self.cache[key]=bool(mask.all())
            if not self.cache[key]:return False
        return True


def structural_profile(mesh, pitch=.001, radii=(.001, .0015, .002, .0025)):
    """Measure erosion bottlenecks, distinguishing necks from terminal tips.

    This is a resolved geometric measurement, not a strength certificate.
    A sphere centre remains inside the eroded material only where a connector
    can contain that radius. Disconnected eroded lobes expose narrow necks.
    The grid resolution and all lobe volumes are recorded, including tiny ends.
    """
    from continuous_legs import raster
    origin=np.floor(mesh.bounds[0]/pitch)*pitch-pitch
    shape=tuple(np.ceil((mesh.bounds[1]-origin)/pitch).astype(int)+2)
    # Exact signed-solid slices retain cavities and avoid adding a half-voxel
    # surface shell, which otherwise overstates thin connector diameters.
    inside=raster(solid(mesh),origin,shape,pitch)
    transform=np.eye(4);transform[:3,:3]*=pitch;transform[:3,3]=origin
    voxels=trimesh.voxel.VoxelGrid(inside,transform=transform)
    distance=ndimage.distance_transform_edt(inside)*pitch
    records=[]
    for radius in radii:
        labels,_=ndimage.label(distance>radius)
        sizes=np.bincount(labels.ravel())[1:]
        sizes=np.sort(sizes[sizes>0])[::-1]
        total=int(sizes.sum())
        records.append(dict(radius_m=radius,diameter_m=2*radius,
            core_components=len(sizes),core_volume_cm3=total*pitch**3*1e6,
            separated_core_fraction=0. if not total else float(1-sizes[0]/total),
            component_volumes_cm3=(sizes*pitch**3*1e6).tolist()))
    return dict(pitch_m=pitch,profiles=records,grid_shape=list(inside.shape),
        true_solid_slices_used=True,cavities_preserved=True,
        sampled_material_cm3=int(inside.sum())*pitch**3*1e6,
        actual_material_cm3=abs(float(mesh.volume))*1e6),voxels,distance


class ThicknessGuard:
    def __init__(self,mesh,radius,maximum_separated_fraction,pitch=.001):
        record,voxels,distance=structural_profile(mesh,pitch,(radius,))
        self.record=record;self.pitch=pitch;self.radius=radius
        self.target=maximum_separated_fraction
        self.core=distance>radius
        self.origin=voxels.transform[:3,3].copy()
        self.indices=np.argwhere(self.core)
        self.points=self.indices*pitch+self.origin
        self.baseline=record['profiles'][0]['separated_core_fraction']
        self.current=self.baseline
        self.fixed=np.zeros(len(self.points),bool)

    def retain_fixed(self,mesh):
        _,voxels,distance=structural_profile(mesh,self.pitch,(self.radius,))
        ids=np.rint((self.points-voxels.transform[:3,3])/self.pitch).astype(int)
        valid=np.all((ids>=0)&(ids<distance.shape),axis=1)
        self.fixed[valid]=distance[tuple(ids[valid].T)]>self.radius

    def assess(self,planes):
        keep=np.ones(len(self.points),bool)
        # Erosion distributes over intersections. Erode every new halfspace
        # by the same physical radius, rather than merely clipping cell centres.
        for normal,limit in planes:
            keep &= self.points@normal <= limit-self.radius
        keep |= self.fixed
        mask=np.zeros_like(self.core)
        ids=self.indices[keep]
        if len(ids):mask[tuple(ids.T)]=True
        labels,_=ndimage.label(mask)
        sizes=np.bincount(labels.ravel())[1:]
        total=int(sizes.sum())
        separated=1. if not total else float(1-sizes.max()/total)
        passed=separated<=max(self.current,self.target)+1e-12
        return passed,separated

    def assess_mesh(self,mesh):
        record,_,_=structural_profile(mesh,self.pitch,(self.radius,))
        separated=record['profiles'][0]['separated_core_fraction']
        return separated<=max(self.current,self.target)+1e-12,separated


def directions(mesh):
    axes=np.vstack([np.eye(3),-np.eye(3),
        np.array([v for v in np.ndindex(3,3,3)if v!=(1,1,1)],float)-1,
        md_to_directions(mesh)])
    axes=axes[np.linalg.norm(axes,axis=1)>1e-9]
    axes=axes/np.linalg.norm(axes,axis=1)[:,None]
    result=[]
    for axis in axes:
        if not any(axis@other>.997 for other in result):result.append(axis)
    return result


def md_to_directions(mesh):
    # Contact/branch extremities supply additional directions beyond XYZ.
    hull=mesh.convex_hull
    normals=hull.face_normals
    ids=np.argsort(-hull.area_faces,kind='stable')[:18]
    return normals[ids]


class OutsideInPruner(Pruner):
    def __init__(self,*args,thickness_guard=None,checkpoint=None,resolution=.001,contact_core=None,**kwargs):
        kwargs.update(grids=(),budget=10**9,require_connected=True)
        super().__init__(*args,**kwargs)
        self.guard=thickness_guard;self.checkpoint=checkpoint
        self.resolution=resolution;self.planes={};self.proposed_planes=None
        self.envelope=box(self.fixture.bounds[0]-.01,self.fixture.bounds[1]+.01)
        self.commits=0
        self.core=self.original if contact_core is None else solid(contact_core)
        self.core_mesh=unpack(self.core)
        self.fixed_feet=self.original-self.core
        self.fixed_mesh=unpack(self.fixed_feet)if volume(self.fixed_feet)>1e-14 else None
        if self.guard is not None and self.fixed_mesh is not None:self.guard.retain_fixed(self.fixed_mesh)

    def accept(self,candidate,operation):
        if volume(candidate)<=1e-14:return False
        planes=list((self.proposed_planes or self.planes).values())
        if self.guard is not None and self.fixed_mesh is None:
            passed,separated=self.guard.assess(planes)
            if not passed:return False
        if not all(row['passed']for row in enclosure(unpack(candidate),self.rows)):
            return False
        previous=(self.body,self.proof,self.best_mesh)
        if not super().accept(candidate,operation):return False
        if not all(row['passed']for row in enclosure(self.best_mesh,self.rows)):
            self.body,self.proof,self.best_mesh=previous
            self.history[-1].update(accepted=False,error='Serialized ground enclosure rejected cut')
            return False
        if self.guard is not None and self.fixed_mesh is not None:
            passed,separated=self.guard.assess_mesh(self.best_mesh)
            if not passed:
                self.body,self.proof,self.best_mesh=previous
                self.history[-1].update(accepted=False,error='Actual sliced solid creates a thinner connector')
                print('THICKNESS_REJECT',round(separated,6),flush=True)
                return False
        if self.guard is not None:self.guard.current=separated
        if self.proposed_planes is not None:self.planes=self.proposed_planes.copy()
        self.commits+=1
        if self.checkpoint:self.checkpoint(self.best_mesh,self.proof,self.history)
        return True

    def run(self):
        self.proof=self.certificate.evaluate(self.fixture);self.best_mesh=self.fixture
        if not self.proof['passed']:raise ValueError('Original complete support fails original demands')
        initial_contacts=self.proof['contact_triangles'].copy()
        force_filter=PlaneForceFilter(self.certificate,self.fixture,self.fixed_mesh)
        axes=directions(self.core_mesh)
        passes=0
        # Exhaust the direction stencil at each scale. There is no arbitrary
        # candidate count or wall-clock expiry that silently retains old teeth.
        for tolerance in [self.resolution*2,self.resolution]:
            while True:
                changes=0;passes+=1
                for index,normal in enumerate(axes):
                    print('PEEL_DIRECTION',passes,index+1,'/',len(axes),'commits',self.commits,flush=True)
                    if self.core.is_empty():continue
                    dots=self.core_mesh.vertices@normal
                    edge=float(dots.max());low=float(dots.min())
                    rejected=low;accepted=edge
                    while accepted-rejected>tolerance:
                        limit=(accepted+rejected)/2
                        cropped_core=self.core.trim_by_plane(tuple(-normal),-limit/SCALE)
                        candidate=cropped_core+self.fixed_feet
                        self.proposed_planes=dict(self.planes)
                        self.proposed_planes[index]=(normal,limit)
                        if (force_filter.passes(list(self.proposed_planes.values()))
                                and self.accept(candidate,f'outer_plane_{index}_limit_{limit:.9f}')):
                            accepted=limit;changes+=1
                            self.core=cropped_core;self.core_mesh=unpack(self.core)
                            self.envelope=self.envelope.trim_by_plane(tuple(-normal),-limit/SCALE)
                        else:rejected=limit
                        self.proposed_planes=None
                print('OUTSIDE_IN_PASS',passes,'accepted',changes,'cm3',round(volume(self.body)*1e6,4),flush=True)
                if not changes:break
        record=dict(passed=True,style='outside_in_convex_envelope',
            local_grid_pockets_used=False,all_contacts_pinned=False,
            contact_core_peeled_separately_from_floor_ends=True,
            direction_count=len(axes),converged_directional_passes=passes,
            inward_resolution_m=self.resolution,candidate_budget=None,
            accepted_cuts=self.commits,attempts=self.attempts,history=self.history,
            initial_contact_triangles=initial_contacts,
            final_contact_triangles=self.proof['contact_triangles'],
            removed_material_cm3=volume(self.original-self.body)*1e6,
            seconds=time.monotonic()-self.began,
            global_minimum_claimed=False)
        return self.best_mesh,record,self.proof

    def round_envelope(self,radius=.003):
        sphere=md.Manifold.sphere(radius/SCALE,20)
        rounded=self.envelope.minkowski_difference(sphere).minkowski_sum(sphere)
        return unpack((self.core^rounded)+self.fixed_feet)
