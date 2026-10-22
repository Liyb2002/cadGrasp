"""Source surface inside a ball intersected with a cone, without surface filters.

Whole source triangles discretize the sector boundary. No normal, visibility,
inner-shell, head-clearance or component filter removes faces from a region.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .step1_3 import _seed_faces

POLICY=Path(__file__).with_name('needle_workspace.json')
MODEL='needle_spherical_sector_surface_intersection'


def workspace(config,inset=.05):
    if not 0 <= inset < .5:
        raise ValueError('Proposal inset must lie in [0, .5)')
    policy=json.loads(POLICY.read_text())
    axis=np.asarray(config['task_constraint']['working_direction_world'],dtype=float)
    axis/=np.linalg.norm(axis)
    length=policy['needle_length_m']
    center=np.asarray(config['task_constraint']['world_position_constraint']['center_world_m'])
    angle=float(config['task_constraint']['cone_half_angle_deg'])
    return dict(model=MODEL,needle_length_m=length,needle_extensible=False,
        nominal_axis_world=axis.tolist(),half_angle_deg=angle,usable_half_angle_deg=angle,
        reference_head_front_world_m=(center-length*axis).tolist(),
        sphere_center='head_front_needle_root',minimum_reference_distance_m=0.,
        maximum_reference_distance_m=length,proposal_inset_fraction=inset,
        proposal_depth_fractions=policy['proposal_depth_fractions'],
        boundary_discretization='whole_source_triangle_fully_inside_convex_sector',
        radial_inset_fraction=0.,angle_inset_fraction=0.,
        normal_filter_applied=False,visibility_filter_applied=False,
        head_clearance_filter_applied=False,flange_reach_filter_applied=False,
        connected_component_filter_applied=False,hole_filling_applied=False,
        reference_head_is_virtual=False,
        head_front_to_flange_m=policy['head_front_to_flange_m'],
        head_proxy_radius_m=policy['head_proxy_radius_m'],needle_outer_radius_m=policy['needle_outer_radius_m'],
        native_robot_certified=False,full_tool_collision_certified=False)


def geometric_mask(mesh,head,forward,ws):
    """Retain every triangle fully inside the ball and forward cone.

    Both sets are convex: all vertices inside proves whole-triangle containment.
    """
    forward=np.asarray(forward,dtype=float);forward=forward/np.linalg.norm(forward)
    delta=mesh.triangles-np.asarray(head)
    distance=np.linalg.norm(delta,axis=2);axial=delta@forward
    return np.all((distance<=ws['maximum_reference_distance_m']+1e-12)
        &(axial>=distance*np.cos(np.deg2rad(ws['half_angle_deg']))-1e-12),axis=1)


def evaluate(mesh,head,outward,config,ws,neighbors=None,anchor=None):
    head=np.asarray(head,dtype=float)
    forward=-np.asarray(outward,dtype=float);forward/=np.linalg.norm(forward)
    mask=geometric_mask(mesh,head,forward,ws)
    if not mask.any() or (anchor is not None and not mask[anchor]):
        return []
    # Reject a head placed inside the solid as a whole pose; this never carves
    # holes out of a selected sector. No head-size or clearance cutoff is used.
    if mesh.is_watertight and mesh.contains(head[None])[0]:
        return []
    ids=np.flatnonzero(mask)
    rotation=Rotation.align_vectors([ws['nominal_axis_world']],[forward])[0].as_matrix()
    translation=np.asarray(ws['reference_head_front_world_m'])-rotation@head
    aim=np.average(mesh.triangles_center[ids],weights=mesh.area_faces[ids],axis=0)
    view=head-aim;view/=np.linalg.norm(view)
    area=float(mesh.area_faces[ids].sum())
    return [dict(face_ids=ids,head_object_m=head,outward_axis_object=-forward,
        rotation_world_object=rotation,translation_world_object_m=translation,
        view_direction_object=view,area_m2=area,selection_score_m2=area)]


_WORKER=None


def _init_worker(vertices,faces,config,ws):
    import trimesh
    global _WORKER
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    _WORKER=(mesh,config,ws)


def proposal_depths(ws):
    return ws['needle_length_m']*np.minimum(ws['proposal_depth_fractions'],1-ws['proposal_inset_fraction'])


def _axis_candidates(axis):
    mesh,config,ws=_WORKER
    # Pose proposals only: opposite-side and occluded faces remain eligible.
    front=mesh.face_normals@axis>0
    centers=[mesh.center_mass,np.average(mesh.triangles_center[front],weights=mesh.area_faces[front],axis=0)]
    return [item for center in centers for depth in proposal_depths(ws)
            for item in evaluate(mesh,center+depth*axis,axis,config,ws)]


def _face_candidates(f):
    mesh,config,ws=_WORKER
    axis=mesh.face_normals[f];center=mesh.triangles_center[f]
    return [item for depth in proposal_depths(ws)
            for item in evaluate(mesh,center+depth*axis,axis,config,ws,anchor=int(f))]


def build_candidates(mesh,config,ws,direction_count=384,workers=8):
    from concurrent.futures import ProcessPoolExecutor
    n=np.arange(direction_count);z=1-2*(n+.5)/direction_count;phi=n*np.pi*(3-np.sqrt(5))
    axes=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
    seeds=_seed_faces(mesh,np.ones(len(mesh.faces),bool),limit=192)
    axes=np.vstack([axes,mesh.face_normals[seeds]])
    seen=set();rows=[];covered=np.zeros(len(mesh.faces),bool)
    def append(items):
        for row in items:
            mask=np.zeros(len(mesh.faces),bool);mask[row['face_ids']]=True
            key=np.packbits(mask).tobytes()
            if key in seen:continue
            seen.add(key);rows.append(row);covered[row['face_ids']]=True
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,
            initargs=(mesh.vertices,mesh.faces,config,ws)) as pool:
        for i,items in enumerate(pool.map(_axis_candidates,axes,chunksize=1)):
            append(items)
            if i%80==0:print('SECTOR CANDIDATES',i,'/',len(axes),'regions',len(rows),flush=True)
        missing=np.flatnonzero(~covered)
        for items in pool.map(_face_candidates,missing,chunksize=1):append(items)
    return sorted(rows,key=lambda row:-row['selection_score_m2'])
