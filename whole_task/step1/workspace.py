"""Station geometry: six axis-aligned object placements on one horizontal tabletop."""
import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation


def surface_workspace(mesh, reference_pose, size_fraction=.5, base_position_world=(0.,0.,0.)):
    """A surface work cube extending inward from the robot-facing object AABB face.

    Every side is size_fraction of the longest reference object AABB side.
    The two adjoining faces share their centre. Only the work box is rendered;
    both its size and placement stay fixed when later object poses change.
    """
    if not np.isfinite(size_fraction) or size_fraction <= 0:
        raise ValueError('Workspace size fraction must be finite and positive')
    rotation=np.asarray(reference_pose['rotation_world_object'],float)
    shift=np.asarray(reference_pose['translation_world_m'],float)
    vertices=mesh.vertices@rotation.T+shift
    reference=np.array([vertices.min(axis=0),vertices.max(axis=0)])
    size=reference[1]-reference[0];edge=float(size_fraction*size.max())
    work_size=np.full(3,edge);center=reference.mean(axis=0)
    face_centers=np.repeat(center[None],6,axis=0)
    for axis in range(3):
        face_centers[2*axis,axis]=reference[0,axis]
        face_centers[2*axis+1,axis]=reference[1,axis]
    nearest=int(np.linalg.norm(face_centers-np.asarray(base_position_world,float),axis=1).argmin())
    axis=nearest//2;sign=1 if nearest%2 else -1;shared=face_centers[nearest]
    work_center=shared.copy();work_center[axis]-=sign*work_size[axis]/2
    bounds=np.array([work_center-work_size/2,work_center+work_size/2])
    return dict(frame='world',shape='cube',applies_to='surface_process_point',reference_pose='+X',
                rule='cube edge is size_fraction times the longest reference object AABB edge; the robot-facing faces share their centre; extend toward the object interior and remain fixed for every task; do not clip cube dimensions to the object AABB',
                object_extents_m=mesh.extents.tolist(),object_aabb_size_world_m=size.tolist(),
                size_fraction=float(size_fraction),cube_edge_m=edge,workspace_size_world_m=work_size.tolist(),
                robot_base_position_world_m=np.asarray(base_position_world,float).tolist(),
                adjoining_axis='xyz'[axis],adjoining_sign=sign,
                shared_face_center_world_m=shared.tolist(),workspace_center_world_m=work_center.tolist(),
                reference_object_bounds_world_m=reference.tolist(),bounds_world_m=bounds.tolist(),
                xyz_ranges_world_m={axis:bounds[:,i].tolist() for i,axis in enumerate('xyz')})


def points_inside_box(points, bounds, tolerance=1e-9):
    """Test surface/process points directly; tool length and head position do not enter."""
    points=np.asarray(points,float)
    bounds=np.asarray(bounds,float)
    return np.isfinite(points).all(axis=-1)&((points>=bounds[0]-tolerance)&(points<=bounds[1]+tolerance)).all(axis=-1)


def box_edges(bounds, shift=(0.,0.,0.)):
    """Twelve box edges in world coordinates, or translated render coordinates."""
    bounds=np.asarray(bounds,float)-np.asarray(shift,float)
    keys=list(product((0,1),repeat=3))
    points={key:np.array([bounds[key[i],i] for i in range(3)]) for key in keys}
    return [(points[a],points[b]) for i,a in enumerate(keys) for b in keys[i+1:]
            if sum(x!=y for x,y in zip(a,b))==1]


def tabletop_placements(mesh, views, setup):
    """Turn each object side toward the robot horizontally, then lower it onto the table.

    The original bunny's +Y stays up for the four side views. Top/bottom views
    use -Z/+Z up, respectively. Tool tilt never changes the object's rotation.
    """
    front = -np.asarray(setup['working_direction_world'], float).copy(); front[2] = 0
    if np.linalg.norm(front) < 1e-9: front = np.array([0., 1., 0.])
    front /= np.linalg.norm(front)
    world_up = np.array([0., 0., 1.])
    world_basis = np.column_stack([np.cross(front, world_up), front, world_up])
    center = mesh.bounds.mean(0); point = np.asarray(setup['work_point_m']); out = []
    for side in np.asarray(views, float):
        up = np.array([0., 1., 0.]) if abs(side[1]) < .5 else np.array([0., 0., -side[1]])
        rotation = world_basis @ np.column_stack([np.cross(side, up), side, up]).T
        vertices = mesh.vertices @ rotation.T
        shift = point-rotation@center
        shift[2] = setup['table_height_m']-vertices[:, 2].min()
        out.append(dict(rotation_world_object=rotation.tolist(), translation_world_m=shift.tolist()))
    return out


def onto(a,b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a=np.asarray(a,float)/np.linalg.norm(a);b=np.asarray(b,float)/np.linalg.norm(b);v=np.cross(a,b);s=np.linalg.norm(v);c=float(a@b)
    if s<1e-12:
        if c>0:return np.eye(3)
        u=np.cross(a,[1.,0,0] if abs(a[0])<.9 else [0,1.,0]);return Rotation.from_rotvec(np.pi*u/np.linalg.norm(u)).as_matrix()
    return Rotation.from_rotvec(v/s*np.arctan2(s,c)).as_matrix()
