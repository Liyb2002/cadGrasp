"""Station geometry: six axis-aligned object placements on one horizontal tabletop."""
import numpy as np
from scipy.spatial.transform import Rotation


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
