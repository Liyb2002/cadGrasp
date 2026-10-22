"""Shared camera and legacy cone drawing mathematics."""
import numpy as np

def cone_segments(workspace):
    cone = workspace['spatial_cone']
    apex = np.asarray(cone['apex_world_m']); axis = np.asarray(cone['axis_world'])
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    length = cone['reference_axial_distance_m']
    extension = length+workspace['usable_radius_m']
    theta = np.linspace(0., 2*np.pi, 129)
    direction = np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v
    ring = apex+length*axis+length*np.tan(np.radians(cone['half_angle_deg']))*direction
    end = apex+(ring-apex)*(extension/length)
    lines = [*np.stack([ring[:-1], ring[1:]], axis=1)]
    for i in range(0, 128, 16):
        lines.append(np.array([apex, ring[i]]))
        for j in range(12):
            lines.append(np.array([ring[i]+(end[i]-ring[i])*j/12,
                                   ring[i]+(end[i]-ring[i])*(j+.50)/12]))
    lines.extend(np.stack([end[:-1:2], end[1::2]], axis=1))
    return np.asarray(lines).reshape(-1, 3), np.concatenate([[apex], ring, end])


def camera_fit(r, direction, coordinates, width, height, margin=1.1):
    direction = np.array(direction, dtype=float); direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    coordinates = np.asarray(coordinates)
    xy = coordinates @ np.stack([right, up], axis=1)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    target = right*((lo[0]+hi[0])/2)+up*((lo[1]+hi[1])/2)+direction*(coordinates.mean(axis=0) @ direction)
    scale = max(hi[0]-lo[0], (hi[1]-lo[1])*width/height)*margin
    return r.orthographic(scale, scale*height/width) @ r.view_matrix(direction, target), target, direction

