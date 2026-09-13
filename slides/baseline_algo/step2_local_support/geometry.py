"""Finite-volume clearance and joined outer skins for local contact heads."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def clip_plane(poly, plane, inset=0.):
    values = poly@plane[:3]+plane[3]+inset
    inside = values <= 0
    if inside.all():
        return poly
    if not inside.any():
        return np.empty((0, 3))
    out = []
    for j in range(len(poly)):
        i = j-1
        if inside[i] != inside[j]:
            out.append(poly[i]+values[i]/(values[i]-values[j])*(poly[j]-poly[i]))
        if inside[j]:
            out.append(poly[j])
    return np.asarray(out)


def hull_mesh(points):
    """Oriented boundary of a convex cell; used for solid union and drawing."""
    import trimesh
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    t = points[faces]
    wrong = (np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])*hull.equations[:, :3]).sum(axis=1) < 0
    faces[wrong] = faces[wrong][:, [0, 2, 1]]
    return trimesh.Trimesh(points, faces, process=True)


class Clearance:
    """Test all triangle area against each whole convex volume, not rays."""
    def __init__(self, mesh, tolerance):
        self.mesh = mesh
        self.triangles = np.asarray(mesh.triangles)
        self.tree = mesh.triangles_tree
        self.tol = tolerance

    def obstruction(self, points):
        planes = ConvexHull(points).equations
        low, high = points.min(axis=0), points.max(axis=0)
        candidates = np.array(list(self.tree.intersection(np.r_[low-self.tol, high+self.tol])), int)
        if not len(candidates):
            return -1
        triangles = self.triangles[candidates]
        values = triangles@planes[:, :3].T+planes[:, 3]
        # A source triangle on the cell boundary is intentional contact.
        possible = ~np.any(np.all(values >= -self.tol, axis=1), axis=1)
        for face in candidates[possible]:
            poly = self.triangles[face]
            for plane in planes:
                poly = clip_plane(poly, plane, self.tol)
                if len(poly) < 3:
                    break
            if len(poly) >= 3:
                area = np.linalg.norm(np.cross(poly[1:-1]-poly[0], poly[2:]-poly[0]), axis=1).sum()/2
                if area > self.tol**2:
                    return int(face)
        return -1


def vertex_offsets(mesh, depth):
    """Continuous outer skin, at least depth beyond every incident face plane.

    Sharing offsets at vertices joins adjacent contact cells with positive-area
    side faces. Separate, disconnected normal prisms are not the contact head.
    """
    directions = np.asarray(mesh.vertex_normals)
    dot = np.einsum('fvc,fc->fv', directions[mesh.faces], mesh.face_normals)
    minimum = np.full(len(mesh.vertices), np.inf)
    np.minimum.at(minimum, mesh.faces.ravel(), dot.ravel())
    valid = minimum > 1e-6
    offsets = np.zeros_like(directions)
    offsets[valid] = depth*directions[valid]/minimum[valid, None]
    # A weighted average need not lie inside the common outward cone at a
    # sharp/saddle vertex. Do not call that a collision: solve all incident
    # face inequalities before rejecting this local offset construction.
    from scipy.optimize import linprog
    for vertex in np.flatnonzero(~valid):
        faces = mesh.vertex_faces[vertex]
        normals = mesh.face_normals[faces[faces >= 0]]
        answer = linprog([0,0,0,1,1,1],
                         A_ub=np.vstack([np.c_[-normals,np.zeros((len(normals),3))],
                                         np.c_[np.eye(3),-np.eye(3)],
                                         np.c_[-np.eye(3),-np.eye(3)]]),
                         b_ub=np.r_[-np.ones(len(normals)),np.zeros(6)],
                         bounds=[(None,None)]*3+[(0,None)]*3, method='highs')
        if answer.success and (normals@answer.x[:3]).min() >= 1-1e-8:
            offsets[vertex] = depth*answer.x[:3]
            valid[vertex] = True
    return offsets, valid


def head_cell(mesh, polygon, face, offsets):
    import trimesh
    tri = np.repeat(mesh.triangles[face][None], len(polygon), axis=0)
    weights = trimesh.triangles.points_to_barycentric(tri, polygon)
    top = polygon+weights@offsets[mesh.faces[face]]
    return np.vstack([polygon, top])


def face_graph(mesh, floor_height, tol):
    graph = [[] for _ in mesh.faces]
    centers = mesh.triangles_center
    for (a, b), edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        vertices = mesh.vertices[edge]
        z = vertices[:, 1]
        if z.max() < floor_height-tol:
            continue
        length = np.linalg.norm(vertices[1]-vertices[0])
        if z.min() < floor_height:
            length *= (z.max()-floor_height)/(z.max()-z.min())
        if length <= 16*tol:
            continue
        distance = float(np.linalg.norm(centers[a]-centers[b]))
        graph[a].append((int(b), distance))
        graph[b].append((int(a), distance))
    return graph

