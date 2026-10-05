"""Audit edge connectivity of the actual clipped surface, independent of growth.

Union fragments inside each original face, then connect only along a positive
length of an original shared edge. Original-face adjacency alone is insufficient:
clipping may remove the shared part. Point touches and nearby walls stay separate.
"""
from __future__ import annotations
import numpy as np
from shapely import union_all
from shapely.geometry import Polygon, LineString, Point

REL_TOL = 1e-9


def components(mesh, triangles, source_faces, return_topology=False):
    tolerance = float(mesh.extents.max())*REL_TOL
    groups, frames, nodes, owners = {}, {}, [], []
    for face in np.unique(source_faces):
        ids = np.flatnonzero(source_faces == face)
        source = mesh.triangles[face]
        edges = np.roll(source, -1, axis=0)-source
        x = edges[np.linalg.norm(edges, axis=1).argmax()]
        x /= np.linalg.norm(x)
        basis = np.array([x, np.cross(mesh.face_normals[face], x)])
        origin = source[0]
        flat = (triangles[ids]-origin)@basis.T
        polygons = [Polygon(t) for t in flat]
        # The triangles share exact polygon vertices. Snapping each triangle
        # independently can split a circle fan at nearly collinear edges.
        # Keep these coordinates; apply tolerance only to adjacency below.
        merged = union_all(polygons)
        parts = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
        frames[int(face)] = (origin, basis)
        groups[int(face)] = []
        assigned = np.zeros(len(ids), int)
        for part in parts:
            if part.geom_type != 'Polygon' or part.is_empty:
                continue
            mask = np.array([part.buffer(3*tolerance).covers(Point(t.mean(axis=0))) for t in flat])
            if not mask.any():
                continue
            node = len(nodes)
            nodes.append(part); owners.append(ids[mask]); assigned += mask
            groups[int(face)].append(node)
        assert np.all(assigned == 1), ('Fragment vanished or overlaps audit components', int(face), assigned.tolist())
    graph = [set() for _ in nodes]
    lengths, shared_edges = [], []
    def ranges(face, node, segment):
        origin, basis = frames[face]
        xy = (segment-origin)@basis.T
        delta = xy[1]-xy[0]
        hit = LineString(xy).intersection(nodes[node].buffer(2*tolerance))
        parts = list(hit.geoms) if hasattr(hit, 'geoms') else [hit]
        result = []
        for part in parts:
            if part.geom_type == 'LineString' and not part.is_empty:
                t = ((np.array(part.coords)-xy[0])@delta)/(delta@delta)
                result.append((float(t.min()),float(t.max())))
        return result
    for (a,b), vertices in zip(mesh.face_adjacency,mesh.face_adjacency_edges):
        if int(a) not in groups or int(b) not in groups:
            continue
        segment = mesh.vertices[vertices]
        length = np.linalg.norm(segment[1]-segment[0])
        for left in groups[int(a)]:
            first = ranges(int(a),left,segment)
            for right in groups[int(b)]:
                second = ranges(int(b),right,segment)
                shared = max([max(0.,min(u[1],v[1])-max(u[0],v[0]))*length for u in first for v in second],default=0.)
                if shared > 16*tolerance:
                    graph[left].add(right); graph[right].add(left); lengths.append(shared)
                    shared_edges.append((left, right, shared))
    unseen, parts = set(range(len(nodes))), []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        todo, group = [seed], []
        while todo:
            node = todo.pop(); group.extend(owners[node].tolist())
            for neighbour in graph[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour); todo.append(neighbour)
        parts.append(np.array(sorted(group),dtype=int))
    assert sum(map(len,parts)) == len(triangles)
    report = dict(components=len(parts), source_pieces=len(nodes), shared_edges=len(lengths),
                  minimum_shared_edge_m=min(lengths) if lengths else None,
                  coordinate_tolerance_m=tolerance)
    if return_topology:
        # Reusable geometry for overlapping regions. A subset must still be
        # checked on its induced graph; full-surface connectivity is not enough.
        topology = dict(owner_faces=np.array([source_faces[ids[0]] for ids in owners], int),
                        graph=graph, edges=np.array(shared_edges).reshape(-1,3),
                        coordinate_tolerance_m=tolerance)
        return parts, report, topology
    return parts, report
