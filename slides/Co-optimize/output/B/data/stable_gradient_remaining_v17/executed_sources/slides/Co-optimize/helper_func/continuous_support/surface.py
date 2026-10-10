"""Moving surface polygons cut directly by convex work/exit prism slices.

Interpolating the minimum of many prism fields at triangle corners misses
interior shadows. Here each prism is intersected with the entire triangle;
only two-dimensional contact polygons are combined, never support solids.
"""
from collections import OrderedDict
import numpy as np
from numba import njit
from shapely.geometry import Polygon, GeometryCollection, LineString
from shapely.ops import unary_union, polygonize
from shapely import set_precision
from trimesh.intersections import mesh_plane
from co_common import transform_points
from .contacts import exit_planes
from co_common import contact_boundary
from whole_search.common import unpack_solid


@njit(cache=True, boundscheck=True)
def prism_slices(triangle, normal, planes, counts, epsilon):
    output = np.zeros((len(counts), planes.shape[1]+3, 3))
    sizes = np.zeros(len(counts), np.int64)
    query = triangle
    for cell in range(len(counts)):
        levels = np.empty((3, counts[cell]))
        outside = False
        for axis in range(counts[cell]):
            plane = planes[cell, axis]
            for vertex in range(3):
                levels[vertex, axis] = query[vertex] @ plane[:3] + plane[3]
            # Contact with an exterior/tangential boundary is permitted;
            # an inward-facing cap blocks the actual contact surface.
            if np.max(np.abs(levels[:, axis])) <= 1e-12 and plane[:3] @ normal > 1e-12:
                outside = True
                break
            if np.min(levels[:, axis]) > 1e-12:
                outside = True
                break
        if outside:
            continue
        vertices = np.zeros((planes.shape[1]+3, 3))
        vertices[:3] = triangle
        size = 3
        for axis in range(counts[cell]):
            plane = planes[cell, axis]
            clipped = np.zeros_like(vertices)
            retained = 0
            for index in range(size):
                a = vertices[(index-1) % size]
                b = vertices[index]
                fa = a @ plane[:3] + plane[3]
                fb = b @ plane[:3] + plane[3]
                if abs(fa) <= 1e-12:fa=0.
                if abs(fb) <= 1e-12:fb=0.
                if (fa < 0 and fb > 0) or (fa > 0 and fb < 0):
                    clipped[retained] = a + fa/(fa-fb)*(b-a)
                    retained += 1
                if fb <= 0:
                    clipped[retained] = b
                    retained += 1
            vertices = clipped
            size = retained
            if size < 3:
                break
        if size >= 3:
            output[cell, :size] = vertices[:size]
            sizes[cell] = size
    return output, sizes


class SurfaceCuts:
    def __init__(self, model):
        self.model = model
        self.cache = OrderedDict()
        self.exit_cache = OrderedDict()
        self.seed_polygons = None
        self.wrap_polygons = {}
        self.body_cache = OrderedDict()
        self.shadow_cache = OrderedDict()
        self.frames = []
        self.triangles = []
        for triangle, normal in zip(model.mesh.triangles, model.mesh.face_normals):
            u = triangle[1]-triangle[0]
            u /= np.linalg.norm(u)
            frame = np.column_stack([u, np.cross(normal, u)])
            self.frames.append(frame)
            self.triangles.append(Polygon((triangle-triangle[0]) @ frame))
        if getattr(model, 'initialization_report', None) is not None:
            with np.load(model.initialization_report.parent/'contacts.npz') as saved:
                self.seed_polygons = self.contact_polygons(saved['triangles_mesh_m'], saved['source_faces'])

    def contact_polygons(self, triangles, sources):
        by_face = [[] for _ in self.triangles]
        for triangle, source in zip(triangles, sources):
            shape = Polygon((triangle-self.model.mesh.triangles[source, 0]) @ self.frames[source])
            if shape.area > 1e-18:
                by_face[source].append(shape)
        return [unary_union(parts) if parts else GeometryCollection() for parts in by_face]

    def material(self, layout, owner):
        # exact() uses the common Step3 seed until a pose is explicitly
        # re-seated. In particular, another pose's removed working faces do
        # NOT magically regain material during a direction-only change.
        if self.seed_polygons is not None and all(np.allclose(layout.placements[k], np.eye(4), atol=1e-12, rtol=0)
                                                 for k in layout.active):
            return self.seed_polygons
        if owner not in self.wrap_polygons:
            triangles, sources = contact_boundary(self.model.mesh, unpack_solid(self.model.wraps[owner]),
                                                  self.model.allowed[owner])
            self.wrap_polygons[owner] = self.contact_polygons(triangles, sources)
        # A conservative contact seed supplied by the owner's own fitted
        # wrap. Additional material from other wraps is never assumed.
        return self.wrap_polygons[owner]

    def body_section(self, relative, face):
        key = (relative.tobytes(), face)
        if key in self.body_cache:
            return self.body_cache[key]
        model = self.model
        triangle = model.mesh.triangles[face]
        normal = model.mesh.face_normals[face]
        query = transform_points(triangle + model.epsilon*normal, relative)
        if np.any(query.min(0) > model.mesh.bounds[1]) or np.any(query.max(0) < model.mesh.bounds[0]):
            return GeometryCollection()
        segments = mesh_plane(model.mesh, normal @ relative[:3, :3].T, query[0])
        inverse = np.linalg.inv(relative)
        local = transform_points(segments.reshape(-1, 3), inverse).reshape(-1, 2, 3)
        frame = self.frames[face]
        contours = [set_precision(LineString((line-triangle[0]) @ frame), 1e-12) for line in local]
        sections = list(polygonize(unary_union(contours)))
        shapes = []
        if sections:
            xy = np.array([[shape.representative_point().x, shape.representative_point().y] for shape in sections])
            points = xy @ frame.T + triangle[0] + model.epsilon*normal
            inside = model.ray.contains_points(transform_points(points, relative))
            shapes = [shape for shape, keep in zip(sections, inside) if keep]
        result = unary_union(shapes) if shapes else GeometryCollection()
        self.body_cache[key] = result
        if len(self.body_cache) > 6000:
            self.body_cache.popitem(last=False)
        return result

    def family(self, layout, owner):
        model = self.model
        parts = []
        sizes = []
        seen = set()
        work_already_cut=self.seed_polygons is not None and all(np.allclose(layout.placements[k],np.eye(4),atol=1e-12,rtol=0) for k in layout.active)
        for blocker in layout.active:
            relative = np.linalg.inv(layout.placements[blocker]) @ layout.placements[owner]
            work = model.work_rays[blocker]
            direction = layout.placements[blocker, :3, :3].T @ layout.directions[blocker]
            key = direction.tobytes()
            if key not in self.exit_cache:
                self.exit_cache[key] = exit_planes(model.mesh, direction, model.length)
                if len(self.exit_cache) > 64:
                    self.exit_cache.popitem(last=False)
            families=[self.exit_cache[key]] if work_already_cut else [(work.planes, work.counts), self.exit_cache[key]]
            for planes, counts in families:
                transformed = planes.copy()
                transformed[:, :, :3] = planes[:, :, :3] @ relative[:3, :3]
                transformed[:, :, 3] = planes[:, :, 3] + planes[:, :, :3] @ relative[:3, 3]
                identity = (transformed.tobytes(), counts.tobytes())
                if identity in seen:
                    continue
                seen.add(identity)
                parts.append(transformed)
                sizes.append(counts)
        width = max(p.shape[1] for p in parts)
        padded = []
        for part in parts:
            padded.append(np.pad(part, ((0, 0), (0, width-part.shape[1]), (0, 0))))
        return np.concatenate(padded), np.concatenate(sizes)

    def polygons(self, layout, owner):
        registered = all(np.allclose(layout.placements[k], np.eye(4), atol=1e-12, rtol=0) for k in layout.active)
        key = (layout.key(), layout.placements[owner].tobytes(), None if registered else owner)
        if key in self.cache:
            return self.cache[key]
        model = self.model
        shadow_key=(layout.key(),layout.placements[owner].tobytes())
        if shadow_key in self.shadow_cache:
            material=self.material(layout,owner)
            result=[base.difference(shadow).buffer(-model.epsilon) if not base.is_empty else GeometryCollection()
                    for base,shadow in zip(material,self.shadow_cache[shadow_key])]
            self.cache[key]=result
            if len(self.cache)>8:self.cache.popitem(last=False)
            return result
        planes, counts = self.family(layout, owner)
        material = self.material(layout, owner)
        relatives = {}
        for blocker in layout.active:
            relative = np.linalg.inv(layout.placements[blocker]) @ layout.placements[owner]
            if not np.allclose(relative, np.eye(4), atol=1e-12, rtol=0):
                relatives[relative.tobytes()] = relative
        result = [];shadows=[]
        for face, triangle in enumerate(model.mesh.triangles):
            pieces, sizes = prism_slices(triangle, model.mesh.face_normals[face],
                                         planes, counts, model.epsilon)
            shapes = []
            frame = self.frames[face]
            base = material[face]
            for vertices, size in zip(pieces, sizes):
                if size < 3:
                    continue
                shape = Polygon((vertices[:size]-triangle[0]) @ frame)
                if not shape.is_valid:
                    shape = shape.buffer(0)
                if shape.area > 1e-18:
                    shapes.append(shape)
            for relative in relatives.values():
                section = self.body_section(relative, face)
                if not section.is_empty:
                    shapes.append(section)
            shadow=unary_union(shapes) if shapes else GeometryCollection()
            shadows.append(shadow)
            free = base.difference(shadow) if shapes else base
            if not free.is_valid:
                free = free.buffer(0)
            # Numerical hairline slivers cannot supply a whole new normal
            # in an unbounded reaction cone. This is conservative guidance;
            # the original final geometry/load tolerances remain unchanged.
            result.append(free.buffer(-model.epsilon))
        self.cache[key] = result
        self.shadow_cache[shadow_key]=shadows
        if len(self.shadow_cache)>8:self.shadow_cache.popitem(last=False)
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return result

    def vertices(self, layout, owner, allowed):
        polygons = self.polygons(layout, owner)
        points = []
        normals = []
        area = 0.
        for face in allowed:
            shape = polygons[face]
            parts = [shape] if shape.geom_type == 'Polygon' else list(getattr(shape, 'geoms', []))
            for polygon in parts:
                if polygon.geom_type != 'Polygon' or polygon.area <= 1e-18:
                    continue
                xy = np.asarray(polygon.exterior.coords)[:-1]
                points.extend(xy @ self.frames[face].T + self.model.mesh.triangles[face, 0])
                normals.extend([self.model.mesh.face_normals[face]]*len(xy))
                area += polygon.area
        return np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3), area
