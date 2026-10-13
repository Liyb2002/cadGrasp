"""Cache fixed exclusion slices; reproduce SurfaceCuts contact polygons."""
from collections import OrderedDict
import time
import numpy as np
from numba import njit
from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import unary_union
from .surface import SurfaceCuts
from .contacts import exit_planes


@njit(cache=True, boundscheck=True)
def prism_slices(triangle, normal, planes, counts, epsilon):
    """Identical clipping, scalar 3D dot products instead of tiny BLAS calls."""
    width = planes.shape[1]+3
    output = np.zeros((len(counts), width, 3)); sizes = np.zeros(len(counts), np.int64)
    for cell in range(len(counts)):
        outside = False
        for axis in range(counts[cell]):
            plane = planes[cell, axis]
            lo = 1e300; hi = -1e300; magnitude = 0.
            for vertex in range(3):
                value = (triangle[vertex, 0]*plane[0]+triangle[vertex, 1]*plane[1]
                         +triangle[vertex, 2]*plane[2]+plane[3])
                lo = min(lo, value); hi = max(hi, value); magnitude = max(magnitude, abs(value))
            facing = plane[0]*normal[0]+plane[1]*normal[1]+plane[2]*normal[2]
            if (magnitude <= 1e-12 and facing > 1e-12) or lo > 1e-12:
                outside = True; break
        if outside: continue
        vertices = np.zeros((width, 3)); vertices[:3] = triangle; size = 3
        for axis in range(counts[cell]):
            plane = planes[cell, axis]; clipped = np.zeros_like(vertices); retained = 0
            for index in range(size):
                previous = (index-1) % size
                fa = vertices[previous, 0]*plane[0]+vertices[previous, 1]*plane[1]+vertices[previous, 2]*plane[2]+plane[3]
                fb = vertices[index, 0]*plane[0]+vertices[index, 1]*plane[1]+vertices[index, 2]*plane[2]+plane[3]
                if abs(fa) <= 1e-12: fa = 0.
                if abs(fb) <= 1e-12: fb = 0.
                if (fa < 0 and fb > 0) or (fa > 0 and fb < 0):
                    weight = fa/(fa-fb)
                    for j in range(3): clipped[retained, j] = vertices[previous, j]+weight*(vertices[index, j]-vertices[previous, j])
                    retained += 1
                if fb <= 0:
                    clipped[retained] = vertices[index]; retained += 1
            vertices = clipped; size = retained
            if size < 3: break
        if size >= 3:
            output[cell, :size] = vertices[:size]; sizes[cell] = size
    return output, sizes


class IncrementalSurfaceCuts(SurfaceCuts):
    def __init__(self, model):
        super().__init__(model)
        self.slice_cache = OrderedDict()
        self.slice_hits = self.slice_misses = 0
        self.seconds = 0.

    def slices(self, planes, counts, needed):
        key = (planes.tobytes(), counts.tobytes())
        if key in self.slice_cache:
            self.slice_hits += 1
            self.slice_cache.move_to_end(key)
            result = self.slice_cache[key]
        else:
            self.slice_misses += 1
            result = [None]*len(self.model.mesh.faces)
        for face in np.flatnonzero(needed):
            if result[face] is not None: continue
            triangle = self.model.mesh.triangles[face]
            pieces, sizes = prism_slices(triangle, self.model.mesh.face_normals[face],
                                         planes, counts, self.model.epsilon)
            shapes = []
            for vertices, size in zip(pieces, sizes):
                if size < 3: continue
                shape = Polygon((vertices[:size]-triangle[0]) @ self.frames[face])
                if not shape.is_valid: shape = shape.buffer(0)
                if shape.area > 1e-18: shapes.append(shape)
            result[face] = unary_union(shapes) if shapes else GeometryCollection()
        self.slice_cache[key] = result
        if len(self.slice_cache) > 256: self.slice_cache.popitem(last=False)
        return result

    def polygons(self, layout, owner, needed=None):
        registered = all(np.allclose(layout.placements[k], np.eye(4), atol=1e-12, rtol=0)
                         for k in layout.active)
        needed = np.ones(len(self.model.mesh.faces), bool) if needed is None else needed
        key = (layout.key(), layout.placements[owner].tobytes(), None if registered else owner, needed.tobytes())
        if key in self.cache: return self.cache[key]
        started = time.monotonic(); model = self.model
        material = self.material(layout, owner)
        shadow_key = (layout.key(), layout.placements[owner].tobytes(), needed.tobytes())
        if shadow_key not in self.shadow_cache:
            families = []; relatives = {}; seen = set()
            for blocker in layout.active:
                relative = np.linalg.inv(layout.placements[blocker]) @ layout.placements[owner]
                if not np.allclose(relative, np.eye(4), atol=1e-12, rtol=0):
                    relatives[relative.tobytes()] = relative
                direction = layout.placements[blocker, :3, :3].T @ layout.directions[blocker]
                dkey = direction.tobytes()
                if dkey not in self.exit_cache:
                    self.exit_cache[dkey] = exit_planes(model.mesh, direction, model.length)
                parts = [self.exit_cache[dkey]]
                if not (registered and self.seed_polygons is not None):
                    parts.insert(0, (model.work_rays[blocker].planes, model.work_rays[blocker].counts))
                for planes, counts in parts:
                    transformed = planes.copy()
                    transformed[:, :, :3] = planes[:, :, :3] @ relative[:3, :3]
                    transformed[:, :, 3] = planes[:, :, 3] + planes[:, :, :3] @ relative[:3, 3]
                    identity = (transformed.tobytes(), counts.tobytes())
                    if identity not in seen:
                        seen.add(identity); families.append(self.slices(transformed, counts, needed))
            shadows = []
            for face in range(len(model.mesh.faces)):
                if not needed[face]:
                    shadows.append(GeometryCollection()); continue
                shapes = [family[face] for family in families if not family[face].is_empty]
                for relative in relatives.values():
                    section = self.body_section(relative, face)
                    if not section.is_empty: shapes.append(section)
                shadows.append(unary_union(shapes) if shapes else GeometryCollection())
            self.shadow_cache[shadow_key] = shadows
            if len(self.shadow_cache) > 32: self.shadow_cache.popitem(last=False)
        result = []
        for base, shadow in zip(material, self.shadow_cache[shadow_key]):
            free = base.difference(shadow) if not shadow.is_empty else base
            if not free.is_valid: free = free.buffer(0)
            result.append(free.buffer(-model.epsilon))
        self.cache[key] = result
        if len(self.cache) > 32: self.cache.popitem(last=False)
        self.seconds += time.monotonic()-started
        return result

    def vertices(self, layout, owner, allowed):
        # A shared shadow is needed only on source faces that can supply a
        # reaction for some owner at this same placement. Full polygons()
        # remains available for geometry verification.
        model = self.model; needed = np.zeros(len(model.mesh.faces), bool)
        for member in layout.active:
            if not np.array_equal(layout.placements[member], layout.placements[owner]): continue
            direction = layout.placements[member, :3, :3].T @ layout.directions[member]
            faces = model.allowed[member]
            needed[faces[model.mesh.face_normals[faces] @ direction <= 1e-9]] = True
        registered = self.seed_polygons is not None and all(np.allclose(layout.placements[k], np.eye(4), atol=1e-12, rtol=0)
                                                          for k in layout.active)
        if registered:
            needed &= np.array([not polygon.is_empty for polygon in self.seed_polygons])
        polygons = self.polygons(layout, owner, needed)
        points = []; normals = []; area = 0.
        for face in allowed:
            shape = polygons[face]
            parts = [shape] if shape.geom_type == 'Polygon' else list(getattr(shape, 'geoms', []))
            for polygon in parts:
                if polygon.geom_type != 'Polygon' or polygon.area <= 1e-18: continue
                xy = np.asarray(polygon.exterior.coords)[:-1]
                points.extend(xy @ self.frames[face].T+model.mesh.triangles[face, 0])
                normals.extend([model.mesh.face_normals[face]]*len(xy)); area += polygon.area
        return np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3), area
