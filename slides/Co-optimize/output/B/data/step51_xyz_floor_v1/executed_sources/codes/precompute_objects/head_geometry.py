"""A shared object-attached head catalogue, evaluated at two task poses.

Canonical coordinates are the first task's world frame. This is a deliberately
restricted two-pose baseline, not a search over independent fixture placements.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from step2_local_support import circles as P, surface as S, geometry as G
from step2_local_support import withdrawal as W
from step3_scheculer import contacts as I


def transform_contact(contact, transform):
    result = dict(contact)
    result.pop('wrench_generators', None)
    result.pop('wrench_com_m', None)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result['triangles_m'] = contact['triangles_m']@rotation.T+translation
    result['center_m'] = contact['center_m']@rotation.T+translation
    return result


def horizontal_catalogue(mesh):
    vectors = []
    def add(value):
        value = np.asarray(value, float).copy()
        value[2] = 0.
        length = np.linalg.norm(value)
        if length < 1e-10:
            return
        value /= length
        if all(np.linalg.norm(value-other) > 1e-7 for other in vectors):
            vectors.append(value)
    for angle in np.deg2rad(np.arange(0, 360, 5)):
        add([np.cos(angle), np.sin(angle), 0])
    # A bounded menu, plus exact tangents on prominent facets. Enumerating every
    # tessellation normal adds thousands of nearly identical expensive sweeps.
    faces = np.argsort(-mesh.area_faces, kind='stable')[:24]
    for normal in np.unique(np.round(mesh.face_normals[faces], 10), axis=0):
        add(normal)
        add(np.cross(normal, [0, 0, 1]))
        add(np.cross([0, 0, 1], normal))
    return dict(vectors=np.asarray(vectors).tolist(),
        global_allowed_directions=W.normalize(range(len(vectors))),
        preferred_withdrawal_direction=None,
        physical_motion='object inserts along +d into fixed heads; equivalent relative head withdrawal +d',
        resolution='5 degree horizontal grid plus normal/tangent seeds of 24 largest triangles',
        finite_horizontal_menu=True, installation=None)


class FreePaths:
    """A finite 3-D roadmap of checked straight segments, not a beam template.

Paths avoid the object and both ground halfspaces. Work surfaces are part of the
object boundary; additional tool-ray volumes are disabled by the baseline model.
No path found is an unresolved finite-roadmap result, not an impossibility proof.
"""
    def __init__(self, mesh, planes, resolution=13):
        self.mesh, self.planes = mesh, np.asarray(planes)
        self.scale = float(mesh.extents.max())
        self.tol = self.scale*1e-8
        axes = [np.linspace(lo-.45*self.scale, hi+.45*self.scale, resolution)
                for lo, hi in mesh.bounds.T]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        points = grid.reshape(-1, 3)
        ground_ok = (points@self.planes[:, :3].T+self.planes[:, 3] > self.tol).all(axis=1)
        valid = ground_ok.copy()
        valid[ground_ok] &= ~mesh.contains(points[ground_ok])
        indices = np.arange(len(points)).reshape(grid.shape[:3])
        edges = []
        for axis in range(3):
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            edges.append(np.c_[indices[tuple(a)].ravel(), indices[tuple(b)].ravel()])
        edges = np.concatenate(edges)
        edges = edges[valid[edges].all(axis=1)]
        edges = edges[self.clear_segments(points[edges[:, 0]], points[edges[:, 1]])]
        graph = coo_matrix((np.ones(2*len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(points), len(points))).tocsr()
        _, labels = connected_components(graph, directed=False)
        self.points, self.labels = points[valid], labels[valid]
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.record = dict(kind='checked_3d_roadmap', resolution=resolution,
            node_count=int(valid.sum()), edge_count=len(edges),
            component_count=len(set(self.labels.tolist())),
            finite_thickness_verified=False, insertion_of_connectors_verified=False,
            tool_ray_volume_enforced=False,
            negative_verdict='no witness in finite roadmap; continuous connectivity unresolved')

    def clear_segments(self, starts, ends):
        starts, ends = np.asarray(starts).reshape(-1, 3), np.asarray(ends).reshape(-1, 3)
        delta = ends-starts
        lengths = np.linalg.norm(delta, axis=1)
        good = lengths > self.tol
        ids = np.flatnonzero(good)
        if not len(ids):
            return good
        directions = delta[ids]/lengths[ids, None]
        hits, rays, _ = self.mesh.ray.intersects_location(starts[ids], directions, multiple_hits=False)
        if len(hits):
            distances = np.linalg.norm(hits-starts[ids[rays]], axis=1)
            good[ids[rays[distances <= lengths[ids[rays]]+self.tol]]] = False
        return good

    def ports(self, point):
        if self.tree is None or np.any(point@self.planes[:, :3].T+self.planes[:, 3] <= self.tol):
            return []
        _, ids = self.tree.query(point, k=min(96, len(self.points)))
        ids = np.atleast_1d(ids)
        good = self.clear_segments(np.broadcast_to(point, (len(ids), 3)), self.points[ids])
        return sorted(set(self.labels[ids[good]].tolist()))


class PairGeometry:
    def __init__(self, problems, count=200, initialize_candidates=True, *, head_exclusion_problems=None, use_precomputed=True):
        self.problems = problems
        self.mesh = problems[0].domain.mesh
        self.scale = float(self.mesh.extents.max())
        self.depth = self.scale*P.DEPTH_FRACTION
        self.target_area = float(self.mesh.area)*P.AREA_FRACTION
        transforms = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
        self.transforms = [t@np.linalg.inv(transforms[0]) for t in transforms]
        for problem, transform in zip(problems, self.transforms):
            np.testing.assert_array_equal(self.mesh.faces, problem.domain.mesh.faces)
            np.testing.assert_allclose(self.mesh.vertices@transform[:3, :3].T+transform[:3, 3],
                                       problem.domain.mesh.vertices, atol=1e-12, rtol=0)
        self.planes = np.array([t[2] for t in self.transforms])
        # The tasks that use a head for load support can be fewer than the tasks
        # in which its material must avoid work surfaces and the ground.
        exclusions = problems if head_exclusion_problems is None else head_exclusion_problems
        head_transforms = [np.asarray(p.domain.data['frame']['T_world_mesh'])@np.linalg.inv(transforms[0])
                           for p in exclusions]
        for problem, transform in zip(exclusions, head_transforms):
            np.testing.assert_array_equal(self.mesh.faces, problem.domain.mesh.faces)
            np.testing.assert_allclose(self.mesh.vertices@transform[:3, :3].T+transform[:3, 3],
                                      problem.domain.mesh.vertices, atol=1e-12, rtol=0)
        self.head_planes = np.array([t[2] for t in head_transforms])
        self.head_exclusion_poses = [p.pose for p in exclusions]
        work = set(np.concatenate([p.domain.work_ids for p in exclusions]).tolist())
        polygons = {}
        for face, triangle in enumerate(self.mesh.triangles):
            if face in work:
                continue
            polygon = triangle.copy()
            for plane in self.head_planes:
                polygon = G.clip_plane(polygon, -plane, inset=S.FLOOR_CLEARANCE_M)
                if len(polygon) < 3:
                    break
            if len(polygon) >= 3 and S.area(polygon) > self.mesh.area*1e-16:
                polygons[face] = polygon
        if not polygons:
            raise ValueError('No shared contact surface remains after work/floor exclusions')
        self.centers, self.faces, self.sampling = S.surface_centers(polygons, count, self.mesh, True)
        self.surface = P.SurfaceCircles(self.mesh, polygons)
        self.clearance = P.LocalClearance(self.mesh, self.depth)
        self.catalogues = [horizontal_catalogue(p.domain.mesh) for p in problems]
        # Omit installation entirely: W.Analyzer's absent-installation branch uses task ground.
        for catalogue in self.catalogues:
            catalogue.pop('installation')
        self.analyzers = [W.Analyzer(p.domain.mesh, self.depth, cat)
                          for p, cat in zip(problems, self.catalogues)]
        self.precomputed = None
        if use_precomputed and len(problems)==1 and head_exclusion_problems is None and 'object' in problems[0].domain.data:
            from codes.precompute_objects import head_cache as HC
            folder=HC.ROOT/'objects'/problems[0].domain.data['object']/'poses'/problems[0].pose
            self.precomputed=HC.load(folder,count)
        if self.precomputed:
            self.paths=FreePaths.__new__(FreePaths)
            self.paths.mesh,self.paths.planes=self.mesh,self.planes
            self.paths.scale,self.paths.tol=self.scale,self.scale*1e-8
            self.paths.points=self.precomputed['arrays']['roadmap_points']
            self.paths.labels=self.precomputed['arrays']['roadmap_labels']
            self.paths.tree=cKDTree(self.paths.points) if len(self.paths.points) else None
            self.paths.record=self.precomputed['roadmap']
            np.testing.assert_array_equal(self.centers,self.precomputed['arrays']['centers'])
            np.testing.assert_array_equal(self.faces,self.precomputed['arrays']['center_faces'])
        else:
            self.paths = FreePaths(self.mesh, self.planes)
        self.cache = {}
        self.initial = []
        if not initialize_candidates:
            return
        if self.precomputed:
            from codes.precompute_objects.head_cache import pool
            self.initial=pool(self.precomputed,P.AREA_FRACTION,self.catalogues[0])
            return
        for index, (center, face) in enumerate(zip(self.centers, self.faces)):
            # Fit the requested area once. A wrap/clearance failure at this
            # size rejects the candidate; it must not silently shrink the head.
            patch, fitted = self.surface.fit_area(center, int(face), self.target_area)
            entry = self.make(index, fitted['radius_m'], patch)
            entry['area_fit'] = fitted
            self.initial.append(entry)
            if (index+1) % 20 == 0:
                print('pair geometry', index+1, '/', count,
                      'legal', sum(e['valid'] for e in self.initial), flush=True)

    def make(self, index, radius, polygons=None, target_area=None):
        target_area = self.target_area if target_area is None else float(target_area)
        key = (int(index), float(radius), target_area)
        if key in self.cache:
            return self.cache[key]
        if polygons is None:
            polygons, _ = self.surface.at_radius(self.centers[index], int(self.faces[index]), radius)
        contact = dict(candidate_index=int(index), candidate_id=f'C{index+1:03d}',
            center_m=self.centers[index].copy(), center_face=int(self.faces[index]), radius_m=float(radius),
            triangles_m=np.concatenate([S.fan(p) for p in polygons.values()]) if polygons else np.empty((0, 3, 3)),
            source_faces=np.concatenate([np.full(len(p), f, int) for f, p in polygons.items()]) if polygons else np.empty(0, int))
        contact['triangle_areas_m2'] = S.areas(contact['triangles_m'])
        entry = dict(contact=contact, valid=False, reason='fixed_area_unavailable', area=I.area([contact]))
        entry['area_fraction'] = entry['area']/self.mesh.area
        entry['relative_area_error'] = abs(entry['area']-target_area)/target_area
        self.cache[key] = entry
        if entry['relative_area_error'] > P.AREA_REL_TOL:
            return entry
        check = self.clearance.check(polygons)
        entry['local_clearance'] = check
        if not check['valid']:
            entry['reason'] = check['status']
            return entry
        cells = [G.head_cell(self.mesh, p, f, self.clearance.offsets) for f, p in polygons.items()]
        points = np.concatenate(cells)
        if (points@self.head_planes[:, :3].T+self.head_planes[:, 3]).min() < -self.scale*1e-10:
            entry['reason'] = 'head_hits_other_task_floor'
            return entry
        tri = self.mesh.triangles[contact['center_face']]
        import trimesh
        weights = trimesh.triangles.points_to_barycentric(tri[None], contact['center_m'][None])[0]
        root = contact['center_m']+.5*weights@self.clearance.offsets[self.mesh.faces[contact['center_face']]]
        entry['root_m'] = root
        entry['path_components'] = self.paths.ports(root)
        if not entry['path_components']:
            entry['reason'] = 'no_roadmap_path_witness'
            return entry
        records = []
        for analyzer, transform in zip(self.analyzers, self.transforms):
            records.append(analyzer.analyze(transform_contact(contact, transform)))
        entry['directions'] = [r['certified_directions']['ids'] for r in records]
        entry['direction_records'] = records
        entry['valid'] = all(entry['directions'])
        entry['reason'] = 'valid' if entry['valid'] else 'no_horizontal_head_insertion_witness'
        return entry

    def expand(self, entry, area_factor):
        """Fit a slightly larger area at the same center, with fresh geometry checks."""
        from step3_scheculer.terminal_expansion import AREA_FACTORS
        if area_factor not in AREA_FACTORS:
            raise ValueError('Expansion must use the bounded terminal area menu')
        index = entry['contact']['candidate_index']
        target = self.target_area*area_factor
        patch, fitted = self.surface.fit_area(self.centers[index], int(self.faces[index]), target)
        trial = self.make(index, fitted['radius_m'], patch, target_area=target)
        trial['area_fit'] = fitted
        return trial

    def group_check(self, entries):
        if any(not e['valid'] for e in entries):
            return dict(passed=False, reason='invalid_head')
        directions = [set(range(len(c['vectors']))) for c in self.catalogues]
        components = set(self.paths.labels.tolist())
        for entry in entries:
            components &= set(entry['path_components'])
            for k in range(len(self.problems)):
                directions[k] &= set(entry['directions'][k])
        return dict(passed=bool(components) and all(directions),
            reason='passed' if components and all(directions) else 'no_common_path_or_direction',
            common_path_components=sorted(components), common_direction_ids=[sorted(d) for d in directions],
            scope='thin terminal paths and per-task head insertion only; complete support not constructed')

    def contacts_by_pose(self, entries):
        return [[transform_contact(e['contact'], t) for e in entries] for t in self.transforms]
