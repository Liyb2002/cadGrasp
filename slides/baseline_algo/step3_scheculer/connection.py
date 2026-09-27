"""Cached, sufficient connection witnesses sharing a certified head direction.

Architecture: strictly interior head sections extruded along withdrawal u to a
rear plane, broad joints, then a tree of thick beams beyond the object. Plane
inequalities certify every primitive, not just a zero-width centerline. Failure
means no witness in this architecture, never general geometric impossibility.
"""
from itertools import product
from pathlib import Path
import numpy as np
from step2_local_support import insertion as H, withdrawal as D
from step2_local_support import support_policy as POLICY
from step3_scheculer import contacts as I

SCHEMA = 'rear_plane_connection_v1'
ROOT_SHRINK = .85
FRAME_WIDTH_FRACTION = .05
FRAME_GAP_FRACTION = .02
CORNERS = np.array(list(product((-1., 1.), repeat=3)))
FLOOR_DOT_ERROR_FACTOR = 64.


def floor_slope_lower(vectors, normal):
    """Conservative dot-product bound for normalized, transformed directions.

    Catalogue tangents can have a tiny positive dot after normalization and
    coordinate transforms. Dividing a required millimetre lift by that noise
    fabricated astronomical rear planes. All extrusion distances are >= 0, so
    using a lower slope bound tightens every floor inequality, including truly
    descending near-tangents; it never rounds a downward ray into a free one.
    """
    vectors = np.asarray(vectors, float)
    normal = np.asarray(normal, float)
    error = FLOOR_DOT_ERROR_FACTOR*np.finfo(float).eps*np.maximum(
        1., np.abs(vectors)@np.abs(normal))
    return vectors@normal-error


def code_hashes():
    return {**D.code_hashes(), **I.hashes([Path(__file__), Path(POLICY.__file__)])}


class Checker:
    def __init__(self, mesh, depth, catalogue, *, floor_planes=None, work_ids=(),
                 work=None, half_width=None, gap=None):
        self.mesh, self.depth, self.catalogue = mesh, float(depth), catalogue
        self.vectors = np.asarray(catalogue['vectors'], float)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:
            raise ValueError('Expected 3-D direction vectors')
        np.testing.assert_allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-12)
        self.scale = float(mesh.extents.max())
        self.radius = .5*FRAME_WIDTH_FRACTION*self.scale if half_width is None else float(half_width)
        self.gap = FRAME_GAP_FRACTION*self.scale if gap is None else float(gap)
        if self.radius <= 0 or self.gap < 0:
            raise ValueError('Positive frame thickness and nonnegative clearance required')
        self.eps = self.scale*1e-9
        self.planes = np.array([[0., 0., 1., 0.]] if floor_planes is None else floor_planes, float)
        if self.planes.ndim != 2 or self.planes.shape[1] != 4 or not np.isfinite(self.planes).all():
            raise ValueError('Floor planes are rows [nx,ny,nz,b] with n.x+b >= 0')
        lengths = np.linalg.norm(self.planes[:, :3], axis=1)
        if np.any(lengths == 0):
            raise ValueError('Floor normal cannot be zero')
        self.planes /= lengths[:, None]
        self.work_ids = set(map(int, work_ids))
        self.work = work
        self.installation = catalogue.get('installation')
        self.analyzer = H.Analyzer(mesh, depth)
        self.support = np.max(np.asarray(mesh.vertices)@self.vectors.T, axis=0)
        self.base_lower = self.support+self.gap+self.radius*np.abs(self.vectors).sum(axis=1)
        self.cache = {}

    def head(self, contact):
        key = D.signature(contact, self.depth)
        if key in self.cache:
            return self.cache[key]
        heads = self.analyzer.heads(contact)
        vertices = [np.asarray(h.vertices) for h in heads]
        points = np.vstack(vertices)
        centers = np.array([v.mean(axis=0) for v in vertices])
        roots = [p+ROOT_SHRINK*(v-p) for p,v in zip(centers,vertices)]
        terminal = centers.mean(axis=0)
        projections = centers@self.vectors.T
        lower = np.maximum(self.base_lower, projections.max(axis=0))
        upper = np.full(len(self.vectors), np.inf)
        # Every shifted root, not only its center, must lie behind O.
        for root, center in zip(roots, centers):
            lower = np.maximum(lower, self.support+self.gap-np.min((root-center)@self.vectors.T, axis=0))
        valid = bool(heads) and not bool(self.work_ids.intersection(map(int, contact['source_faces'])))
        if self.installation:
            plane=np.asarray(self.installation['floor_plane'])
            valid &= np.min(np.asarray(contact['triangles_m'])@plane[:3]+plane[3]) >= self.installation['contact_floor_clearance_m']-self.eps
        min_height = None
        for plane in self.planes:
            a,b = plane[:3],plane[3]
            height = float(np.min(points@a+b))
            min_height = height if min_height is None else min(min_height,height)
            valid &= height >= -self.eps
            slope = floor_slope_lower(self.vectors, a)
            down, up = slope < 0., slope > 0.
            flat = ~(down | up)
            root_heights = np.array([np.min(r@a+b) for r in roots])
            if down.any():
                upper[down] = np.minimum(upper[down], np.min(
                    projections[:,down]+root_heights[:,None]/(-slope[down]), axis=0))
            # Axis-aligned cubes: support radius is r*||a||_1, not r.
            need = self.radius*np.abs(a).sum()+self.gap-(terminal@a+b)
            terminal_projection = terminal@self.vectors.T
            if up.any():
                lower[up] = np.maximum(lower[up], terminal_projection[up]+need/slope[up])
            if down.any():
                upper[down] = np.minimum(upper[down], terminal_projection[down]+need/slope[down])
            if need > 0:
                upper[flat] = -np.inf
        row = dict(signature=key, heads=heads, centers=centers, roots=roots,
                   terminal=terminal, lower=lower, upper=upper, valid=valid,
                   minimum_head_floor_height_m=min_height)
        self.cache[key] = row
        return row

    def _witness(self, rows, index, plane):
        return dict(kind='rear_plane_tree', direction_id=int(index),
                    withdrawal_direction=self.vectors[index].tolist(), plane_offset_m=float(plane),
                    frame_half_width_m=self.radius, object_frame_clearance_m=self.gap,
                    root_shrink=ROOT_SHRINK,
                    geometry_signatures=[r['signature'] for r in rows])

    def parts(self, contacts, witness):
        """Materialize the positive-volume witness for audit/optional obstacles."""
        rows = [self.head(c) for c in contacts]
        if witness['kind'] == 'single_head':
            return [h for row in rows for h in row['heads']]
        u = self.vectors[witness['direction_id']]; q = witness['plane_offset_m']
        parts, cubes = [], []
        for row in rows:
            ends=[]
            for p,root in zip(row['centers'],row['roots']):
                end=root+(q-p@u)*u
                parts.append(H.engine.hull_mesh(np.vstack([root,end])))
                ends.append(end)
            center=row['terminal']+(q-row['terminal']@u)*u
            cube=center+self.radius*CORNERS
            cubes.append(cube)
            parts.append(H.engine.hull_mesh(np.vstack([*ends,cube])))
        for cube in cubes[1:]:
            parts.append(H.engine.hull_mesh(np.vstack([cubes[0],cube])))
        return parts

    def check(self, contacts, allowed):
        ids = np.asarray(allowed['ids'], int)
        result = dict(schema=SCHEMA, passed=False, directions=D.normalize(), witness=None,
                      floor_planes=self.planes.tolist(), finite_thickness=True,
                      floor_slope_roundoff_guard=dict(factor=FLOOR_DOT_ERROR_FACTOR,
                          policy='Use dot-product lower bound for nonnegative extrusion distances'),
                      architecture='interior_necks_rear_joints_and_beam_tree',
                      scope='Sufficient construction conditional on shared full-head sweeps; no arbitrary-path impossibility claim.',
                      work_access_volume_enforced=self.work is not None)
        if not contacts:
            return dict(result, status='no_contacts')
        if not len(ids):
            return dict(result, status='no_common_head_direction')
        rows = [self.head(c) for c in contacts]
        if not all(r['valid'] for r in rows):
            return dict(result, status='head_in_forbidden_region')
        if len(rows) == 1:
            # A single, connected fitted head needs no additional bridge.
            if self.work is not None and not self.work.check_parts(rows[0]['heads'])['passed']:
                return dict(result, status='head_work_access_not_verified')
            return dict(result, passed=True, status='single_head_connected', directions=D.normalize(ids),
                        witness=dict(kind='single_head', geometry_signatures=[rows[0]['signature']]))
        low=np.max([r['lower'][ids] for r in rows], axis=0)
        high=np.min([r['upper'][ids] for r in rows], axis=0)
        good=np.flatnonzero(low+self.eps < high)
        witnesses=[]
        for offset in good:
            step=min(.005*self.scale, .5*(high[offset]-low[offset]))
            q=low[offset]+step
            witness=self._witness(rows,ids[offset],q)
            if self.work is not None and not self.work.check_parts(self.parts(contacts,witness))['passed']:
                continue
            witnesses.append(witness)
        if not witnesses:
            return dict(result, status='no_connection_witness')
        result.update(passed=True, status='connected_insertion_witness',
                      directions=D.normalize(w['direction_id'] for w in witnesses), witness=witnesses[0])
        return result


def for_problem(problem, catalogue=None):
    """One cache per immutable case, shared by filtering and all radius trials."""
    if hasattr(problem, '_connection_checker'):
        return problem._connection_checker
    from step2_local_support import insertion_directions as K
    cat=K.read(problem.name) if catalogue is None else catalogue
    work=None
    if POLICY.ENFORCE_PROCESS_ACCESS:
        from step2_local_support import work_volume as W
        from step1.cases import pose_name
        from step1.needs import OUTPUTS
        work=W.WorkVolume.read(OUTPUTS/problem.name/pose_name()/W.STAGE/'work_volume.json')
    initial=cat['direction_catalogue'].get('installation')
    planes=[[0.,0.,1.,0.],initial['floor_plane']] if initial else None
    problem._connection_checker=Checker(problem.domain.mesh,cat['normal_depth_m'],cat['direction_catalogue'],
        floor_planes=planes,work_ids=problem.domain.work_ids,work=work)
    return problem._connection_checker
