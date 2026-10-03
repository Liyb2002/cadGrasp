"""A small executable contact grammar with real-mesh parameter descent.

Rewrites change head count/surface charts. Parameters move centers and radii;
the compiler projects centers to admissible mesh faces and clips connected patches.
No soft contact activation or fictitious interpolated normals are used.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import nnls
import trimesh
from pathlib import Path

from step2_local_support import circles as P, surface as S, geometry as G
from step3_scheculer import contacts as I, passive_support as U
from step3_scheculer.floor_support import columns as floor_columns
from step3_scheculer import gpu_nnls
from step3_scheculer import exit_options as EXIT


def sources():
    return [Path(__file__), Path(gpu_nnls.__file__)]+EXIT.sources()


@dataclass(frozen=True)
class Patch:
    ident: int
    center: tuple
    radius: float


@dataclass(frozen=True)
class Program:
    patches: tuple
    angle: float = 0.

    def delete(self, index):
        return replace(self, patches=self.patches[:index]+self.patches[index+1:])

    def add(self, patch):
        if patch.ident in {p.ident for p in self.patches}:
            raise ValueError('Each physical head needs a unique identity')
        return replace(self, patches=self.patches+(patch,))

    def substitute(self, index, patch):
        return self.delete(index).add(patch)


def parameters(program, scale):
    return np.r_[np.array([(*p.center, p.radius) for p in program.patches]).ravel()/scale,
                 program.angle]


def from_parameters(program, values, scale):
    values = np.asarray(values)
    blocks = values[:-1].reshape(-1, 4)*scale
    return Program(tuple(Patch(p.ident, tuple(row[:3]), float(row[3]))
                         for p, row in zip(program.patches, blocks)), float(values[-1]))


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.])


def object_direction(world, task):
    rotation = np.asarray(task.domain.data['frame']['T_world_mesh'])[:3, :3]
    # Row-vector world -> original object coordinates. Object exit is -world.
    return -np.asarray(world)@rotation


def alignment(vectors):
    vectors = np.asarray(vectors)
    if len(vectors) < 2:
        return 0.
    return float(np.mean([1-np.clip(a@b, -1, 1)
                         for i, a in enumerate(vectors) for b in vectors[i+1:]]))


def cone_distance(full, targets):
    """Exact NNLS residual in the conditioned seven-row no-uplift system.

    This is a search loss, NEVER the final LP feasibility decision.
    """
    matrix = np.asarray(full, float).T
    targets = U.target(targets, matrix.shape[0])
    rows = []
    for target in targets:
        try:
            _, residual = nnls(matrix, target, maxiter=max(300, 12*matrix.shape[1]))
        except (RuntimeError, np.linalg.LinAlgError):
            from scipy.optimize import lsq_linear
            result = lsq_linear(matrix, target, bounds=(0, np.inf), tol=1e-10,
                                max_iter=150, lsq_solver='exact', method='bvls')
            residual = np.linalg.norm(matrix@result.x-target)
        rows.append(residual**2/max(float(target@target), 1e-12))
    return np.asarray(rows)


def reduced_rays(full, limit=48):
    """A deterministic subset of REAL rays, used only for inexpensive proposals."""
    if len(full) <= limit:
        return full
    rng = np.random.default_rng(42017)
    probes = np.vstack([np.eye(full.shape[1]), -np.eye(full.shape[1]),
                        rng.normal(size=(limit, full.shape[1]))])
    unit = full/np.maximum(np.linalg.norm(full, axis=1, keepdims=True), 1e-12)
    ids = np.unique(np.argmax(unit@probes.T, axis=0))
    return np.ascontiguousarray(full[ids])


class ConeBatch:
    """CUDA batched projected NNLS proposals; never a feasibility certificate."""
    def __init__(self, device='cpu', iterations=600):
        self.iterations = iterations
        self.device = device
        if device != 'cpu':
            import torch
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                if device == 'cuda':
                    raise RuntimeError('CUDA requested but unavailable')
                self.device = 'cpu'
            else:
                self.device = 'cuda'
                self.torch = torch
        self.info = dict(device=self.device, iterations=iterations,
            method='batched Lawson-Hanson active-set NNLS' if self.device == 'cuda' else 'SciPy NNLS',
            dtype='float64', feasibility_certificate=False)
        self.info['cpu_fallback_batches'] = 0
        if self.device == 'cuda':
            self.info['gpu'] = self.torch.cuda.get_device_name()

    def solve(self, fulls, targets):
        if self.device == 'cpu':
            return np.array([cone_distance(f, targets) for f in fulls])
        torch = self.torch
        nrows = fulls[0].shape[1]
        b = U.target(targets, nrows)
        m = max(len(f) for f in fulls)
        rays = np.zeros((len(fulls), m, nrows))
        for i, f in enumerate(fulls):
            rays[i, :len(f)] = f/np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        with torch.inference_mode():
            w = torch.as_tensor(rays, dtype=torch.float64, device='cuda')
            rhs = torch.as_tensor(b, dtype=torch.float64, device='cuda')[None]
            wt = w.transpose(1, 2)
            try:
                x = gpu_nnls.solve(torch, w, rhs, max_iterations=self.iterations)
            except RuntimeError as error:
                if isinstance(error, torch.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    self.info['cpu_fallback_batches'] += 1
                    return np.array([cone_distance(f, targets) for f in fulls])
                if 'NNLS inner active-set iteration limit' not in str(error):
                    raise
                self.info['cpu_fallback_batches'] += 1
                return np.array([cone_distance(f, targets) for f in fulls])
            gradient = (rhs-x@w)@wt
            violation = torch.maximum(gradient.clamp_min(0.).amax(dim=2),
                torch.where(x > 1e-9, gradient.abs(), 0.).amax(dim=2))
            if violation.max().item() > 1e-7:
                # Nearly dependent active sets can exhaust the numerical budget.
                self.info['cpu_fallback_batches'] += 1
                return np.array([cone_distance(f, targets) for f in fulls])
            residual = ((x@w-rhs)**2).sum(dim=2)/((rhs**2).sum(dim=2).clamp_min(1e-12))
            return residual.cpu().numpy()


class Compiler:
    def __init__(self, task, floor_tasks, clearance=.002, device='cpu', gpu_iterations=600, excluded_work_ids=None):
        self.task = task
        self.mesh = task.domain.mesh
        self.scale = float(self.mesh.extents.max())
        owner = np.asarray(task.domain.data['frame']['T_world_mesh'])
        self.transforms = [np.asarray(t.domain.data['frame']['T_world_mesh'])@np.linalg.inv(owner)
                           for t in floor_tasks]
        self.polygons = {}
        work = set(task.domain.work_ids.tolist() if excluded_work_ids is None else excluded_work_ids)
        for f, tri in enumerate(self.mesh.triangles):
            if f in work:
                continue
            poly = tri.copy()
            for transform in self.transforms:
                poly = G.clip_plane(poly, -transform[2], inset=clearance)
                if len(poly) < 3:
                    break
            if len(poly) >= 3 and S.area(poly) > self.mesh.area*1e-14:
                self.polygons[f] = poly
        if not self.polygons:
            raise ValueError('No non-working surface survives the saved floor-margin constraints')
        self.surface = P.SurfaceCircles(self.mesh, self.polygons)
        self.faces = np.array(list(self.polygons), int)
        self.triangles = np.concatenate([S.fan(poly) for poly in self.polygons.values()])
        self.triangle_faces = np.concatenate([np.full(len(poly), f, int)
                                              for f, poly in self.polygons.items()])
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.triangles.mean(axis=1))
        self.floor = U.floor(floor_columns(task.floor, task.domain.com), task.scale)
        self.cache = {}
        self.force_cache = {}
        self.backend = ConeBatch(device, gpu_iterations)
        self.exit_catalogue = EXIT.catalogue(self.mesh)
        self.exit_vectors = np.asarray(self.exit_catalogue['vectors'])
        # Numerical domain bounds, not manufacturing or pressure constraints.
        self.min_radius = self.scale*.003
        self.max_radius = self.scale*.6

    def project(self, center):
        center = np.asarray(center)
        _, ids = self.tree.query(center, k=min(48, len(self.triangles)))
        ids = np.atleast_1d(ids)
        near = trimesh.triangles.closest_point(self.triangles[ids], np.repeat(center[None], len(ids), axis=0))
        chosen = int(np.argmin(np.linalg.norm(near-center, axis=1)))
        # Move a tiny amount inside the chosen admissible triangle.
        point = (1-1e-7)*near[chosen]+1e-7*self.triangles[ids[chosen]].mean(axis=0)
        return point, int(self.triangle_faces[ids[chosen]])

    def compile_patch(self, patch):
        key = (patch.ident, *patch.center, patch.radius)
        if key in self.cache:
            return self.cache[key]
        center, face = self.project(patch.center)
        radius = float(np.clip(patch.radius, self.min_radius, self.max_radius))
        polygons, area = self.surface.at_radius(center, face, radius)
        normals_ok = bool(polygons) and P.normal_spread(self.mesh, polygons)['wrap_limit_satisfied']
        if not normals_ok:
            return None
        contact = dict(candidate_index=patch.ident, candidate_id=f'{self.task.pose}_D{patch.ident:03d}',
            center_m=center, center_face=face, radius_m=radius,
            triangles_m=np.concatenate([S.fan(p) for p in polygons.values()]),
            source_faces=np.concatenate([np.full(len(p), f, int) for f, p in polygons.items()]))
        contact['triangle_areas_m2'] = S.areas(contact['triangles_m'])
        self.cache[key] = contact
        if len(self.cache) > 2048:
            self.cache.pop(next(iter(self.cache)))
        return contact

    def compile(self, program):
        contacts = [self.compile_patch(p) for p in program.patches]
        if any(c is None for c in contacts):
            return None
        for i, a in enumerate(contacts):
            if any(np.linalg.norm(a['center_m']-b['center_m']) < self.scale*1e-6
                   for b in contacts[i+1:]):
                return None
        return contacts

    def seeds(self, count=48):
        centers, faces = S.surface_centers(self.polygons, count, self.mesh)
        result = []
        for i, (center, face) in enumerate(zip(centers, faces)):
            _, fitted = self.surface.fit_area(center, int(face), .01*self.mesh.area)
            result.append(Patch(1000+i, tuple(center), float(fitted['radius_m'])))
        return result

    def loss(self, program, sample_ids, reference=None, exits=False):
        return self.loss_batch([program], sample_ids, reference, exits)[0]

    def loss_batch(self, programs, sample_ids, reference=None, exits=False):
        """Compile true mesh patches on CPU, evaluate distinct cone losses together."""
        compiled = [self.compile(p) for p in programs]
        keys = [((tuple(sample_ids)), tuple((p.center, p.radius) for p in program.patches))
                for program in programs]
        missing = {}
        for key, contacts in zip(keys, compiled):
            if contacts is not None and key not in self.force_cache and key not in missing:
                missing[key] = reduced_rays(self.task.supply(contacts))
        if missing:
            residuals = self.backend.solve(list(missing.values()), self.task.targets[sample_ids])
            for key, rows in zip(missing, residuals):
                self.force_cache[key] = float(rows.mean()+.5*rows.max(initial=0.))
        result = [self._terms(p, c, self.force_cache.get(k, 100.), reference, exits)
                  for p, c, k in zip(programs, compiled, keys)]
        if len(self.force_cache) > 4096:
            self.force_cache.clear()
        return result

    def _terms(self, program, contacts, force, reference, exits):
        if contacts is None:
            return 100., dict(force=100., normal=0., alignment=0.)
        normal_loss = 0.
        if exits and contacts:
            normals = np.vstack([self.mesh.face_normals[np.unique(c['source_faces'])] for c in contacts])
            # Preserve ANY escape opening. No inter-pose parallelism reward.
            blocking = np.maximum(-(normals@self.exit_vectors.T).min(axis=0),0.)**2
            normal_loss = float(blocking.min())
        total = force+(.5*normal_loss if exits else 0.)
        return total, dict(force=force, normal=normal_loss, alignment=0.)


def descend(compiler, program, sample_ids, reference=None, exits=False, steps=12):
    """Projected numerical gradient descent with true-patch recompilation.

    Central differences cross real piecewise-smooth clipping boundaries. This
    is explicitly numerical differentiation, not an autodiff smooth-CAD claim.
    """
    x = parameters(program, compiler.scale)
    history = []
    for iteration in range(steps):
        current = from_parameters(program, x, compiler.scale)
        value, terms = compiler.loss(current, sample_ids, reference, exits)
        grad = np.zeros_like(x)
        perturbations, indices = [], []
        for j in range(len(x)):
            if j == len(x)-1:
                continue
            epsilon = .004 if j != len(x)-1 else .015
            plus, minus = x.copy(), x.copy()
            plus[j] += epsilon; minus[j] -= epsilon
            perturbations.extend([from_parameters(program, plus, compiler.scale),
                                  from_parameters(program, minus, compiler.scale)])
            indices.append((j, epsilon))
        batch = getattr(compiler, 'loss_batch', None)
        values = batch(perturbations, sample_ids, reference, exits) if batch else [
            compiler.loss(p, sample_ids, reference, exits) for p in perturbations]
        for i, (j, epsilon) in enumerate(indices):
            a, b = values[2*i][0], values[2*i+1][0]
            grad[j] = (a-b)/(2*epsilon)
        length = np.linalg.norm(grad)
        accepted = False
        if length > 1e-10:
            update = grad/max(length, 1.)
            for learning_rate in (.04, .015, .005, .001):
                trial = x-learning_rate*update
                trial[:-1].reshape(-1, 4)[:, 3] = np.clip(trial[:-1].reshape(-1, 4)[:, 3],
                    compiler.min_radius/compiler.scale, compiler.max_radius/compiler.scale)
                candidate = from_parameters(program, trial, compiler.scale)
                new_value = compiler.loss(candidate, sample_ids, reference, exits)[0]
                if new_value < value-1e-11:
                    x = trial; accepted = True; break
        history.append(dict(iteration=iteration, loss=value, **terms, gradient_norm=float(length), accepted=accepted))
        if not accepted:
            break
    return from_parameters(program, x, compiler.scale), history
