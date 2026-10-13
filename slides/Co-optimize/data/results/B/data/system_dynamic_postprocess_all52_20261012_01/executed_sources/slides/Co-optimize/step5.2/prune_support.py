"""Delete redundant contact material with the original full-demand certificate.

Frozen poses, exits, ground reactions and all 32,768 demands are preserved.
Whole disconnected pieces are tried first, followed by bounded spatial cuts.
A needed isolated piece is retained for Step5.3 rather than falsely accepted.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
import time
import numpy as np

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2'), str(COOPT/'step5.3')]
import _bootstrap
from deterministic_base import solid, unpack, volume, union, box, VOLUME_TOL
from whole_search.common import C, transform_mesh
from whole_search.classify import classify


def material_components(value):
    """Positive connected components intersected back with the signed solid.

    This keeps real internal cavities when a negative boundary is present;
    assembling positive boundary shells alone would accidentally add material.
    """
    parts=list(value.decompose())
    signed=[float(part.volume())*C.S.SCALE**3 for part in parts]
    positive_count=sum(amount>VOLUME_TOL for amount in signed)
    if positive_count==1:
        # A single material body already includes its genuine inner cavities.
        # Intersecting its outer boundary back into itself is redundant.
        return [value]
    has_cavity=any(amount < -VOLUME_TOL for amount in signed)
    # Without a genuine inverse boundary each positive part already IS its
    # exact material component. A coincident intersection can manufacture
    # slivers and repeatedly multiply triangles on nearly aligned floors.
    return sorted([part ^ value if has_cavity else part
                   for part,amount in zip(parts,signed) if amount>VOLUME_TOL],
                  key=volume,reverse=True)


class ForceCertificate:
    def __init__(self, poses, layout, body, rows, name='B'):
        self.layout = layout
        self.mesh = body
        self.states = [C.state(name, pose) for pose in poses]
        self.tasks = [row[0] for row in self.states]
        self.native = np.array([row[1] for row in self.states])
        self.allowed = [np.setdiff1d(np.arange(len(body.faces)), task.domain.work_ids)
                        for task in self.tasks]
        self.floors = [C.U.floor(C.FLOOR.columns(task.floor, task.domain.com), task.scale)
                       for task in self.tasks]
        self.grounded = [float((body.vertices@row['object_transform'][:3, :3].T+
                               row['object_transform'][:3, 3])[:, 2].min()) <= 1e-9
                         for row in rows]
        self.bases = {k: [] for k in layout.active}
        self.cache = {}
        self.calls = self.classifications = self.reused = 0

    def evaluate_candidate(self,mesh):
        return self.evaluate(mesh,reject_early=True)

    def evaluate(self, mesh, reject_early=False):
        self.calls += 1
        masks, rays, contacts = {}, {}, {}
        for k in self.layout.active:
            Q = self.layout.placements[k]
            direction = Q[:3, :3].T@self.layout.directions[k]
            ids = self.allowed[k][self.mesh.face_normals[self.allowed[k]]@direction <= 1e-9]
            triangles, sources = C.contact_boundary(transform_mesh(self.mesh, Q), mesh, ids)
            points = C.transform_points(triangles.reshape(-1, 3), self.native[k]@np.linalg.inv(Q))
            normals = np.repeat(-self.tasks[k].domain.mesh.face_normals[sources], 3, axis=0)
            raw = np.c_[normals, np.cross(points-self.tasks[k].domain.com, normals)]
            floor = self.floors[k] if self.grounded[k] else self.floors[k][-1:]
            full = C.I.merge_columns(floor, C.U.heads(raw, self.tasks[k].scale))
            fingerprint = (k, hashlib.sha256(full.tobytes()).hexdigest())
            if fingerprint in self.cache:
                mask = self.cache[fingerprint]
                self.reused += 1
            else:
                column_ids = [int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), 'little')
                              for row in full]
                if len(set(column_ids)) != len(column_ids):
                    raise ValueError('Reaction-column identity collision')
                mask, _ = classify(full, self.tasks[k].targets,
                    basis_cache=self.bases[k], column_ids=column_ids,
                    stop_on_counterexample=reject_early)
                self.cache[fingerprint] = mask
                self.classifications += 1
            masks[k] = mask
            rays[k] = full
            contacts[k] = len(triangles)
            if not mask.all():
                return dict(passed=False, counts={str(j):int(v.sum()) for j,v in masks.items()},
                            failed_pose=k, masks=masks, supplies=rays, contact_triangles=contacts)
        return dict(passed=True, counts={str(k):int(v.sum()) for k,v in masks.items()},
                    masks=masks, supplies=rays, contact_triangles=contacts)


class Pruner:
    def __init__(self, fixture, witness, certificate, budget=64, grids=(3, 5)):
        self.fixture = fixture
        self.original = solid(fixture)
        self.allowed = solid(witness)
        # An already legal boundary must keep its exact contact polygons;
        # a redundant coincident Boolean can change a near-boundary witness.
        self.body = (self.original if volume(self.original-self.allowed) <= 1e-10
                     else self.original ^ self.allowed)
        self.certificate = certificate
        self.connected_input = len(material_components(self.body)) == 1
        self.budget = budget
        self.grids = grids
        self.history = []
        self.attempts = 0
        self.proof = None
        self.began = time.monotonic()

    def accept(self, candidate, operation):
        if self.attempts >= self.budget or volume(candidate) <= VOLUME_TOL:
            return False
        from material_builder_closed import regularize_external_boundaries
        candidate,_=regularize_external_boundaries(candidate)
        if volume(self.body-candidate) <= VOLUME_TOL:
            return False
        if volume(candidate-self.body) > 1e-10:
            self.attempts += 1
            self.history.append(dict(operation=operation,accepted=False,
                error='Numerical deletion Boolean added material; original body retained'))
            return False
        # Do not cut a connected main body into force-required islands. The
        # largest surviving component is tried separately by the caller.
        if self.connected_input and len(material_components(candidate)) > 1:
            return False
        self.attempts += 1
        try:
            mesh = unpack(candidate)
            if not mesh.is_watertight:
                return False
            proof = (self.certificate.evaluate_candidate(mesh)
                     if hasattr(self.certificate,'evaluate_candidate') else self.certificate.evaluate(mesh))
        except (ValueError, RuntimeError) as error:
            self.history.append(dict(operation=operation, accepted=False, error=str(error)))
            return False
        row = dict(operation=operation, accepted=proof['passed'], counts=proof['counts'],
                   counts_are_lower_bounds_on_rejection=hasattr(self.certificate,'evaluate_candidate')and not proof['passed'],
                   removed_cm3=volume(self.body-candidate)*1e6,
                   component_count=len(material_components(candidate)))
        self.history.append(row)
        if proof['passed']:
            self.body = candidate
            self.proof = proof
            self.best_mesh=mesh
            print('PRUNE', operation, round(row['removed_cm3'], 3), 'cm3', flush=True)
            return True
        return False

    def remove_islands(self):
        parts = material_components(self.body)
        if len(parts) <= 1:
            return
        if self.accept(parts[0], 'keep_connected_main_body'):
            return
        for part in parts[1:]:
            self.accept(self.body-part, 'delete_disconnected_component')

    def run(self):
        baseline_mesh = self.fixture if self.body is self.original else unpack(self.body)
        self.proof = self.certificate.evaluate(baseline_mesh)
        self.best_mesh=baseline_mesh
        if not self.proof['passed']:
            raise ValueError('Input contact material does not pass all original demands: '+
                             str(self.proof['counts']))
        initial_components = len(material_components(self.body))
        self.remove_islands()
        low, high = unpack(self.body).bounds
        for grid in self.grids:
            widths = (high-low)/grid
            candidates = []
            for index in np.ndindex(grid, grid, grid):
                a = low+np.array(index)*widths
                b = a+widths
                cutter = box(a, b)
                removed = volume(self.body ^ cutter)
                if removed > VOLUME_TOL:
                    candidates.append((removed, index, cutter))
            for _, index, cutter in sorted(candidates, key=lambda row: (-row[0], row[1])):
                if self.attempts >= self.budget:
                    break
                candidate = self.body-cutter
                # Prefer a single remaining body when a cut separates a now
                # redundant island. Both variants need the original forces.
                pieces = material_components(candidate)
                if len(pieces) > 1 and self.accept(pieces[0], f'grid_{grid}_{index}_main_body'):
                    continue
                if self.accept(candidate, f'grid_{grid}_{index}'):
                    self.remove_islands()
            if self.attempts >= self.budget:
                break
        from material_builder_closed import regularize_external_boundaries
        regularized, discarded = regularize_external_boundaries(self.body)
        cleanup_rejected=False
        if discarded:
            cleaned = unpack(regularized)
            proof = self.certificate.evaluate(cleaned)
            if not proof['passed']:
                cleanup_rejected=True
                discarded=[]
            else:
                self.body, self.proof = regularized, proof
                self.best_mesh=cleaned
        mesh = self.best_mesh
        parts = material_components(self.body)
        record = dict(passed=self.proof['passed'], stage_type='support_pruning',
            input_material_cm3=volume(self.original)*1e6,
            retained_material_cm3=volume(self.body)*1e6,
            removed_material_cm3=volume(self.original-self.body)*1e6,
            initial_component_count=initial_components, component_count=len(parts),
            connected=len(parts)==1, force_required_isolated_components=max(0,len(parts)-1),
            watertight=bool(mesh.is_watertight), all_original_demands_passed=True,
            counts=self.proof['counts'], contact_triangles=self.proof['contact_triangles'],
            outside_saved_legal_component_cm3=volume(self.body-self.allowed)*1e6,
            added_material_cm3=volume(self.body-self.original)*1e6,
            attempts=self.attempts, candidate_budget=self.budget, history=self.history,
            discarded_numerical_boundaries=discarded,
            numerical_cleanup_rejected_to_preserve_original_demands=cleanup_rejected,
            certificate_calls=self.certificate.calls,
            force_classifications=self.certificate.classifications,
            reused_identical_reaction_cones=self.certificate.reused,
            force_solver_run=True, layout_changed=False, demand_resampling=False,
            global_minimum_claimed=False, seconds=time.monotonic()-self.began)
        return mesh, record, self.proof
