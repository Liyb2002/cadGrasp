"""Value-first contact search: score single patches, then compare grow and add.

Every score uses the full paired demand table and a shared six-row solution.
The search keeps the best singleton as its root, then chooses the largest gain
per added contact area. It records all alternatives, not just the chosen path.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.spatial import ConvexHull

OPTIONS = {"primal_feasibility_tolerance": 1e-9,
           "dual_feasibility_tolerance": 1e-9}
N_GLOBAL = 768
N_REFINE = 32
REFINE_ROOTS = 12
RADIUS_LEVELS = np.array([1., 1.2, 1.4, 1.7, 2.])
MAX_STEPS = 12
MIN_AREA_FRACTION = .35


class Search:
    def __init__(self, S, region, fps):
        self.S, self.region, self.fps = S, region, fps
        self.target = S.targets*S.scale
        self.columns = np.c_[-S.skin.ns, np.cross(S.skin.cs-S.com, -S.skin.ns)]*S.scale
        self.floor = S.floor6*S.scale
        self.patches, self.reduced = {}, {}
        self.states = {}
        self.catalog, self.trace = [], []
        self.started = time.time()

    def footprint(self, seed, level=0):
        key = int(seed), int(level)
        if key not in self.patches:
            radius = self.S.radius*RADIUS_LEVELS[level]
            self.patches[key] = np.flatnonzero(self.region(self.S, int(seed), radius))
        return self.patches[key]

    def area(self, ids):
        return float(1e4*self.S.skin.area[ids].sum())

    def extreme_faces(self, seed, level):
        """Exact convex-hull reduction within each planar source triangle.

        On one source triangle the normal is fixed and the wrench is affine in
        the contact position. Every interior point is a convex combination of
        planar hull vertices. All retained generators are actual drawn faces.
        """
        key = int(seed), int(level)
        if key not in self.reduced:
            ids = self.footprint(seed, level)
            source = self.S.skin.src[ids]
            selected = []
            for face in np.unique(source):
                group = ids[source == face]
                if len(group) <= 3:
                    selected.extend(group)
                    continue
                points = self.S.skin.cs[group]
                centred = points-points.mean(axis=0)
                _, singular, vh = np.linalg.svd(centred, full_matrices=False)
                assert singular[-1] < 1e-10
                if singular[0] < 1e-12:
                    keep = np.array([0])
                elif singular[1] < 1e-12:
                    x = centred@vh[0]
                    keep = np.unique([x.argmin(), x.argmax()])
                else:
                    keep = ConvexHull(centred@vh[:2].T).vertices
                selected.extend(group[keep])
            self.reduced[key] = np.array(sorted(set(selected)), dtype=int)
        return self.reduced[key]

    def classify(self, A, known=None, inherited_duals=None):
        """Certify all samples with primal solutions or reusable separators."""
        T = self.target
        ok = np.zeros(len(T), bool) if known is None else known.copy()
        pending = ~ok
        duals = []
        if inherited_duals is not None and len(inherited_duals):
            valid = inherited_duals[(inherited_duals@A.T).max(axis=1) < 1e-10]
            if len(valid):
                pending[(T@valid.T > 1e-8).any(axis=1)] = False
                duals.extend(valid)
        if pending.any():
            b = T[pending]
            # One separating functional often rules out the complete remaining
            # set, especially when scoring single small regions.
            common = linprog(np.r_[np.zeros(6), -1],
                             A_ub=np.r_[np.c_[A, np.zeros(len(A))], np.c_[-b, np.ones(len(b))]],
                             b_ub=np.zeros(len(A)+len(b)), bounds=[(-1, 1)]*6+[(0, None)],
                             method="highs", options=OPTIONS)
            if not common.success:
                raise RuntimeError(common.message)
            y = common.x[:6]
            if common.x[-1] > 1e-8:
                assert (A@y).max() < 1e-8 and (b@y).min() > 1e-8
                duals.append(y)
                pending[:] = False
        matrix = A.T.copy(order="F")
        phase = np.c_[matrix, np.eye(6), -np.eye(6)]
        objective = np.r_[np.zeros(len(A)), np.ones(12)]
        def accept_basis(x):
            # A feasible reaction usually uses at most six columns. Reuse that
            # basis for the other demands, accepting only explicit nonnegative
            # solutions with checked residuals; all others still use NNLS/LP.
            active = np.flatnonzero(x > 1e-12)
            indices = np.flatnonzero(pending)
            if not len(active) or not len(indices):
                return
            basis = matrix[:, active]
            coefficients = T[indices]@np.linalg.pinv(basis).T
            coefficients = np.maximum(coefficients, 0)
            residual = coefficients@basis.T-T[indices]
            solved = np.max(np.abs(residual), axis=1) < 1e-9
            ok[indices[solved]] = True
            pending[indices[solved]] = False

        for i in np.flatnonzero(pending):
            if not pending[i]:
                continue
            try:
                x = nnls(matrix, T[i], maxiter=max(1000, 3*len(A)))[0]
            except RuntimeError:
                x = np.zeros(len(A))
            if np.max(np.abs(matrix@x-T[i])) < 1e-9:
                ok[i], pending[i] = True, False
                accept_basis(x)
                continue
            result = linprog(objective, A_eq=phase, b_eq=T[i], bounds=(0, None),
                             method="highs", options=OPTIONS)
            if not result.success:
                raise RuntimeError(result.message)
            if result.fun <= 1e-8:
                assert np.max(np.abs(matrix@result.x[:len(A)]-T[i])) < 2e-8
                ok[i], pending[i] = True, False
                accept_basis(result.x[:len(A)])
            else:
                y = result.eqlin.marginals
                assert (A@y).max() < 1e-8 and T[i]@y > 1e-8
                pending[T@y > 1e-8] = False
                duals.append(y)
        return ok, np.array(duals).reshape(-1, 6)

    def evaluate(self, state, parent=None):
        key = tuple(sorted(state))
        if key not in self.states:
            ids = np.unique(np.concatenate([self.extreme_faces(*patch) for patch in state]))
            A = np.vstack([self.columns[ids], self.floor[None]])
            known, dual = (None, None) if parent is None else parent
            self.states[key] = self.classify(A, known, dual)
        return self.states[key]

    def scan(self, seeds, phase):
        existing = {row["seed"] for row in self.catalog}
        for seed in seeds:
            seed = int(seed)
            if seed in existing:
                continue
            existing.add(seed)
            ids = self.footprint(seed)
            ok, _ = self.evaluate(((seed, 0),))
            area = self.area(ids)
            self.catalog.append(dict(seed=seed, stage=phase, area_cm2=area,
                                     eligible=area >= MIN_AREA_FRACTION*np.pi*(100*self.S.radius)**2,
                                     joint_count=int(ok.sum())))
        counts = [r["joint_count"] for r in self.catalog]
        print(self.S.name, phase, "singletons", len(counts), "positive", sum(c > 0 for c in counts),
              "best", max(counts), "/", len(self.target),
              "seconds", round(time.time()-self.started, 1), flush=True)

    def initial_scan(self):
        S = self.S
        extent = 1/S.scale[-1]
        # Position AND normal cover both sides of thin parts and changes in
        # direction at curved regions. There is no camera or old-solution filter.
        features = np.c_[S.skin.cs/extent, .08*S.skin.ns]
        start = int(np.argmax(np.linalg.norm(S.skin.cs-S.com, axis=1)))
        seeds, _ = self.fps(features, start, N_GLOBAL)
        self.scan(seeds, "whole_surface")
        ranked = sorted((r for r in self.catalog if r["eligible"]),
                        key=lambda r: (-r["joint_count"], r["area_cm2"], r["seed"]))
        roots = [r for r in ranked if r["joint_count"] > 0][:REFINE_ROOTS]
        if not roots:
            # Expand sampling density before deciding whether the chosen small
            # radius offers a useful singleton. Zero is never called high value.
            seeds, _ = self.fps(features, start, 2*N_GLOBAL)
            self.scan(seeds, "denser_surface")
            ranked = sorted((r for r in self.catalog if r["eligible"]),
                            key=lambda r: (-r["joint_count"], r["area_cm2"], r["seed"]))
            roots = [r for r in ranked if r["joint_count"] > 0][:REFINE_ROOTS]
        extra = []
        for row in roots:
            ids = np.array(S.skin.tree.query_ball_point(S.skin.cs[row["seed"]], 1.5*S.radius))
            local, _ = self.fps(features[ids], int(np.argmin(np.linalg.norm(
                S.skin.cs[ids]-S.skin.cs[row["seed"]], axis=1))), N_REFINE)
            extra.extend(ids[local])
        self.scan(extra, "refine_valuable_locations")
        best = max((r for r in self.catalog if r["eligible"]),
                   key=lambda r: (r["joint_count"], -r["area_cm2"], -r["seed"]))
        if not best["joint_count"]:
            raise RuntimeError("No positive-value singleton in this radius/menu; revise the candidate scale")
        return ((best["seed"], 0),)

    def compatible(self, state, seed):
        fresh = self.footprint(seed)
        for old_seed, level in state:
            if old_seed == seed or np.intersect1d(fresh, self.footprint(old_seed, level)).size:
                return False
            apart = np.linalg.norm(self.S.skin.cs[seed]-self.S.skin.cs[old_seed])
            if apart < self.S.radius*(1+RADIUS_LEVELS[level]) and \
                    self.S.skin.ns[seed]@self.S.skin.ns[old_seed] >= -.5:
                return False
        return True

    def alternatives(self, state):
        current = self.evaluate(state)
        count = int(current[0].sum())
        old_area = sum(self.area(self.footprint(*patch)) for patch in state)
        moves = []
        options = []
        for j, (seed, level) in enumerate(state):
            for new_level in range(level+1, len(RADIUS_LEVELS)):
                grown = self.footprint(seed, new_level)
                if any(np.intersect1d(grown, self.footprint(*other)).size
                       for k, other in enumerate(state) if k != j):
                    continue
                child = state[:j]+((seed, new_level),)+state[j+1:]
                options.append(("grow", j+1, child))
        # Singleton score orders evaluation; zero-valued singletons remain
        # eligible because their marginal value in a combination can be large.
        for row in sorted(self.catalog, key=lambda r: -r["joint_count"]):
            if row["eligible"] and self.compatible(state, row["seed"]):
                options.append(("add", len(state)+1, state+((row["seed"], 0),)))
        for move_index, (action, number, child) in enumerate(options, start=1):
            area = sum(self.area(self.footprint(*patch)) for patch in child)
            added = area-old_area
            if added <= 1e-12:
                continue
            ok, _ = self.evaluate(child, current)
            assert np.all(~current[0] | ok)
            gain = int(ok.sum())-count
            moves.append(dict(action=action, patch=number, state=[list(x) for x in child],
                              joint_count=int(ok.sum()), gain=gain,
                              added_area_cm2=added, total_area_cm2=area,
                              gain_per_cm2=gain/added))
            if move_index % 200 == 0:
                print(self.S.name, 'compared', move_index, '/', len(options),
                      'grow/add moves from', count, 'solved demands;',
                      round(time.time()-self.started, 1), 's', flush=True)
        return moves

    @staticmethod
    def move_key(move):
        return (move["gain_per_cm2"], move["gain"], -len(move["state"]),
                -move["total_area_cm2"], tuple(tuple(-v for v in x) for x in move["state"]))

    def run(self):
        state = self.initial_scan()
        self.trace.append(dict(action="start", patch=1, state=[list(x) for x in state],
                               joint_count=int(self.evaluate(state)[0].sum()),
                               gain=int(self.evaluate(state)[0].sum()),
                               total_area_cm2=self.area(self.footprint(*state[0])),
                               compared=[]))
        while not self.evaluate(state)[0].all():
            if len(self.trace) >= MAX_STEPS:
                raise RuntimeError("Search reached its step limit before full sampled coverage")
            moves = self.alternatives(state)
            productive = [m for m in moves if m["gain"] > 0]
            if not productive:
                raise RuntimeError("Both growth and new-patch moves stalled; further branching is required")
            chosen = max(productive, key=self.move_key)
            state = tuple(tuple(x) for x in chosen["state"])
            self.trace.append(dict(chosen, compared=moves))
            print(self.S.name, "STEP", len(self.trace), chosen, "alternatives", len(moves),
                  "seconds", round(time.time()-self.started, 1), flush=True)
        return self.trace

    def save_catalog(self, directory):
        directory = Path(directory)
        seeds = np.array([r["seed"] for r in self.catalog])
        verdicts = np.array([self.evaluate(((int(seed), 0),))[0] for seed in seeds])
        faces = [self.footprint(int(seed)) for seed in seeds]
        report = dict(object=self.S.name, n_samples=len(self.target),
                      source_demand_sha256=hashlib.sha256((directory/f'demand_{self.S.name}_tip1.npz').read_bytes()).hexdigest(),
                      geometry_sha256=self.S.geometry_sha256,
                      normal_angle_from_seed_max_deg=self.S.normal_angle,
                      base_radius_mm=1000*self.S.radius, radius_levels=RADIUS_LEVELS.tolist(),
                      minimum_candidate_area_cm2=MIN_AREA_FRACTION*np.pi*(100*self.S.radius)**2,
                      singleton_objective="maximum joint solved count; ties prefer less area",
                      move_objective="maximum newly solved paired demands / added contact area; then gain, then fewer patches",
                      candidates=self.catalog, steps=self.trace,
                      all_scores_use_full_sample=True,
                      solver_tolerances=OPTIONS,
                      reduction="planar convex-hull vertices within each original source triangle",
                      seconds=time.time()-self.started)
        (directory/f"area_{self.S.name}_search.json").write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(directory/f"area_{self.S.name}_candidates.npz", seeds=seeds,
                            ok_joint=verdicts, face_offsets=np.r_[0, np.cumsum([len(f) for f in faces])],
                            face_ids=np.concatenate(faces))
