"""Local actual-area secants with a soft proxy and exact acceptance gates.

The proxy is only a proposal heuristic. It never supplies a coverage mask or a
physical certificate. The seventh (shared no-uplift) equation remains exact even
in the residual LP; only the six wrench equations have an L1 residual.
"""
import time
import numpy as np
from scipy.optimize import linprog
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import contacts as I
from step2_local_support import withdrawal as D

A = load_stage('optimize', 'adjust')
C = A.C
PROBE_PER_STRATUM = 8
PROXY_TRIALS = 10
EXACT_TRIALS = 4  # includes the original, not final independent verification


class ProposalLPError(RuntimeError):
    pass


def representative_loads(mask, random_count, per_stratum=PROBE_PER_STRATUM):
    """Fixed, deterministically stratified quadrature; retain every counterexample."""
    mask = np.asarray(mask, bool)
    indices, weights = [], []
    for value in (False, True):
        group = np.flatnonzero(mask[:random_count] == value)
        if len(group):
            chosen = group[np.linspace(0, len(group)-1, min(per_stratum, len(group)), dtype=int)]
            indices.extend(chosen.tolist())
            weights.extend([len(group)/(len(mask)*len(chosen))]*len(chosen))
    for index in range(random_count, len(mask)):
        indices.append(index)
        weights.append(1/len(mask))
    return np.asarray(indices, int), np.asarray(weights, float)


def residual_gap(full, target):
    """Dual of min ||A6*x-b||_1, x>=0, A7*x=0, with seven dual variables."""
    full = np.asarray(full, float)
    target = C.U.target(np.asarray(target, float), full.shape[1])
    assert full.shape[1] == 7 and target[-1] == 0
    norms = np.maximum(np.linalg.norm(full, axis=1), 1e-30)
    result = linprog(-target, A_ub=full/norms[:, None], b_ub=np.zeros(len(full)),
                     bounds=[(-1., 1.)]*6+[(None, None)], method='highs',
                     options=dict(primal_feasibility_tolerance=1e-9,
                                  dual_feasibility_tolerance=1e-9))
    if not result.success:
        raise ProposalLPError('Residual proposal LP unresolved: '+result.message)
    return max(0., float(-result.fun))


class Proxy:
    def __init__(self, problem, initial):
        self.problem = problem
        self.indices, self.weights = representative_loads(
            initial['mask'], problem.problem.random_sample_count)
        self.targets = problem.targets[self.indices]
        self.values = {}
        self.invalid = set()
        self.seconds = 0.
        self.lp_count = 0
        self.lp_failures = []
        gaps = self.gaps(initial['full'])
        positive = gaps[gaps > 1e-8]
        self.tau = max(float(np.median(positive)) if len(positive) else .02, 1e-3)
        self.values[float(problem.initial_radius)] = self.row(initial, gaps)

    def gaps(self, full):
        started = time.monotonic()
        values = []
        for index, target in zip(self.indices, self.targets):
            try:
                values.append(residual_gap(full, target))
            except ProposalLPError as error:
                # Zero reactions obey the hard no-uplift equality. Their L1
                # residual is a conservative proxy fallback, never a proof.
                values.append(float(np.abs(target).sum()))
                self.lp_failures.append(dict(load_index=int(index), reason=str(error)))
        values = np.asarray(values)
        self.lp_count += len(values)
        self.seconds += time.monotonic()-started
        return values

    def row(self, entry, gaps):
        contribution = float(self.weights@np.exp(-gaps/self.tau))
        return dict(radius_m=entry['row']['radius_m'], area_m2=entry['row']['area_m2'],
                    total_area_m2=entry['row']['total_area_m2'], contribution=contribution,
                    utility=contribution/entry['row']['total_area_m2'], gaps=gaps.tolist())

    def evaluate(self, radius):
        radius = float(radius)
        if radius in self.invalid:
            return None
        if radius not in self.values:
            entry = self.problem.geometry(radius)
            if not entry['row']['geometry_valid'] or entry['row']['area_m2'] <= self.problem.minimum_area:
                self.invalid.add(radius)
                return None
            full = I.merge_columns(self.problem.fixed_full, A.patch_columns(
                self.problem.domain, entry['patch'], self.problem.floor, self.problem.scale))
            if not C.gravity_check(full, self.problem.domain, self.problem.scale)['passed']:
                self.invalid.add(radius)
                return None
            self.values[radius] = self.row(entry, self.gaps(full))
        return self.values[radius]


def acceptable(before, candidate, count):
    """Exact sampled acceptance, independent of all surrogate predictions."""
    if before['row']['covered_count'] == count:
        return (candidate['row']['covered_count'] == count and
                candidate['row']['total_area_m2'] <= before['row']['total_area_m2']*(1+1e-10))
    return (candidate['row']['covered_count'] == count or
            candidate['row']['efficiency'] >= before['row']['efficiency']*(1-1e-12))


def result_key(entry, count):
    row = entry['row']
    if row['covered_count'] == count:
        return (1, -row['total_area_m2'])
    return (0, row['efficiency'])


def search(problem, budget):
    started = time.monotonic()
    initial_radius = float(problem.initial_radius)
    initial = problem.score(initial_radius)
    full_initial = bool(initial['mask'].all())
    proxy = Proxy(problem, initial)
    proposals, secants = [], []
    trials = 0
    # All differences use the same loads, weights and tau within this coordinate.
    for direction in ((-1,) if full_initial else (1, -1)):
        radius = initial_radius
        step = .20
        for _ in range(4):
            if trials >= PROXY_TRIALS:
                break
            trial = float(np.clip(radius*np.sqrt(1+direction*step),
                                  problem.minimum_radius, problem.cap))
            if abs(trial-radius) <= problem.tolerance:
                break
            trials += 1
            old = proxy.values[radius]
            row = proxy.evaluate(trial)
            if row is None:
                step *= .5
                continue
            if trial not in proposals:
                proposals.append(trial)
            delta = row['area_m2']-old['area_m2']
            slope = (row['contribution']-old['contribution'])/delta if abs(delta) > 1e-20 else 0.
            secants.append(dict(before_radius_m=radius, radius_m=trial,
                                area_delta_m2=delta, contribution_per_m2=slope,
                                current_mean_contribution_per_m2=old['utility']))
            improved = row['utility'] > old['utility']*(1+1e-4)
            if improved:
                radius = trial
                step = min(.65, step*1.7)
            elif direction > 0 and row['contribution'] <= old['contribution']+1e-6:
                # A broader trial can cross a mesh/coverage plateau; a zero
                # local secant is not evidence that expansion cannot help.
                step = min(3., step*2.5)
            else:
                step *= .5
    if not full_initial and trials < PROXY_TRIALS:
        trial = min(problem.cap, initial_radius*2.)
        if trial not in proxy.values:
            trials += 1
            row = proxy.evaluate(trial)
            if row is None and trials < PROXY_TRIALS:
                largest = max([initial_radius]+[r for r in proposals if r < trial])
                trial = (largest+trial)/2
                trials += 1
                row = proxy.evaluate(trial)
            if row is not None:
                proposals.append(trial)
    # Check a promising enlargement even when the average-area ratio favors
    # shrinkage: full coverage takes priority over the ratio objective.
    by_utility = sorted(proposals, key=lambda r: proxy.values[r]['utility'], reverse=True)
    if full_initial:
        # Preserve budget for a small useful shrink before aggressive proposals.
        by_utility.sort(key=lambda r: abs(r-initial_radius))
    growing = [r for r in proposals if r > initial_radius]
    preferred = []
    if growing:
        preferred.append(max(growing, key=lambda r: (proxy.values[r]['contribution'], -r)))
    preferred += by_utility
    if growing:
        preferred.insert(2, max(growing))
    pending = list(dict.fromkeys(preferred))
    best_radius, best = initial_radius, initial
    exact_limit = min(budget, EXACT_TRIALS)
    def evaluations():
        return sum('mask' in entry for entry in problem.cache.values())
    checked, rejected = [], []
    while pending and evaluations() < exact_limit:
        radius = pending.pop(0)
        if radius in checked:
            continue
        checked.append(radius)
        if not D.nonempty(problem.insertion_directions(radius)):
            rejected.append(dict(radius_m=radius, reason='no_certified_common_direction'))
            middle = (radius+initial_radius)/2
            if len(proxy.values) < PROXY_TRIALS+3 and proxy.evaluate(middle) is not None:
                pending.append(middle)
            continue
        entry = problem.score(radius)
        if acceptable(best, entry, len(problem.targets)) and result_key(entry, len(problem.targets)) > result_key(best, len(problem.targets)):
            best_radius, best = radius, entry
        else:
            rejected.append(dict(radius_m=radius, reason='exact_coverage_or_efficiency_gate'))
            if full_initial and radius < initial_radius:
                # The active load subset may miss a limiting load. Backtrack
                # toward the exact feasible incumbent; never accept its loss.
                middle = (radius+best_radius)/2
                if proxy.evaluate(middle) is not None:
                    pending.insert(0, middle)
    assert evaluations() <= budget
    return best_radius, dict(method='actual_area_secant_with_exact_acceptance',
        converged=False, relative_gap=None, evaluated_sizes=evaluations(), max_evaluations=budget,
        exact_trial_limit=exact_limit, full_coverage_found=bool(best['mask'].all()),
        proxy=dict(method='stratified_exp_negative_L1_equilibrium_residual',
                   no_uplift_equation_hard=True, indices=proxy.indices.tolist(),
                   weights=proxy.weights.tolist(), tau=proxy.tau,
                   trials=list(proxy.values.values()), secants=secants,
                   residual_lp_count=proxy.lp_count, elapsed_seconds=proxy.seconds,
                   residual_lp_failures=getattr(proxy, 'lp_failures', []),
                   scope='Local proposal heuristic only; neither global differentiability nor optimality claimed.'),
        exact_checked_radii_m=checked, rejected=rejected,
        elapsed_seconds=time.monotonic()-started)
