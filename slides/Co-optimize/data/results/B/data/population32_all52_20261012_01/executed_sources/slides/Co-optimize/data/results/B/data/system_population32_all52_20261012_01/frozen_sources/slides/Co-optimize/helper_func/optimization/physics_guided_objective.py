"""Physics-envelope optimization over unbounded contact cones.

Potential rays pay an acquisition cost; no artificial contact capacities or
area-scaled feasibility constraints are introduced. All seven existing
scaled coordinates, including no-uplift, remain in the balance equation.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


def acquisition_equilibrium(floor, contacts, target, costs, penalty=.05):
    floor, contacts = np.asarray(floor), np.asarray(contacts)
    lengths = np.linalg.norm(contacts, axis=1)
    if np.any(lengths <= 0) or np.any(costs < 0) or penalty <= 0:
        raise ValueError('positive ray lengths and nonnegative costs required')
    normalized = contacts / lengths[:,None]
    rays = np.vstack([floor, normalized])
    count = len(rays)
    objective = np.r_[np.zeros(len(floor)), penalty*np.asarray(costs)+1e-9, np.ones(14)]
    equality = np.c_[rays.T, np.eye(7), -np.eye(7)]
    lp = linprog(objective, A_eq=equality, b_eq=target, bounds=(0,None), method='highs',
        options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not lp.success:
        raise RuntimeError('relaxed equilibrium: '+lp.message)
    residual = rays.T @ lp.x[:count] - target
    balance_error = np.max(np.abs(equality @ lp.x - target))
    if balance_error > 1e-7*max(1.,np.linalg.norm(target)):
        raise RuntimeError('unresolved relaxed equilibrium balance')
    reactions = lp.x[len(floor):count]
    return dict(value=float(lp.fun), reactions=reactions, cost_gradient=penalty*reactions,
                residual=residual, balance_error=float(balance_error), dual=lp.eqlin.marginals)


def aggregate(values, gradients, temperature=.005):
    values = np.asarray(values)
    if temperature <= 0 or not len(values):
        raise ValueError('positive temperature and nonempty working set required')
    weights = np.exp(values/temperature-logsumexp(values/temperature))
    value = temperature*(logsumexp(values/temperature)-np.log(len(values)))
    return float(value), np.einsum('l,lni->ni',weights,np.asarray(gradients))
