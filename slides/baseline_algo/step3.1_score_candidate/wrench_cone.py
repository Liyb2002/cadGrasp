"""Polyhedral joint-wrench cones and independent nonnegative force witnesses."""
from decimal import Decimal, localcontext
import numpy as np
from scipy.linalg import null_space, qr
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError
from types import SimpleNamespace
from step3_scheculer import passive_support as U


def hull_of_section(points, conditioned=False):
    """Keep exact input rays when coplanar circular boundaries need a merge."""
    if conditioned:
        origin=points.mean(axis=0)
        _,singular,vt=np.linalg.svd(points-origin,full_matrices=False)
        transform=vt.T/singular
        hull=hull_of_section((points-origin)@transform)
        normals=hull.equations[:,:-1]@transform.T
        return SimpleNamespace(equations=np.c_[normals,hull.equations[:,-1]-normals@origin])
    try:
        return ConvexHull(points)
    except QhullError:
        # Q12 allows Qhull's coplanar-facet merge without perturbing the rays.
        # The resulting cone still has to contain every original generator
        # and agree with independent nonnegative equilibrium solves.
        return ConvexHull(points, qhull_options='Qx Q12')


def cone(full, conditioned=False):
    """Add extreme rays until every original ray lies in the retained cone."""
    dimension = full.shape[1]
    # A full-rank strictly positive dependence proves positive span of R^6.
    # Use a seven-row LP, avoiding a large facet enumeration for such regions.
    _, singular, vt = np.linalg.svd(full, full_matrices=False)
    transform = vt.T/singular
    white = full@transform
    matrix = np.vstack([np.c_[white.T, white.sum(axis=0)],
                        np.r_[np.ones(len(full)), len(full)]])
    interior = linprog(np.r_[np.zeros(len(full)), -1.], A_eq=matrix,
                       b_eq=np.r_[np.zeros(dimension), 1.], bounds=(0, None), method='highs')
    if interior.success and interior.x[-1] > 1e-10:
        coefficients = interior.x[:-1]+interior.x[-1]
        if np.max(np.abs(white.T@coefficients)) < 1e-10:
            return np.empty((0, dimension))
    # Condition the pointed cross-section too. Near-coplanar mesh contact
    # normals otherwise make Qhull merge distinct facets in the raw frame.
    rays=white
    positive = linprog(np.r_[np.zeros(dimension), -1.], A_ub=np.c_[-rays, np.ones(len(rays))],
                       b_ub=np.zeros(len(rays)), bounds=[(-1, 1)]*dimension+[(0, None)], method='highs')
    if positive.success and positive.x[-1] > 1e-8:
        axis = positive.x[:dimension]/np.linalg.norm(positive.x[:dimension])
        basis = null_space(axis[None])
        section = (rays/(rays@axis)[:, None])@basis
        keep=np.unique(np.r_[qr(rays.T,mode='economic',pivoting=True)[2][:dimension],
                             section.argmin(axis=0),section.argmax(axis=0)])
        for _ in range(80):
            try:
                hull = hull_of_section(section[keep],conditioned=conditioned)
            except QhullError:
                if conditioned:raise
                # Rescale the affine cross-section before retrying. Contact
                # positions/rays stay exact; no random perturbation is used.
                return cone(full,conditioned=True)
            H = hull.equations[:, :dimension-1]@basis.T+hull.equations[:, dimension-1, None]*axis
            H /= np.linalg.norm(H,axis=1)[:,None]
            scores=H@rays.T
            indices=scores.argmax(axis=1)
            add=np.setdiff1d(indices[scores[np.arange(len(H)),indices]>1e-10],keep)
            if not len(add):
                break
            keep=np.union1d(keep,add)
        else:
            raise RuntimeError('Extreme-ray enumeration did not finish')
        H = np.vstack([H, -axis])
    else:
        # Origin facets also describe cones containing lines. No pointedness
        # assumption is required by this bounded-hull construction.
        points = np.vstack([np.zeros(dimension), rays/np.linalg.norm(rays, axis=1)[:, None]])
        hull = hull_of_section(points)
        H = hull.equations[np.abs(hull.equations[:, -1]) < 1e-10, :dimension]
    if len(H):
        H = H@transform.T
        H /= np.linalg.norm(H, axis=1)[:, None]
        keep = np.unique(np.round(H, 10), axis=0, return_index=True)[1]
        H = H[np.sort(keep)]
        if (H@full.T).max()>=2e-9 and not conditioned:
            # Dividing rays by the cross-section axis can undo the original
            # conditioning. Rebuild in centered, rescaled section coordinates.
            # No perturbation, removed facet, or relaxed containment tolerance.
            return cone(full,conditioned=True)
        assert (H@full.T).max() < 2e-9
    return np.ascontiguousarray(H)


def precise_basis(full, target, initial):
    """Re-solve a six-column LP basis using 60-decimal arithmetic.

    Large opposing unlimited forces can lose digits in double-precision
    substitution. Decimal inputs are the exact supplied binary floats.
    """
    dimension = full.shape[1]
    ids = np.flatnonzero(initial > 0)
    assert len(ids) == dimension, len(ids)
    with localcontext() as ctx:
        ctx.prec = 60
        matrix = [[Decimal.from_float(float(full[j, i])) for j in ids]
                  + [Decimal.from_float(float(target[i]))] for i in range(dimension)]
        for col in range(dimension):
            pivot = max(range(col, dimension), key=lambda row:abs(matrix[row][col]))
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            divisor = matrix[col][col]
            assert divisor != 0
            matrix[col] = [v/divisor for v in matrix[col]]
            for row in range(dimension):
                if row != col:
                    factor = matrix[row][col]
                    matrix[row] = [a-factor*b for a, b in zip(matrix[row], matrix[col])]
        x = [matrix[i][-1] for i in range(dimension)]
        assert min(x) >= 0
        residual = max(abs(sum(Decimal.from_float(float(full[j, i]))*v for j, v in zip(ids, x))
                           -Decimal.from_float(float(target[i]))) for i in range(dimension))
        assert residual < Decimal('1e-45')
        return dict(indices=ids.tolist(), coefficients=[str(v) for v in x], residual=float(residual))


def exact_separator(full,target):
    """Resolve an uncertain primal verdict only with an exact separating plane."""
    for proposal_scale in [1.,1e3,1e6]:
        proof=_exact_separator_at_scale(full,target,proposal_scale)
        if proof is not None:
            return dict(proof,proposal_constraint_scale=proposal_scale)
    return None


def _exact_separator_at_scale(full,target,proposal_scale):
    """Positive rescaling changes only the numerical proposal, never its proof."""
    dimension = full.shape[1]
    from fractions import Fraction
    from scipy.linalg import qr
    dual=linprog(-target,A_ub=full*proposal_scale,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*dimension,
                 method='highs-ipm',options=dict(presolve=False,primal_feasibility_tolerance=1e-9,
                                                dual_feasibility_tolerance=1e-9))
    if not dual.success or target@dual.x<=1e-9:return None
    active=np.flatnonzero(np.abs(full@dual.x)<1e-12)
    rows=[];rhs=[]
    if len(active):
        _,_,order=qr(full[active].T,pivoting=True)
        for index in active[order]:
            trial=rows+[full[index]]
            if np.linalg.matrix_rank(np.asarray(trial),tol=1e-12)>len(rows):
                rows.append(full[index]);rhs.append(0.)
            if len(rows)==dimension-1:break
    for index in np.argsort(-np.abs(dual.x)):
        row=np.eye(dimension)[index]
        if np.linalg.matrix_rank(np.asarray(rows+[row]),tol=1e-12)>len(rows):
            rows.append(row);rhs.append(float(dual.x[index]))
        if len(rows)==dimension:break
    matrix=[[Fraction(float(x)) for x in row]+[Fraction(value)] for row,value in zip(rows,rhs)]
    for col in range(dimension):
        pivot=max(range(col,dimension),key=lambda i:abs(matrix[i][col]))
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        value=matrix[col][col]
        if value==0:return None
        matrix[col]=[x/value for x in matrix[col]]
        for i in range(dimension):
            if i!=col:
                factor=matrix[i][col]
                matrix[i]=[x-factor*y for x,y in zip(matrix[i],matrix[col])]
    h=[row[-1] for row in matrix]
    def product(row):return sum(Fraction(float(x))*y for x,y in zip(row,h))
    margin=product(target)
    if margin<=0 or any(product(row)>0 for row in full):return None
    return dict(normal_exact=[str(x) for x in h],target_margin_exact=str(margin),
                every_original_generator_checked_exactly=True)


def solve(full, target):
    """Independent LP on the original supply columns, with equation residuals."""
    dimension = full.shape[1]
    target = U.target(target, dimension)
    _, singular, vt = np.linalg.svd(full, full_matrices=False)
    rank = int((singular > singular[0]*1e-10).sum()) if len(singular) and singular[0] > 0 else 0
    basis = vt[:rank].T
    if np.max(np.abs(target-target@basis@basis.T)) > 2e-9:
        return None
    if not rank:
        return dict(indices=[],coefficients=[],residual=float(np.max(np.abs(target))),precise=None)
    transform = basis/singular[:rank]
    transformed = target@transform
    scale = max(float(np.max(np.abs(transformed))), 1.)
    options = {'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    result = linprog(np.zeros(len(full)), A_eq=(full@transform).T,
                     b_eq=transformed/scale, bounds=(0,None), method='highs', options=options)
    if result.status not in (0,2):
        result = linprog(np.zeros(len(full)), A_eq=(full@transform).T,
                         b_eq=transformed/scale, bounds=(0,None), method='highs-ipm',
                         options=dict(options,presolve=False,ipm_optimality_tolerance=1e-10))
    if result.status not in (0,2):
        result = linprog(np.zeros(len(full)), A_eq=(full@transform).T,
                         b_eq=transformed/scale, bounds=(0,None), method='highs-ds',
                         options=dict(options,presolve=False,simplex_dual_edge_weight_strategy='dantzig'))
    solution_scale=scale
    if result.status not in (0,2):
        # Whitening an almost rank-deficient supply can confuse presolve/IPM.
        # Retry the SAME six equations with positive column rescaling, then
        # verify the witness in the original coordinates below.
        column_scale=np.maximum(np.linalg.norm(full,axis=1),np.finfo(float).tiny)
        result=linprog(np.ones(len(full)),A_eq=(full/column_scale[:,None]).T,
                       b_eq=target,bounds=(0,None),method='highs-ds',options=dict(options,presolve=False))
        solution_scale=1/column_scale
    if result.status not in (0,2):
        if exact_separator(full,target) is not None:
            return None
        raise RuntimeError(result.message)
    if not result.success:
        return None
    x = np.maximum(result.x*solution_scale,0.)
    residual = float(np.max(np.abs(full.T@x-target)))
    ids = np.flatnonzero(x>0)
    precise = precise_basis(full,target,x) if rank == dimension and len(ids) == dimension and x.max()>1e5 else None
    if precise:
        residual = precise['residual']
    if residual > 2e-8:
        raise RuntimeError(f'Unresolved equilibrium residual {residual}')
    return dict(indices=ids.tolist(),coefficients=x[ids].tolist(),
                residual=residual,precise=precise)


def full_space_certificate(full):
    dimension = full.shape[1]
    witnesses=[]
    for target in np.vstack([np.eye(dimension),-np.eye(dimension)]):
        witness=solve(full,target)
        if witness is None:
            raise RuntimeError('Full-space cone claim failed an independent basis LP')
        witnesses.append(dict(target=target.tolist(),**witness))
    error=max(w['residual'] for w in witnesses)*np.sqrt(dimension)
    assert error<1e-7
    return dict(kind='positive_span_of_coordinate_cross_polytope',
                interior_margin_lower_bound=float(1/np.sqrt(dimension)-error),witnesses=witnesses)
