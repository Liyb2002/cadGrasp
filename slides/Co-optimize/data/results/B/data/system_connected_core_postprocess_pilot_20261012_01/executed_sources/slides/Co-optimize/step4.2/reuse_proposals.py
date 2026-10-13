"""Bounded mixed registered/explicit-seat layouts with material-port matching."""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from whole_search.common import tangent_frame, legal_direction, initialize_close_directions
from fixture_state import installed_state
from translation.layout_update import shift
from material_evaluator import membership, target_gap, align_world_xy, mixed_target_applicable


def structure(layout):
    from whole_search.reuse_first import registered
    return (tuple(k for k in layout.active if registered(layout, k)),
            tuple(int(layout.hosts[k]) for k in layout.active))


def port_matching_offsets(model, layout, guest, anchor, rotation=None, maximum=2):
    """Sparse translations matching original contact locations and normals.

    This is only proposal guidance; original all-pose availability and reaction
    cones judge the resulting layout. No contact is moved by the discretization.
    """
    cg = model.mesh.center_mass
    Q = layout.placements[guest]
    R = Q[:3, :3] if rotation is None else rotation
    ga = np.isin(model.sources, model.allowed[guest])
    ha = np.isin(model.sources, model.allowed[anchor])
    g = (model.points[ga]-cg)@R.T+cg
    h = model.points[ha]@layout.placements[anchor, :3, :3].T+layout.placements[anchor, :3, 3]
    if not len(g) or not len(h):
        return [np.zeros(3)]
    # Limit correlation work; this does not alter final physical contact ports.
    gi = np.linspace(0, len(g)-1, min(len(g), 128)).astype(int)
    hi = np.linspace(0, len(h)-1, min(len(h), 256)).astype(int)
    g = g[gi]; h = h[hi]
    gn = model.point_normals[ga][gi]@R.T
    hn = model.point_normals[ha][hi]@layout.placements[anchor, :3, :3].T
    _, ids = cKDTree(h).query(g, k=min(3, len(h)))
    ids = np.asarray(ids).reshape(len(g), -1)
    translations = (h[ids]-g[:, None]).reshape(-1, 3)
    weights = .1+np.maximum(0, (gn[:, None]*hn[ids]).sum(2)).ravel()
    pitch = model.extent/24
    bins = {}; groups = np.round(translations/pitch).astype(int)
    for i, key in enumerate(map(tuple, groups)):
        bins.setdefault(key, []).append(i)
    ranked = sorted(bins.values(), key=lambda z: -float(weights[z].sum()))
    peaks = []
    for ids in ranked:
        t = np.average(translations[ids], axis=0, weights=weights[ids])
        if not any(np.linalg.norm(t-old) < pitch for old in peaks):
            peaks.append(t)
        if len(peaks) == maximum:
            break
    return peaks or [np.zeros(3)]


def set_exit_directions(model, layout, mode):
    trial = layout.copy()
    normals = np.array([model.floor_normal(layout, k) for k in layout.active])
    if mode == 'close':
        ds, _ = initialize_close_directions(normals, fixed_starts=8, max_iterations=40)
        for k, d in zip(layout.active, ds):
            trial.directions[k] = legal_direction(d, model.floor_normal(trial, k))
    elif mode == 'up':
        for k in layout.active:
            trial.directions[k] = model.floor_normal(trial, k)
    return trial


def assign_seat(model, layout, guest, anchor, *, preserve_state=False, peak=0,
                offset_fraction=0., angle=0.):
    trial = layout.copy(); old = layout
    if preserve_state:
        host = int(old.hosts[guest]); R = old.placements[guest, :3, :3]
    else:
        host = int(old.hosts[anchor])
        R = model.native[host, :3, :3].T@model.native[guest, :3, :3]
    offsets = port_matching_offsets(model, old, guest, anchor, R)
    t = offsets[peak % len(offsets)]
    cg = model.mesh.center_mass
    trial.hosts[guest] = host
    trial.placements[guest] = np.eye(4)
    trial.placements[guest, :3, :3] = R
    trial.placements[guest, :3, 3] = cg-R@cg+t
    normal = model.floor_normal(trial, guest); plane = tangent_frame(normal)
    trial.placements[guest, :3, 3] += model.extent*offset_fraction*plane@np.array([np.cos(angle), np.sin(angle)])
    world = model.native[int(old.hosts[guest]), :3, :3]@old.directions[guest]
    trial.directions[guest] = legal_direction(model.native[host, :3, :3].T@world, normal)
    # Existing nonpenetration projection, including seat height.
    trial = shift(model, trial, guest, np.zeros(3))
    model.validate_placement(trial, guest)
    return trial


def initial_topologies(model, initial, rng, budget=96):
    """Requested mixed subsets, or unrestricted roles outside the target scope."""
    if budget<=0:return []
    n = len(initial.active)
    normals = np.array([model.floor_normal(initial, k) for k in initial.active])
    subsets = []
    prior=getattr(model,'registered_floor_prior',None)
    sizes = (range(max(2, n-4), min(4, n-2)+1)
             if mixed_target_applicable(initial) else range(1, n+1))
    for size in sizes:
        choices = list(combinations(initial.active, size))
        if not mixed_target_applicable(initial) and n > 8:
            count = max(2, budget//n)
            if len(choices) > count:
                indices = sorted(rng.choice(len(choices), size=count, replace=False))
                choices = [choices[index] for index in indices]
        for ids in choices:
            # Nearby floor normals are a cheap compatibility prior, not a test.
            similarity = np.linalg.norm(normals[list(ids)].mean(0))
            assessment=prior.assess(ids) if prior is not None else {}
            if assessment.get('status')=='certified_infeasible':continue
            height=sum(assessment.get('heights_m',[]))
            unknown=assessment.get('status','necessary_feasible')!='necessary_feasible'
            subsets.append(((unknown,float(height),-float(similarity)), ids))
    subsets.sort(key=lambda z:z[0])
    ordered = []; sizes = sorted(set(len(ids) for _, ids in subsets))
    queues = {size: [ids for _, ids in subsets if len(ids) == size] for size in sizes}
    while any(queues.values()):
        for size in sizes:
            if queues[size]:
                ordered.append(queues[size].pop(0))
    rows = []; seen = set()
    # Cover every allowed registered subset before adding a second seat family.
    for sweep in range(4):
        for position, ids in enumerate(ordered):
            q = initial.copy()
            assessment=prior.assess(ids) if prior is not None else {}
            if assessment.get('status')=='necessary_feasible':
                q.state_offsets[list(ids),2]=assessment['heights_m']
            guests = [k for k in q.active if k not in ids]
            for order, k in enumerate(guests):
                anchors = sorted(ids, key=lambda h: -float(np.trace(
                    model.native[h, :3, :3].T@model.native[k, :3, :3])))
                anchor = anchors[(sweep//2+order%2) % min(2, len(anchors))]
                preserve = sweep == 1 or (sweep == 3 and order % 2 == 0)
                q = assign_seat(model, q, k, anchor, preserve_state=preserve,
                    peak=sweep%2, offset_fraction=([1/3, 1/16, 2/3, 1/3][sweep]),
                    angle=(order+position%4)*np.pi/2)
            q = set_exit_directions(model, q, 'close' if sweep%2 == 0 else 'up')
            q = align_world_xy(model, q)
            if target_gap(q) or q.key() in seen:
                continue
            seen.add(q.key())
            rows.append(('mixed-initialize', q, dict(registered_indices=list(ids),
                seat_family=sweep, body_overlap_is_not_objective=True,
                material_port_correlation=True)))
            if len(rows) >= budget:
                return rows
    return rows


def bounded_seat_proposals(model, layout, rng, round_index, maximum=4):
    from whole_search.reuse_first import registered
    reuse = [k for k in layout.active if registered(layout, k)]
    guests = [k for k in layout.active if k not in reuse]
    critical,_=model.useful_coordinates({'layout':layout})
    guests.sort(key=critical.index);reuse.sort(key=critical.index)
    rows = []
    if guests:
        losses=model.proxy(layout)['residual_loss']
        worst=max(guests,key=lambda k:losses[list(layout.active).index(k)])
        A = (worst if round_index%3==0 else guests[0] if round_index%3==1
             else guests[int(rng.integers(len(guests)))])
        candidates = [k for k in layout.active if k != A]
        flags=model.contact_delta.state(layout).available
        candidates.sort(key=lambda h: (-float(model.point_areas[flags[h]].sum()),
            -float(np.trace(model.native[int(layout.hosts[h]), :3, :3].T@model.native[A, :3, :3]))))
        anchors = [candidates[0]]
        others = candidates[1:]
        if others:
            anchors.append(others[int(rng.integers(len(others)))])
        for h in anchors:
            for peak in range(2):
                q = assign_seat(model, layout, A, h,
                    preserve_state=(round_index%3 == 1), peak=peak,
                    offset_fraction=([1/32,1/8,1/3] if peak==0 else
                                     [1/8,1/3,2/3])[round_index%3],
                    angle=float(rng.integers(8))*np.pi/4)
                if not target_gap(q) and q.key() != layout.key():
                    rows.append(('material-juxtapose', q, dict(guest=A, anchor=h, peak=peak)))
    # Exchange the role of one registered pose and one explicit guest. This
    # explores different combinations without making a whole cohort jump.
    if guests and reuse and round_index%2:
        A = guests[int(rng.integers(len(guests)))]
        owner_losses=model.proxy(layout)['residual_loss']
        worst=max(reuse,key=lambda k:owner_losses[list(layout.active).index(k)])
        B = worst if round_index%4==1 else reuse[int(rng.integers(len(reuse)))]
        q = model.restore_registered(layout, A)
        q = assign_seat(model, q, B, A, peak=round_index%2, offset_fraction=1/16)
        if not target_gap(q):
            rows = rows[:maximum-1]+[('mixed-role-exchange', q, dict(restored=A, moved=B))]
    # Preserve one affine ground/seat branch even if its first force loss is
    # worse. It must reach the population rather than vanish in local descent.
    if rows and reuse and round_index%3!=2:
        from material_floor import project_floor_layout
        fixed=(reuse[round_index%len(reuse)],) if round_index%3==0 else ()
        q=project_floor_layout(model,layout,fixed_ground=fixed)
        if q is not None and q.key()!=layout.key():
            slot=min(1,len(rows)-1)
            if rows[slot][0]=='mixed-role-exchange':slot=0
            rows[slot]=('material-floor-projection',q,
                        dict(ground_hosts=list(fixed),original_pressure_demands=True))
    return rows[:maximum]


def population_move(model,layout,donors,rng,round_index):
    """One bounded DE proposal across several guest positions/directions.

    All ORIGINAL rotations and registered seats stay fixed. Donor centres
    live in the shared material frame even when donor groupings differ.
    The original all-pose force/geometry gates still judge this whole layout.
    """
    from whole_search.reuse_first import registered
    if len(donors)<3:return None
    a,b=rng.choice(len(donors),2,replace=False)
    first,second=donors[int(a)],donors[int(b)]
    trial=layout.copy();cg=model.mesh.center_mass
    amplitude=.6 if round_index<8 else .3
    guests=[k for k in layout.active if not registered(layout,k)]
    if not guests:return None
    forced=guests[int(rng.integers(len(guests)))]
    for k in guests:
        if k!=forced and rng.random()>.65:continue
        ca=first.placements[k,:3,:3]@cg+first.placements[k,:3,3]
        cb=second.placements[k,:3,:3]@cg+second.placements[k,:3,3]
        noise=model.extent*(1/16 if round_index<8 else 1/64)*rng.normal(size=3)
        trial=shift(model,trial,k,amplitude*(ca-cb)+noise)
    for k in layout.active:
        if rng.random()<.35:
            axis=rng.normal(size=3);axis/=max(np.linalg.norm(axis),1e-12)
            from continuous_support.fast_gradient import turn
            trial.directions[k]=turn(model,trial,k,np.radians(12.)*axis).directions[k]
    return trial if not target_gap(trial) and trial.key()!=layout.key() else None
