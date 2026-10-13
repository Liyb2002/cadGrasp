"""Enumerate re-seating jumps; rank whole demand loss and refine branches."""
import numpy as np
from co_common import transform_points
from whole_search.common import transform_mesh
from whole_search.reuse_first import registered


def structural_key(layout,trial,guest,extent):
    # Tiny direction updates must not reopen an already tried host choice.
    # A changed discrete host pattern opens a genuinely new search context.
    return (tuple(layout.hosts),guest,int(trial.hosts[guest]),
            np.round(trial.placements[guest,:3,:3],8).tobytes(),
            np.round(trial.placements[guest,:3,3]/extent,2).tobytes())


def juxtapose(objective, current, *, budget=8, refine=None, excluded=None, max_refined=2):
    model = objective.model; layout = current.layout
    excluded = excluded if excluded is not None else set()
    losses = (current.residual_loss if current.residual_loss is not None else 1-current.coverage)
    active = list(layout.active)
    guests = sorted(active, key=lambda k: -losses[active.index(k)])
    state = model.contact_delta.state(layout)
    priorities = {blocker: sum(float(losses[active.index(owner)]) * float(
        model.point_areas[state.locks[owner, blocker] & (state.coverage_counts[owner] > 0)].sum())
        for owner in active if owner != blocker) for blocker in active}
    blockers = sorted(active, key=lambda k: -priorities[k])
    order = list(dict.fromkeys(guests[:2]+blockers[:2]+guests[2:]))
    per_guest = []; seen = set()
    for guest in order:
        world = model.native[layout.hosts[guest], :3, :3] @ layout.directions[guest]
        hosts = sorted((k for k in active if k != guest), key=lambda k:
                       -float(world @ (model.native[layout.hosts[k], :3, :3] @ layout.directions[k])))
        proposals = []
        for host in hosts:
            trial = model.juxtapose(layout, guest, host)
            center_guest = transform_points(model.mesh.center_mass[None], trial.placements[guest])[0]
            center_host = transform_points(model.mesh.center_mass[None], layout.placements[host])[0]
            normal = model.floor_normal(trial, guest)
            alignment = center_host-center_guest; alignment -= normal*(alignment@normal)
            trial.placements[guest, :3, 3] += alignment
            key = trial.key()
            operation_key=structural_key(layout,trial,guest,model.extent)
            if key == layout.key() or key in seen or operation_key in excluded: continue
            if trial.hosts[guest]==layout.hosts[guest] and np.allclose(trial.placements[guest,:3,:3],layout.placements[guest,:3,:3],atol=1e-10,rtol=0) and np.linalg.norm(trial.placements[guest,:3,3]-layout.placements[guest,:3,3])<.01*model.extent:continue
            a = transform_mesh(model.mesh, trial.placements[guest]).bounds
            b = transform_mesh(model.mesh, layout.placements[host]).bounds
            intersection = np.maximum(0, np.minimum(a[1], b[1])-np.maximum(a[0], b[0]))
            overlap = float(np.prod(intersection)/min(np.prod(a[1]-a[0]), np.prod(b[1]-b[0])))
            if overlap < .03: continue
            seen.add(key)
            proposals.append((trial, dict(guest=model.poses[guest], host=model.poses[host],
                body_bbox_overlap_fraction=overlap, placement='world-horizontal centroid alignment')))
        if not registered(layout, guest):
            trial = layout.copy(); trial.hosts[guest] = guest; trial.placements[guest] = np.eye(4)
            trial.directions[guest] = layout.placements[guest, :3, :3].T @ layout.directions[guest]
            key = trial.key()
            if key not in seen and structural_key(layout,trial,guest,model.extent) not in excluded and key != layout.key():
                proposals.insert(0, (trial, dict(guest=model.poses[guest], host=model.poses[guest],
                    placement='restore native rotating reuse')))
                seen.add(key)
        per_guest.append(proposals)
    # Round-robin keeps the finite budget from being consumed by one guest.
    rows = [items[index] for index in range(max((len(x) for x in per_guest), default=0))
            for items in per_guest if index < len(items)][:budget]
    trials = []; evaluated = []
    for candidate, detail in rows:
        excluded.add(structural_key(layout,candidate,model.poses.index(detail['guest']),model.extent))
        result = objective.evaluate(candidate); evaluated.append((result, detail))
        trials.append(dict(**detail, coverage=result.coverage.tolist(),
            residual_loss=result.residual_loss.tolist(), estimated_material_cm3=result.volume_cm3))
    if not evaluated:
        return current, dict(operation='juxtapose', accepted=False, discrete_trials=[], all_poses_active=True)
    evaluated.sort(key=lambda row: (not objective.covered(row[0]), objective.coverage_loss(row[0]), row[0].volume_cm3))
    branches = []
    for candidate, detail in evaluated[:max_refined]:
        raw = candidate.layout
        jump_layout = dict(placements=raw.placements.tolist(), directions=raw.directions.tolist(),
                           hosts=raw.hosts.tolist(), active=list(raw.active))
        records = []
        if refine is not None: candidate, records = refine(candidate, [model.poses.index(detail['guest'])])
        branches.append((candidate, detail, jump_layout, records))
    candidate, detail, jump_layout, records = min(branches,
        key=lambda row: (not objective.covered(row[0]), objective.coverage_loss(row[0]), row[0].volume_cm3))
    accepted = objective.better(candidate, current)
    summary = [dict(selected=detail, loss=objective.coverage_loss(result),
                    coverage=result.coverage.tolist(), operations=records)
               for result, detail, _, records in branches]
    return (candidate if accepted else current), dict(operation='juxtapose', accepted=accepted,
        selected=detail, jump_layout=jump_layout, discrete_trials=trials,
        continuous_refinement=records, refined_branches=summary,
        random_design_sampling=False, all_poses_active=True)
