"""Certify a dead append-only prefix using only an original sampled load."""
import numpy as np
from step3_scheculer import contacts as I
from step3_scheculer.pair_scoring import C
from step3_scheculer.run_pairs import save


def remaining_pool_certificate(search, entries, proposals, base, folder, step):
    # Every future active group is a subset of this optimistic union: adding
    # immutable heads can only shrink the common direction/component sets.
    optimistic = entries+[p for p in proposals if p is not None]
    for task, (problem, mask) in enumerate(zip(search.problems, base['masks'])):
        missing = np.flatnonzero(~mask)
        if not len(missing):
            continue
        active = [e for e in optimistic if task in e['active_tasks']]
        full = problem.supply([search.geometry.view(e, task)['contact'] for e in active])
        for index in sorted(set(missing[[0, len(missing)//2, -1]].tolist())):
            target = C.U.target(problem.targets[index], full.shape[1])
            if C.W.solve(full, target) is not None:
                continue
            proof = C.W.exact_separator(full, target)
            if proof is None:
                continue  # A numerical failure alone never prunes a chain.
            arrays = folder/f'round_{step:03d}_remaining_pool.npz'
            np.savez_compressed(arrays, full=full, target=target)
            report = dict(pose=problem.pose, sample_index=index, original_sample=True,
                selected_head_count=len(entries), optimistic_active_head_ids=[e['contact']['candidate_id'] for e in active],
                exact_separator=proof, arrays=arrays.name, arrays_sha256=I.sha256(arrays),
                scope='This frozen prefix and finite candidate pool only; mutual geometry of new heads relaxed',
                future_active_sets_are_subsets=True, heads_removed_or_changed=False,
                extra_loads_added=False)
            path = arrays.with_suffix('.json')
            save(path, report)
            return dict(report, report=path.name, report_sha256=I.sha256(path))
    return None
