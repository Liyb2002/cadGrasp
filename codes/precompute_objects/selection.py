"""Deterministic finite combination enumeration for precomputation."""
import itertools
import random

def shuffled_combinations(available, n, seed):
    """Every unordered set appears at most once; exhaustion always terminates."""
    available = tuple(sorted(available, key=lambda p: int(p.split('_')[1])))
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(available):
        raise ValueError(f'n must be between 2 and {len(available)} for this multi-pose baseline')
    if len(set(available)) != len(available):
        raise ValueError('Available poses must be distinct')
    groups = list(itertools.combinations(available, n))
    random.Random(seed).shuffle(groups)
    return groups

def choose(available, n, seed, evaluate, progress=None):
    """Pure selection loop. Never advance downstream on failed floor demands."""
    groups = shuffled_combinations(available, n, seed)
    attempts = []
    for group in groups:
        verdict = evaluate(group)
        attempts.append(dict(poses=list(group), **verdict))
        if progress:
            progress(attempts[-1], len(attempts), len(groups))
        if verdict['passed']:
            return dict(passed=True, status='pose_set_selected', selected_poses=list(group),
                        attempted_count=len(attempts), total_combinations=len(groups), attempts=attempts)
    return dict(passed=False, status='no_floor_compatible_pose_set', selected_poses=None,
                attempted_count=len(attempts), total_combinations=len(groups), attempts=attempts)

def choose_many(available, n, seed, evaluate, count=2, progress=None):
    """Continue the same random permutation until count distinct sets pass."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Positive integer group count required')
    groups = shuffled_combinations(available, n, seed)
    attempts, selected = [], []
    for group in groups:
        verdict = evaluate(group)
        attempts.append(dict(poses=list(group), **verdict))
        if progress:
            progress(attempts[-1], len(attempts), len(groups))
        if verdict['passed']:
            selected.append(list(group))
            if len(selected) == count:
                break
    return dict(passed=len(selected) == count,
        status='pose_sets_selected' if len(selected) == count else 'insufficient_floor_compatible_pose_sets',
        requested_count=count, selected_groups=selected, attempted_count=len(attempts),
        total_combinations=len(groups), exhausted=len(attempts) == len(groups), attempts=attempts)
