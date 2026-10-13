"""Independent candidate scoring processes; the parent alone selects heads."""
import os
import numpy as np

_search = None


def initialize(name, poses, count):
    global _search
    from step3_scheculer.run_joint import JointSearch
    from step3_scheculer.joint_prepared import PreparedGeometry
    _search = JointSearch(name, poses, seed=0, particles=1, count=count)
    if not isinstance(_search.geometry, PreparedGeometry):
        raise RuntimeError('Parallel scoring requires a valid prepared geometry cache')


def score_batch(entries, base, indices):
    from step3_scheculer.run_sequential import numerical_recovery
    results = []
    with numerical_recovery(_search.out/'numerical_retries'/f'worker_{os.getpid()}', _search.recoveries):
        for index in indices:
            row, proposal = _search.candidate_row(entries, _search.geometry.initial[index], base, index)
            results.append((row, None if proposal is None else proposal['active_tasks']))
    return results, _search.recoveries
