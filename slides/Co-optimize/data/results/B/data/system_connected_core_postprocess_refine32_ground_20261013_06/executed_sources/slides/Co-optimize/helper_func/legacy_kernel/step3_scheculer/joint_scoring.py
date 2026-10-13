"""Immediate new coverage, weighted by each task's current uncovered fraction."""
import numpy as np


def coverage_summary(masks):
    if not masks or any(np.asarray(m).ndim != 1 or not len(m) for m in masks):
        raise ValueError('Nonempty one-dimensional task masks required')
    counts = [int(np.count_nonzero(m)) for m in masks]
    sizes = [len(m) for m in masks]
    fractions = [c/n for c, n in zip(counts, sizes)]
    return dict(covered_counts=counts, sample_counts=sizes, covered_fractions=fractions,
                mean_coverage=float(np.mean(fractions)),
                all_sampled_complete=all(c == n for c, n in zip(counts, sizes)))


def weighted_gain(before, after):
    before, after = np.asarray(before, float), np.asarray(after, float)
    if before.ndim != 1 or not len(before) or before.shape != after.shape:
        raise ValueError('Matching nonempty task coverage vectors required')
    if (not np.isfinite([before, after]).all() or
            np.any(before < 0) or np.any(after < 0) or np.any(before > 1) or np.any(after > 1)):
        raise ValueError('Coverage must be finite and between zero and one')
    gains = after-before
    if np.any(gains < -1e-12):
        raise ValueError('Adding a fixed head must not reduce task coverage')
    gains = np.maximum(gains, 0.)
    weights = 1-before
    values = weights*gains
    return dict(value=float(values.mean()), uncovered_weights=weights.tolist(),
                coverage_gains=gains.tolist(), value_by_pose=values.tolist())


def top5_distribution(rows, before, top_k=5):
    if top_k < 1:
        raise ValueError('Positive top_k required')
    legal = [(r, weighted_gain(before, r['covered_fractions'])['value'])
             for r in rows if r['eligible']]
    ranked = sorted(legal, key=lambda item: (-item[1], item[0]['id']))[:top_k]
    values = np.array([v for _, v in ranked])
    probabilities = (values/values.sum() if values.sum() else
                     np.full(len(ranked), 1/len(ranked)) if ranked else np.empty(0))
    return [r['index'] for r, _ in ranked], probabilities
