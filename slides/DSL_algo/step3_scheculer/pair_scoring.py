"""Independent reaction solves for one contact design evaluated in two tasks.

Task loads are alternatives, never simultaneous loads on a combined body.
Geometry correspondence belongs to the caller; this module only scores it.
"""
from dataclasses import dataclass
import numpy as np

from step3_scheculer import contacts as I, passive_support as U, floor_support as F
from step3_scheculer.stage_imports import load_stage

C = load_stage('score', 'contribution')
J = load_stage('score', 'joint_samples')


@dataclass
class TaskProblem:
    pose: str
    domain: object
    floor: np.ndarray
    targets: np.ndarray
    scale: np.ndarray

    def supply(self, contacts):
        groups = [U.floor(F.columns(self.floor, self.domain.com), self.scale)]
        for contact in contacts:
            if 'wrench_generators' in contact:
                np.testing.assert_array_equal(contact['wrench_com_m'], self.domain.com)
                groups.append(U.heads(contact['wrench_generators'], self.scale))
                continue
            points = contact['triangles_m'].reshape(-1, 3)
            normals = np.repeat(-self.domain.mesh.face_normals[contact['source_faces']], 3, axis=0)
            groups.append(U.heads(np.c_[normals, np.cross(points-self.domain.com, normals)], self.scale))
        return I.merge_columns(*groups)


def coverage_summary(masks):
    if len(masks) != 2 or any(np.asarray(m).ndim != 1 or not len(m) for m in masks):
        raise ValueError('Two nonempty one-dimensional task masks are required')
    counts = [int(np.count_nonzero(m)) for m in masks]
    sizes = [len(m) for m in masks]
    fractions = [c/n for c, n in zip(counts, sizes)]
    return dict(covered_counts=counts, sample_counts=sizes,
                covered_fractions=fractions, mean_coverage=float(np.mean(fractions)),
                both_sampled_complete=all(c == n for c, n in zip(counts, sizes)))


def score_tasks(problems, contacts_by_pose, known_masks=None):
    if len(problems) != 2 or len(contacts_by_pose) != 2:
        raise ValueError('Exactly two task problems and two transformed contact sets are required')
    masks, gravity, classifiers = [], [], []
    for k, (problem, contacts) in enumerate(zip(problems, contacts_by_pose)):
        full = problem.supply(contacts)
        check = C.gravity_check(full, problem.domain, problem.scale)
        gravity.append(check)
        # An unfinished design can cover useful loads without balancing gravity.
        # Score each task independently; the separate gravity check is diagnostic.
        mask, classifier = J.classify(full, problem.targets,
            known_covered=None if known_masks is None else known_masks[k])
        masks.append(mask)
        classifiers.append(classifier)
    summary = coverage_summary(masks)
    gravity_passed = all(g['passed'] for g in gravity)
    summary['load_samples_complete'] = summary['both_sampled_complete']
    return dict(**summary, masks=masks, gravity=gravity,
                gravity_passed=gravity_passed, classifiers=classifiers)


def top5_distribution(rows, base_fractions, top_k=5):
    """Equal task weighting; reuse baseline's positive-gain sampling convention."""
    if top_k < 1 or len(base_fractions) != 2:
        raise ValueError('Positive top_k and two baseline fractions required')
    legal = [r for r in rows if r['eligible']]
    ranked = sorted(legal, key=lambda r: (-float(np.mean(r['covered_fractions'])), r['id']))[:top_k]
    gains = np.array([np.mean(np.asarray(r['covered_fractions'])-base_fractions) for r in ranked])
    if np.any(gains < -1e-10):
        raise ValueError('Adding unchanged contacts must not reduce coverage')
    gains = np.maximum(gains, 0.)
    probabilities = (gains/gains.sum() if gains.sum() else
                     np.full(len(ranked), 1/len(ranked)) if ranked else np.empty(0))
    return [r['index'] for r in ranked], probabilities
