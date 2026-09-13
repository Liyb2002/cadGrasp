"""Compare existing contacts with the candidate pool using no downward point forces.

This is a stronger restriction than nonnegative net vertical force per support.
The candidate-pool certificate checks contact capacity, not simultaneous placement,
per-support floor moments, insertion, or the scheduler's final contact count.
The current scheduler and its saved scores are not modified by this experiment.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
from step3_scheculer import verification as V
from step3_scheculer import contacts as I

C = load_stage('score', 'contribution')

from step1.cases import pose_name


def upward_supply(problem, contacts):
    supply = V.Supply(problem, contacts)
    # The sign test uses the original normals, without a tolerance allowing
    # slightly negative forces. Zero-height directions remain admissible.
    keep = supply.raw[:, 1] >= 0
    for key in ['points', 'normals', 'owners', 'raw', 'full']:
        setattr(supply, key, getattr(supply, key)[keep])
    supply.exact_cache.clear()
    assert (supply.normals[:, 1] >= 0).all()
    return supply


def run(name):
    problem = C.Problem(name)
    folder = C.OUTPUTS/name/pose_name()/'step3_scheculer'
    selected_path = folder/'final_contacts.npz'
    selected = I.read_contacts(selected_path)
    current = upward_supply(problem, selected)
    current_mask, current_info = C.J.classify(current.full, problem.targets)
    candidates = [problem.candidate(i) for i in np.flatnonzero(problem.data.valid)]
    pool = upward_supply(problem, candidates)
    mask, pool_info = C.J.classify(pool.full, problem.targets)
    lower, upper, box = V.outer_box(problem)
    witnesses = [pool.witness(target) for target in box]
    exact_passed = sum(witness is not None for witness in witnesses)
    assert exact_passed == 64, 'No continuous certificate: one or more exact bases failed'
    # witness() solves on exact rational points/normals, then checks every
    # coefficient sign and all six equilibrium equations exactly.
    used_rays = sorted({i for witness in witnesses for i in witness['indices']})
    used_patches = sorted({int(pool.owners[i]) for i in used_rays if pool.owners[i] >= 0})
    out = folder/'verification'
    arrays = out/'passive_capacity.npz'
    np.savez_compressed(arrays, pool_points_m=pool.points, pool_inward_normals=pool.normals,
                        pool_contact_owners=pool.owners, current_covered=current_mask,
                        pool_covered=mask, box_vertices=box)
    result = dict(object=name, experiment='no_downward_point_reactions',
        force_restriction='Each active point reaction has inward_normal_y >= 0; thus every support has nonnegative net vertical force.',
        current_selected_ids=[contact['candidate_id'] for contact in selected],
        current_selected_covered_count=int(current_mask.sum()),
        current_selected_covered_percent=100*float(current_mask.mean()),
        current_classification=current_info,
        sample_count=len(mask), candidate_count=len(candidates),
        pool_covered_count=int(mask.sum()), pool_classification=pool_info,
        continuous_contact_capacity=dict(status='verified',
            method='64_continuous_outer_box_vertices_with_exact_nonnegative_reactions',
            exact_vertices_passed=exact_passed, witnesses=witnesses,
            lower_wrench=lower.tolist(), upper_wrench=upper.tolist(),
            numerical_padding_conditioned=1e-9,
            scope='Current work-face mesh; full local 30-degree caps, even occluded directions; all magnitudes in [0,0.5mg].'),
        certificate_used_candidate_ids=[candidates[i]['candidate_id'] for i in used_patches],
        certificate_used_ray_count=len(used_rays),
        contact_count_minimized=False, physical_supports_verified=False,
        scope='Candidate-pool capacity only. Overlap, per-body floor moments, finite floor friction, connections and insertion are not certified.',
        provenance=dict(inputs=I.hashes(problem.inputs+[selected_path]),
                        code=dict(C.code_hashes(), **I.hashes([Path(__file__), Path(V.__file__)]))),
        arrays_sha256=C.sha256(arrays))
    I.save(out/'passive_capacity.json', result)
    print(name, 'current selected:', result['current_selected_covered_percent'], '%;',
          'pool:', int(mask.sum()), '/', len(mask), '; exact box:', exact_passed,
          '; certificate uses', len(used_patches), 'candidate regions (not minimized)', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    args = parser.parse_args()
    for name in args.objects or C.OBJECTS:
        if name not in C.OBJECTS:
            parser.error('objects must be A1-f, B or C5')
        run(name)
