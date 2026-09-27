"""Select the greatest joint contribution from this round's Step 3.1 table."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
from step3_scheculer import contacts as I
from step3_scheculer import paths as PTH

OUTPUT_NAME = 'step3.2_select_contact'


def rank_candidates(rows):
    return sorted((r['index'] for r in rows if r['status'] == 'scored_joint'),
                  key=lambda index: (-rows[index]['covered_count'], rows[index]['id']))


def top_k_distribution(rows, base_count, top_k=5, excluded_indices=()):
    """Only eligible candidates; weight new coverage, with uniform zero-gain fallback."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    indices = [i for i in rank_candidates(rows) if i not in excluded_indices][:top_k]
    gains = np.array([rows[i]['covered_count']-base_count for i in indices], dtype=float)
    if np.any(gains < 0):
        raise ValueError('Adding a contact cannot lose previously covered loads')
    probabilities = (gains/gains.sum() if gains.sum() else
                     np.full(len(indices), 1/len(indices)) if indices else np.empty(0))
    return indices, probabilities


def run(name, round_number=1, state_path=None, problem=None, rng=None, top_k=5, excluded_indices=(), proposal_seed=None):
    problem = problem or C.Problem(name)
    score = C.read(name, round_number)
    fixed, base, paths = problem.load_state(state_path)
    assert score['selected_indices'] == sorted(p['candidate_index'] for p in fixed)
    assert round_number == len(fixed)+1
    source = PTH.folder(name, C.OUTPUT_NAME, round_number)
    out = PTH.folder(name, OUTPUT_NAME, round_number)
    out.mkdir(parents=True, exist_ok=True)
    rows = score['contributions']
    order = rank_candidates(rows)
    sampling = None
    chosen = order[0] if order else None
    if rng is not None:
        indices, probabilities = top_k_distribution(rows, int(base.sum()), top_k, excluded_indices)
        # Record the actual uniform draw so audits do not depend on RNG implementations.
        draw = float(rng.random()) if indices else None
        if indices:
            chosen = indices[min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(indices)-1)]
        sampling = dict(top_k=top_k, indices=indices, probabilities=probabilities.tolist(),
                        uniform_draw=draw, base_covered_count=int(base.sum()),
                        gains=[rows[i]['covered_count']-int(base.sum()) for i in indices])
        if proposal_seed is not None:
            sampling.update(excluded_indices=list(excluded_indices), proposal_seed=proposal_seed)
        if not indices:
            chosen = None
    winner = dict(rows[chosen]) if chosen is not None else None
    artifacts = {}
    if winner is not None:
        selected = problem.candidate(winner['index'])
        I.save_contacts(out/'selected_contact.npz', [selected])
        I.save_contacts(out/'contacts_before_optimization.npz', fixed+[selected])
        masks = I.load_npz(source/'sample_coverage.npz')
        np.testing.assert_array_equal(base, masks['base'])
        np.savez_compressed(out/'sample_coverage.npz', base=base, selected=masks['covered'][winner['index']])
        artifacts = {f: C.sha256(out/f) for f in
                     ['selected_contact.npz', 'contacts_before_optimization.npz', 'sample_coverage.npz']}
    result = dict(object=name, round=round_number, complete=True,
        objective='sample_top_k_by_marginal_coverage' if rng is not None else 'maximize_joint_covered_sample_count', winner=winner,
        ranking=order, sample_count=score['sample_count'], fixed_indices=score['selected_indices'],
        tied_best_ids=[rows[i]['id'] for i in order if winner is not None and rows[i]['covered_count'] == winner['covered_count']],
        tie_break='candidate_id_ascending', sampling=sampling, current_contact_resized=False,
        provenance=dict(inputs=I.hashes(problem.inputs+paths+[source/'contributions.json', source/'sample_coverage.npz']),
                        code=dict(C.code_hashes(), **I.hashes([Path(__file__)]))), artifacts=artifacts)
    I.save(out/'selection.json', result)
    print(name, 'round', round_number, 'Step 3.2 selected', winner['id'] if winner else None,
          'covered', winner['covered_percent'] if winner else None, flush=True)
    return result


def read(name, round_number=1):
    return I.check_report(PTH.folder(name, OUTPUT_NAME, round_number)/'selection.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name, args.round, args.state)
