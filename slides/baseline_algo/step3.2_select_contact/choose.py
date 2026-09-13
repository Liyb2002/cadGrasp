"""Select the greatest joint contribution from this round's Step 3.1 table."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
from step3_scheculer import contacts as I

OUTPUT_NAME = 'step3.2_select_contact'


def rank_candidates(rows):
    return sorted((r['index'] for r in rows if r['status'] == 'scored_joint'),
                  key=lambda index: (-rows[index]['covered_count'], rows[index]['id']))


def run(name, round_number=1, state_path=None, problem=None):
    problem = problem or C.Problem(name)
    score = C.read(name, round_number)
    fixed, base, paths = problem.load_state(state_path)
    assert score['selected_indices'] == sorted(p['candidate_index'] for p in fixed)
    assert round_number == len(fixed)+1
    source = I.folder(name, C.OUTPUT_NAME, round_number)
    out = I.folder(name, OUTPUT_NAME, round_number)
    out.mkdir(parents=True, exist_ok=True)
    rows = score['contributions']
    order = rank_candidates(rows)
    winner = dict(rows[order[0]]) if order else None
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
        objective='maximize_joint_covered_sample_count', winner=winner,
        ranking=order, sample_count=score['sample_count'], fixed_indices=score['selected_indices'],
        tied_best_ids=[rows[i]['id'] for i in order if rows[i]['covered_count'] == winner['covered_count']],
        tie_break='candidate_id_ascending', current_contact_resized=False,
        provenance=dict(inputs=I.hashes(problem.inputs+paths+[source/'contributions.json', source/'sample_coverage.npz']),
                        code=dict(C.code_hashes(), **I.hashes([Path(__file__)]))), artifacts=artifacts)
    I.save(out/'selection.json', result)
    print(name, 'round', round_number, 'Step 3.2 selected', winner['id'] if winner else None,
          'covered', winner['covered_percent'] if winner else None, flush=True)
    return result


def read(name, round_number=1):
    return I.check_report(I.folder(name, OUTPUT_NAME, round_number)/'selection.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name, args.round, args.state)
