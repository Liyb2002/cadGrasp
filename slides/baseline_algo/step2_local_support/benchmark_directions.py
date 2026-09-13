"""Time serial/process head checks without changing the baseline geometry/results."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose, pose_name
from step2_local_support import circles as C, insertion_directions as I, withdrawal as D
from step2_local_support import parallel_directions as P


def run(name, count, workers):
    domain, data, source = C.read(name)
    reference_path = I.path(name)
    reference = json.loads(reference_path.read_text())
    # Historical code hashes can differ; the geometry/input hashes must agree.
    I.I.check_hashes(reference['provenance']['inputs'])
    contacts = [I.candidate(data, source, int(i)) for i in np.flatnonzero(data.valid)]
    catalogue = D.make_catalogue(domain.mesh, domain.work_ids, contacts)
    assert catalogue == reference['direction_catalogue']
    count = min(count, len(contacts))
    selected = [contacts[i] for i in np.linspace(0,len(contacts)-1,count,dtype=int)] if count else []
    expected = [reference['candidates'][c['candidate_index']] for c in selected]
    record = dict(object=name, pose=pose_name(), candidate_count=count,
                  candidate_indices=[c['candidate_index'] for c in selected],
                  scope='Evenly spaced geometry-valid candidates; includes analyzer/process startup, excludes catalogue construction and checkpoint writes',
                  reference_sha256=C.sha256(reference_path),
                  geometry_inputs=reference['provenance']['inputs'], runs=[])
    for number in workers:
        started = time.monotonic()
        actual = list(P.rows(domain.mesh, source['normal_depth_m'], catalogue, selected, workers=number))
        elapsed = time.monotonic()-started
        assert actual == expected, 'Direction evidence differs from the reference'
        row = dict(workers=number, elapsed_seconds=elapsed, all_records_equal=True)
        record['runs'].append(row)
        print(json.dumps(row), flush=True)
    folder = reference_path.parent
    C.save(folder/'timing_directions.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=C.OBJECTS)
    parser.add_argument('--pose', default='pose_1')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--workers', type=int, nargs='+', default=[1,4])
    args = parser.parse_args()
    if args.count < 1 or any(w < 1 for w in args.workers):
        parser.error('count and worker counts must be positive')
    with selected_pose(args.pose):
        run(args.name, args.count, args.workers)
