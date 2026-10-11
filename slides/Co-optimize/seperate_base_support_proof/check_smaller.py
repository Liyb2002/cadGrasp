"""Extend the existing read-only certificates to saved two/three-pose sets."""
from pathlib import Path
import json
import sys
import time

PROOF = Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import certify as K
C = K.C


if __name__ == '__main__':
    directory = PROOF/'smaller_pose_checks'/str(time.time_ns())
    directory.mkdir(parents=True,exist_ok=False)
    code = C.freeze(directory)
    source = PROOF/'summary.json'
    source_hash = C.digest(source)
    cases = [row['case'] for row in json.loads(source.read_text())['step5_results']
             if row['case']['pose_count']<4]
    records = []
    for case in cases:
        out = PROOF/'certificates'/case['id']/'report.json'
        records.append(json.loads(out.read_text()) if out.exists() else K.certify(case))
    assert C.digest(source)==source_hash
    for path,digest in code.items():
        assert C.digest(C.ROOT/path)==digest
        assert C.digest(directory/'executed_sources'/path)==digest
    C.save(directory/'report.json',dict(complete=True,source=C.relative(source),
        source_sha256=source_hash,source_unchanged=True,executed_sources=code,
        source_snapshots_match=True,cases=records,step4_rerun=False,
        force_solver_replayed=False))
    print('SMALLER COMPLETE',len(records),'sets',flush=True)
