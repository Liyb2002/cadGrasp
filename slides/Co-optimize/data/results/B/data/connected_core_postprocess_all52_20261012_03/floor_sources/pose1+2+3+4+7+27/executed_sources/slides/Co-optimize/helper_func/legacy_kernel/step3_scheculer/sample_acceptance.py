"""The user's fixed finite-load acceptance rule for the active baseline."""
import json
from pathlib import Path

SAMPLE_COUNT = 32768
SCHEMA = 'shared_object_attached_heads_two_pose_samples_v3'
FIXED_AREA_SCHEMA = 'shared_object_attached_heads_two_pose_fixed_area_v4'
COMPLETION_SCHEMA = 'shared_object_attached_heads_two_pose_terminal_expansion_v5'
RULE = 'All fixed 32768 Step1 loads pass in each pose; no continuous validation or additional loads.'


def read_report(folder):
    """Prefer native sample-only runs; relabelled historical runs keep provenance."""
    from step1.needs import sha256
    folder = Path(folder)
    source = folder/'schedule.json'
    report = json.loads(source.read_text())
    if report.get('schema') in (SCHEMA, FIXED_AREA_SCHEMA, COMPLETION_SCHEMA):
        return report
    path = folder/'sample_result.json'
    if not path.exists():
        raise ValueError('Historical result needs fixed-sample acceptance; run relabel_pairs.py')
    result = json.loads(path.read_text())
    if result['source_schedule_sha256'] != sha256(source):
        raise ValueError('Fixed-sample result is stale relative to its source search')
    return result
