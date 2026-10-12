"""Verify old Step4.1 against its preserved, pre-height-extension source."""
import json
from .common import ROOT, C
from artifact_store import check_sources, resolve_artifact


def check_saved_report(path):
    path = resolve_artifact(path)
    result = json.loads(path.read_text())
    assert result['complete']
    C.I.check_hashes(result['provenance']['inputs'])
    check_sources(result['provenance']['code'])
    for filename, expected in result.get('artifacts', {}).items():
        if C.I.sha256(path.parent/filename) != expected:
            raise RuntimeError(f'Changed initialization artifact: {filename}')
    return result
