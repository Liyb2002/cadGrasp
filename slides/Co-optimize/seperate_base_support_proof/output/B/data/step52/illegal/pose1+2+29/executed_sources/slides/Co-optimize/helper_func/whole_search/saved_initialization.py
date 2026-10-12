"""Verify old Step4.1 against its preserved, pre-height-extension source."""
import json
from .common import ROOT, C

ARCHIVE = ROOT/'slides/Co-optimize/data/code_history/before_xyz_translation_20261010'
LEGACY_MODEL = 'slides/Co-optimize/helper_func/whole_search/model.py'
RENDER_ARCHIVE = ROOT/'slides/Co-optimize/data/code_history/before_uniform_xyz_force_only_20261010'
LEGACY_RENDER = 'slides/Co-optimize/vis_func/whole_step4_render.py'


def check_saved_report(path):
    result = json.loads(path.read_text())
    assert result['complete']
    C.I.check_hashes(result['provenance']['inputs'])
    for relative, expected in result['provenance']['code'].items():
        current = ROOT/relative
        if C.I.sha256(current) == expected:
            continue
        historical = ((ARCHIVE if relative==LEGACY_MODEL else RENDER_ARCHIVE)/relative)
        if relative not in [LEGACY_MODEL,LEGACY_RENDER] or not historical.is_file() or C.I.sha256(historical) != expected:
            raise RuntimeError(f'Missing or stale initialization source: {relative}')
    for filename, expected in result.get('artifacts', {}).items():
        if C.I.sha256(path.parent/filename) != expected:
            raise RuntimeError(f'Changed initialization artifact: {filename}')
    return result
