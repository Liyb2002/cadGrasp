"""Result-file lifecycle only; no geometry, search or physical-model changes.

Final numerical artifacts live in data/results, final visualizations in output.
Immutable older reports may still name the original output paths.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
from urllib.parse import urlsplit

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
RESULTS = COOPT / 'data/results'
VISUALS = COOPT / 'output'
INITIALIZATION_SOURCES = COOPT / 'data/initialization_sources'
VISUAL_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.svg', '.gif', '.mp4', '.webm', '.html'}


def result_root(name):
    return RESULTS / name


def normalized(path):
    # Do not follow symlinks: report paths denote logical artifacts.
    return Path(os.path.abspath(path))


def resolve_artifact(path):
    path = normalized(path)
    # A stage directory exists in both stores; readers of old stage references
    # need its numerical side, rather than the remaining visual-only directory.
    if path.is_relative_to(VISUALS) and path.is_dir():
        numerical = RESULTS / path.relative_to(VISUALS)
        if numerical.is_dir():
            return numerical
    if path.exists():
        return path
    for source, target in [(VISUALS, RESULTS), (RESULTS, VISUALS)]:
        if path.is_relative_to(source):
            relocated = target / path.relative_to(source)
            if relocated.exists():
                return relocated
    return path


def digest(path):
    return hashlib.sha256(resolve_artifact(path).read_bytes()).hexdigest()


def check_sources(records):
    """Verify saved sources by their actual bytes, including frozen initialization."""
    for relative, expected in records.items():
        current = resolve_artifact(ROOT / relative)
        if current.is_file() and digest(current) == expected:
            continue
        relocated = COOPT / 'helper_func/legacy_kernel' / relative.removeprefix('slides/baseline_algo/')
        if relative.startswith('slides/baseline_algo/') and relocated.is_file() and digest(relocated) == expected:
            continue
        archived = INITIALIZATION_SOURCES / (expected + '.py')
        if not archived.is_file() or digest(archived) != expected:
            raise RuntimeError(f'Missing or stale saved source: {relative}')


def install_legacy_io(module):
    """Adapt the relocated legacy reader without editing the numerical kernel."""
    if getattr(module, '_result_io_installed', False):
        return

    def check_hashes(records):
        for relative, expected in records.items():
            path = resolve_artifact(ROOT / relative)
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f'Missing or stale dependency: {relative}')

    def check_report(path):
        path = resolve_artifact(path)
        result = json.loads(path.read_text())
        assert result['complete']
        check_hashes(result['provenance']['inputs'])
        check_sources(result['provenance']['code'])
        for filename, expected in result.get('artifacts', {}).items():
            if digest(path.parent / filename) != expected:
                raise RuntimeError(f'Changed artifact: {path.parent / filename}')
        return result

    module.sha256 = digest
    module.check_hashes = check_hashes
    module.check_report = check_report
    module._result_io_installed = True


def _visual_target(path):
    path = normalized(path)
    return VISUALS / path.relative_to(RESULTS) if path.is_relative_to(RESULTS) else path


def _rewrite_links(content, source, destination):
    def replace(match):
        value = match.group(3)
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
            return match.group(0)
        target = normalized(source.parent / parsed.path)
        # Old galleries still have relative links from their original output location.
        if target.is_relative_to(VISUALS) and target.suffix.lower() not in VISUAL_SUFFIXES:
            target = RESULTS / target.relative_to(VISUALS)
        elif target.suffix.lower() in VISUAL_SUFFIXES:
            target = _visual_target(target)
        link = os.path.relpath(target, destination.parent)
        if parsed.query:
            link += '?' + parsed.query
        if parsed.fragment:
            link += '#' + parsed.fragment
        return match.group(1) + match.group(2) + link + match.group(2)
    return re.sub(r'((?:href|src)\s*=\s*)([\"\'])(.*?)\2', replace, content, flags=re.I)


def publish_visuals(directory):
    """Move finished images/videos to output; HTML remains a visual reader."""
    directory = normalized(directory)
    if not directory.is_relative_to(RESULTS):
        return
    paths = [p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in VISUAL_SUFFIXES]
    for source in paths:
        destination = _visual_target(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == '.html':
            destination.write_text(_rewrite_links(source.read_text(), source, destination))
            source.unlink()
        else:
            source.replace(destination)


def publish_gallery(path):
    path = normalized(path)
    if path.is_relative_to(RESULTS) and path.exists():
        destination = _visual_target(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_rewrite_links(path.read_text(), path, destination))
        path.unlink()


def cleanup_stage(directory):
    """Called after process rendering: retain final layout, masks, demands, meshes."""
    directory = Path(directory)
    for name in ['process_states', 'mesh_states', 'sampled_states', 'exact_states', 'data/initial_exact_states', 'data/worker_logs']:
        shutil.rmtree(directory / name, ignore_errors=True)
    for name in ['process.json', 'trace.json', 'sampled_layout.npz', 'accepted_layout.npz',
                 'accepted_layout.json', 'search_report.json', 'oracle_sampling.json',
                 'initial_metrics.json', 'run.log']:
        (directory / name).unlink(missing_ok=True)
    for path in directory.rglob('*.log'):
        path.unlink()


def finish_batch(directory):
    """Retire execution copies/progress after their hashes have been checked."""
    directory = Path(directory)
    shutil.rmtree(directory / 'executed_sources', ignore_errors=True)
    (directory / 'progress.json').unlink(missing_ok=True)
