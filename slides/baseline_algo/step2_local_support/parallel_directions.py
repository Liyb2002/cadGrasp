"""Evaluate independent heads in isolated processes, retaining candidate order."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import trimesh

from step2_local_support import withdrawal as D


_analyzer = None


def initialize(vertices, faces, depth, catalogue):
    global _analyzer
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _analyzer = D.Analyzer(mesh, depth, catalogue)


def analyze(contact):
    row = _analyzer.analyze(contact)
    row['status'] = ('certified_directions_available' if row['has_certified_direction']
                     else 'no_certified_direction')
    return row


def rejected(index, candidate_id):
    return dict(candidate_index=index, candidate_id=candidate_id,
                status='rejected_step2_geometry', certified_directions=D.normalize(),
                has_certified_direction=False, representative=None,
                locked_direction_ids=[], unresolved_direction_ids=[])


def rows(mesh, depth, catalogue, contacts, workers=1):
    """Yield the same records as the serial analyzer, including rejected slots.

    Each contact is either a full valid contact or just its index and ID for an
    invalid candidate. Ordered publication makes every checkpoint a resumable
    prefix even when later workers finish first. Workers never write files.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Direction worker count must be a positive integer')
    contacts = list(contacts)
    valid = [c for c in contacts if 'triangles_m' in c]
    if not valid:
        for c in contacts:
            yield rejected(c['candidate_index'], c['candidate_id'])
        return
    if workers == 1:
        analyzer = D.Analyzer(mesh, depth, catalogue)
        for c in contacts:
            if 'triangles_m' not in c:
                yield rejected(c['candidate_index'], c['candidate_id'])
                continue
            row = analyzer.analyze(c)
            row['status'] = ('certified_directions_available' if row['has_certified_direction']
                             else 'no_certified_direction')
            yield row
        return
    pool = ProcessPoolExecutor(max_workers=min(workers, len(valid)),
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(mesh.vertices, mesh.faces, depth, catalogue))
    try:
        results = iter(pool.map(analyze, valid, chunksize=1))
        for c in contacts:
            yield (next(results) if 'triangles_m' in c else
                   rejected(c['candidate_index'], c['candidate_id']))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class Checkpoint:
    """Atomically publish prefixes every ten results or ten seconds.

    Call with force=True on completion/error. A hard process kill can lose the
    work since the last checkpoint, but cannot expose a partially written JSON.
    """
    def __init__(self, path, name, provenance, initial_count=0):
        self.path, self.name, self.provenance = Path(path), name, provenance
        self.last_count = initial_count
        self.last_time = time.monotonic()
        self.write_count = 0
        self.elapsed_seconds = 0.

    def save(self, rows, force=False):
        now = time.monotonic()
        if len(rows) == self.last_count:
            return
        if not force and len(rows)-self.last_count < 10 and now-self.last_time < 10.:
            return
        started = time.monotonic()
        payload = dict(object=self.name, complete=False, provenance=self.provenance, candidates=rows)
        temporary = self.path.with_name(self.path.name+'.tmp')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(self.path)
        self.last_count, self.last_time = len(rows), time.monotonic()
        self.write_count += 1
        self.elapsed_seconds += self.last_time-started
