"""Cache shared 3-D locked directions and continuous withdrawal rays for every Step2 head."""
import argparse
import json
import os
from contextlib import closing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step2_local_support import withdrawal as D
from step2_local_support import circles as P
from step2_local_support import parallel_directions as PD

from step1.cases import pose_name


def path(name):
    return P.OUTPUTS/name/pose_name()/'step2_local_support/insertion_directions.json'


def candidate(data, report, index):
    a, b = data.offsets[index:index+2]
    return dict(candidate_id=report['patches'][index]['id'], candidate_index=int(index),
                center_m=data.centers_m[index], center_face=int(data.center_faces[index]),
                radius_m=float(data.radius_m[index]), triangles_m=data.triangles[a:b],
                source_faces=data.source_faces[a:b], triangle_areas_m2=data.triangle_areas[a:b])


def read(name):
    result = I.check_report(path(name))
    if result['provenance']['code'] != code_hashes():
        raise RuntimeError('Insertion catalogue code changed')
    return result


def code_hashes():
    return dict(D.code_hashes(), **I.hashes([Path(__file__), Path(PD.__file__)]))


def run(name, resume=True, workers=None):
    workers = int(os.environ.get('CADGRASP_DIRECTION_WORKERS', '1')) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Direction worker count must be a positive integer')
    started = time.monotonic()
    domain, data, source = P.read(name)
    out = path(name)
    inputs = [out.parent/'circles.json', out.parent/'circles.npz',
              P.OUTPUTS/name/pose_name()/'step_1_needs/needs.json']
    provenance = dict(inputs=I.hashes(inputs), code=code_hashes())
    rows = []
    progress_path = out.with_name('insertion_directions_progress.json')
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['provenance'] == provenance:
            rows = progress['candidates']
            if len(rows) > len(data.valid) or any(r['candidate_index'] != i for i,r in enumerate(rows)):
                raise RuntimeError('Direction checkpoint must be an ordered candidate prefix')
    resumed_count = len(rows)
    direction_catalogue = D.make_catalogue(domain.mesh, domain.work_ids,
        [candidate(data, source, i) for i in range(len(data.valid)) if data.valid[i]])
    pending = [candidate(data, source, i) if data.valid[i] else
               dict(candidate_index=i, candidate_id=source['patches'][i]['id'])
               for i in range(resumed_count, len(data.valid))]
    checkpoint = PD.Checkpoint(progress_path, name, provenance, resumed_count)
    preparation_seconds = time.monotonic()-started
    try:
        with closing(PD.rows(domain.mesh, source['normal_depth_m'], direction_catalogue, pending, workers)) as results:
            for row in results:
                assert row['candidate_index'] == len(rows)
                rows.append(row)
                checkpoint.save(rows)
                if len(rows) % 10 == 0 or len(rows) == len(data.valid):
                    print(name, 'Step 2 insertion', len(rows), '/', len(data.valid),
                          'with directions', sum(r['has_certified_direction'] for r in rows),
                          'seconds', round(time.monotonic()-started, 1), flush=True)
    finally:
        checkpoint.save(rows, force=True)
    result = dict(object=name, complete=True, motion=D.MOTION,
                  mode=D.MODE, direction_catalogue=direction_catalogue,
                  normal_depth_m=source['normal_depth_m'], candidates=rows,
                  candidate_count=len(rows), geometry_valid_count=int(data.valid.sum()),
                  candidates_with_certified_direction=sum(r['has_certified_direction'] for r in rows),
                  geometry='Initial Step 2 contact areas; optimized geometry is recomputed by the scheduler',
                  no_direction_meaning='No certified direction in this motion family; unresolved bands are not a proof of impossibility',
                  elapsed_seconds=time.monotonic()-started, provenance=provenance,
                  execution=dict(workers=workers, resumed_candidate_count=resumed_count,
                      preparation_seconds=preparation_seconds, checkpoint_writes=checkpoint.write_count,
                      checkpoint_seconds=checkpoint.elapsed_seconds,
                      checkpoint_policy='atomic ordered prefix every 10 candidates or 10 seconds; flush on completion/error'))
    I.save(out, result)
    progress_path.unlink(missing_ok=True)
    return result


def ensure(name, workers=None):
    try:
        result = read(name)
        print(name, 'Step 2 insertion: reused hash-validated direction cache', flush=True)
        return result
    except (OSError, RuntimeError, AssertionError):
        return run(name, workers=workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--force', action='store_true', help='Rebuild even if the completed cache is current')
    parser.add_argument('--workers', type=int, default=None, help='Independent head processes (default CADGRASP_DIRECTION_WORKERS or 1)')
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error('--workers must be positive')
    for name in args.objects or P.OBJECTS:
        if name not in P.OBJECTS:
            parser.error('objects must be A1-f, B or C5')
        run(name, resume=False, workers=args.workers) if args.force else ensure(name, workers=args.workers)
