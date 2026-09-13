"""Cache shared 3-D locked directions and continuous withdrawal rays for every Step2 head."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step2_local_support import withdrawal as D
from step2_local_support import circles as P

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
    if result['provenance']['code'] != dict(D.code_hashes(), **I.hashes([Path(__file__)])):
        raise RuntimeError('Insertion catalogue code changed')
    return result


def run(name, resume=True):
    started = time.monotonic()
    domain, data, source = P.read(name)
    out = path(name)
    inputs = [out.parent/'circles.json', out.parent/'circles.npz',
              P.OUTPUTS/name/pose_name()/'step_1_needs/needs.json']
    provenance = dict(inputs=I.hashes(inputs), code=dict(D.code_hashes(), **I.hashes([Path(__file__)])))
    rows = []
    progress_path = out.with_name('insertion_directions_progress.json')
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['provenance'] == provenance:
            rows = progress['candidates']
    direction_catalogue = D.make_catalogue(domain.mesh, domain.work_ids,
        [candidate(data, source, i) for i in range(len(data.valid)) if data.valid[i]])
    analyzer = D.Analyzer(domain.mesh, source['normal_depth_m'], direction_catalogue)
    for index in range(len(rows), len(data.valid)):
        if data.valid[index]:
            row = analyzer.analyze(candidate(data, source, index))
            row['status'] = ('certified_directions_available' if row['has_certified_direction']
                             else 'no_certified_direction')
        else:
            row = dict(candidate_index=index, candidate_id=source['patches'][index]['id'],
                       status='rejected_step2_geometry', certified_directions=D.normalize(),
                       has_certified_direction=False, representative=None, locked_direction_ids=[], unresolved_direction_ids=[])
        rows.append(row)
        I.save(progress_path, dict(object=name, complete=False, provenance=provenance, candidates=rows))
        if (index+1) % 10 == 0 or index == len(data.valid)-1:
            print(name, 'Step 2 insertion', index+1, '/', len(data.valid),
                  'with directions', sum(r['has_certified_direction'] for r in rows),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    result = dict(object=name, complete=True, motion=D.MOTION,
                  mode=D.MODE, direction_catalogue=direction_catalogue,
                  normal_depth_m=source['normal_depth_m'], candidates=rows,
                  candidate_count=len(rows), geometry_valid_count=int(data.valid.sum()),
                  candidates_with_certified_direction=sum(r['has_certified_direction'] for r in rows),
                  geometry='Initial Step 2 contact areas; optimized geometry is recomputed by the scheduler',
                  no_direction_meaning='No certified direction in this motion family; unresolved bands are not a proof of impossibility',
                  elapsed_seconds=time.monotonic()-started, provenance=provenance)
    I.save(out, result)
    progress_path.unlink(missing_ok=True)
    return result


def ensure(name):
    try:
        return read(name)
    except (OSError, RuntimeError, AssertionError):
        return run(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--force', action='store_true', help='Rebuild even if the completed cache is current')
    args = parser.parse_args()
    for name in args.objects or P.OBJECTS:
        if name not in P.OBJECTS:
            parser.error('objects must be A1-f, B or C5')
        run(name, resume=False) if args.force else ensure(name)
