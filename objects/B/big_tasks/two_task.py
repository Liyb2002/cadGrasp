"""2task: two small work areas on bunny B, far apart, as the big tasks are written (generate.py's format).

The two body faces farthest apart along the surface (geodesic, the ears and head left out: below PEAK of the
height), each grown to SHARE of the surface in order of geodesic distance, smoothed as generate.py smooths.

  PYTHONPATH=. .venv/bin/python -B objects/B/big_tasks/two_task.py
"""
import json
from pathlib import Path

import numpy as np

from whole_task_2.step1.common import load_shape, sha256
from generate import MESH, Figure, Surface, fill_holes, record, smooth

HERE = Path(__file__).resolve().parent
NAME = '2task'
SHARE = .05                     # area fraction of each piece
PEAK = .6                       # source faces below this share of the height (original +Y up): the body


def main():
    mesh = load_shape(MESH); surface = Surface(mesh); y = mesh.triangles_center[:, 1]
    body = (y-y.min())/np.ptp(y) < PEAK; distance = lambda f: np.asarray(surface.distance([f])).ravel()
    a = int(np.flatnonzero(body)[0])
    for _ in range(3):                                          # the farthest pair, by repeated farthest faces
        d = distance(a).copy(); d[~body] = -1; a = int(np.argmax(d))
    d = distance(a).copy(); d[~body] = -1; b = int(np.argmax(d))
    pieces = []
    for f in (a, b):
        d = distance(f); order = np.argsort(d, kind='stable')
        piece = np.zeros(len(d), bool); piece[order[:np.searchsorted(np.cumsum(surface.area[order]), SHARE*surface.total)+1]] = True
        pieces.append(piece)
    pieces = fill_holes(surface, smooth(surface, pieces))
    row = record(NAME, surface, pieces, None, sha256(MESH))
    (HERE/f'{NAME}.json').write_text(json.dumps(row)+'\n')
    Figure(mesh).save(np.any(pieces, axis=0), f'{NAME}   {100*row["area_fraction"]:.1f}%', HERE/f'{NAME}.png')
    print(NAME, f'{100*row["area_fraction"]:.1f}%', 'pieces', [round(100*p['area_fraction'], 1) for p in row['pieces']],
          'sources', a, b, '%.0f mm apart (straight)' % (1e3*np.linalg.norm(mesh.triangles_center[a]-mesh.triangles_center[b])), flush=True)


if __name__ == '__main__':
    main()
