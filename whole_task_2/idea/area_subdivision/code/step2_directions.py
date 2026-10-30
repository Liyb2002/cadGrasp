"""Step2's figure (step2/directions_figure.py, the same sheet: Start / 1 Cover / every direction's cover / 2 Split /
3 Check) for a run, into idea/area_subdivision/<object's run>_directions.png, and one area's cover drawn large as the
sheet's 1 Cover picture draws W01 (its faces in its colour, Step1's needle and head at its seed with the gold cone of
tilts, Step1's work cube) into idea/area_subdivision/<run>_<area>_cover.png.  Everything from the run's saved data with
Step2's own drawing; nothing of the run is written (the sheet is rendered into the scratch folder and copied here).

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/step2_directions.py B+whole_shape --cover W07
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np

from whole_task_2.step1.common import ROOT, load_shape, object_of
from whole_task_2.step5.video.raster import GRAY
from whole_task_2.step2 import directions_figure as DF

HERE = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('STEP2_FIGURE_SCRATCH', '/tmp/claude-1000/-home-yli581-Desktop-cadGrasp/093a55e2-7a8c-4226-abf3-06a98066f6d2/scratchpad/step2_figure'))
LARGE = (1800, 1100)    # px: the large cover picture


def name_of(run):
    return run.removeprefix(object_of(run)+'+')


def sheet(run, folder='step2_seats'):
    """Step2's sheet for the run, copied here."""
    made = DF.render(run, output_root=SCRATCH, folder=folder); path = HERE/f'{name_of(run)}_directions.png'
    shutil.copyfile(made, path); print('SHEET', path, flush=True); return path


def cover(run, area, folder='step2_seats', size=LARGE):
    """One area's cover as the sheet's 1 Cover picture (DF.needle_picture), drawn at `size`."""
    work = ROOT/'whole_task_2/data'/run; plan = json.loads((work/'step1_reach/plan.json').read_text()); s = plan['setup']
    report = json.loads((work/folder/'report.json').read_text())
    with np.load(work/'step1_reach/plan.npz') as z: target = z['target'].astype(bool)
    with np.load(work/'step1_reach/screen.npz') as z: axes, reach = z['axes'], z['reach']
    with np.load(work/folder/'supports.npz') as z: saved = {k: z[k] for k in z.files}
    mesh = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl'); areas = plan['areas']; j = [a['id'] for a in areas].index(area)
    V = np.asarray([DF.unit(a['axis_object']) for a in areas]); cosb = np.cos(np.radians(s['max_tilt_deg']))
    mine = reach[axes@V[j] >= cosb-1e-12].any(0)
    # Its seed as the sheet takes it: Step2's block seed (its report), else the face of its cover facing its needle squarest.
    seed = next((c['seed'] for c in report.get('blocks') or [] if DF.unit(c['direction'])@V[j] >= 1-1e-6), None)
    if seed is None:
        root = np.flatnonzero(target & reach.any(0)); ok = root[mine[root]]; seed = int(ok[np.argmax(mesh.face_normals[ok]@-V[j])])
    colour = np.asarray(DF.AREA_COLORS[j % len(DF.AREA_COLORS)]); painted = np.tile(GRAY, (len(mesh.faces), 1))
    painted[target] = DF.PALE; painted[mine & target] = colour
    P = saved[area+'_pose']; w = DF.unit(s['working_direction_world']); tip = DF.placed(mesh.triangles_center[seed], P)
    cube = DF.cube_edges(tip, np.eye(3), .5*float(mesh.extents.max()))
    keep = DF.NEEDLE; DF.NEEDLE = size
    try: picture = DF.needle_picture(mesh, s, P, w, tip, painted, cube, DF.station_view(w))
    finally: DF.NEEDLE = keep
    path = HERE/f'{name_of(run)}_{area}_cover.png'; picture.save(path); print('COVER', path, flush=True); return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run'); p.add_argument('--cover', nargs='*', default=())
    a = p.parse_args(); sheet(a.run)
    for x in a.cover: cover(a.run, x)
