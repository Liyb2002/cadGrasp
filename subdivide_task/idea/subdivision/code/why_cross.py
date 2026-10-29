"""Why does a cross-group Rotational reuse fail?  A one-task group's seat (its Step4 start) + a task of another Step1
group: every task's paired-load residual on the seat's support, whole and opened by the new task's keep-out, and the
two gravity directions in the object frame.

  PYTHONPATH=. .venv/bin/python -B subdivide_task/idea/subdivision/code/why_cross.py big_08 B3 A4
"""
import json, sys, numpy as np
from subdivide_task.step4 import combine as C
from subdivide_task.step3 import independent_geometry as G
run, gname, tname = sys.argv[1:4]
d = json.loads((C.ROOT/'subdivide_task/data'/run/'step4_combine/search.json').read_text())
group = next(g for g in d['start'] if g['name'] == gname)
c = C.context(run); ids = [r['id'] for r in c.base.records]; i = ids.index(tname); j = group['slots'][0]['members'][0]
s = C.group_solids(run, group)['s']; X = C.seats(group)[0][0]
grav = lambda k: C.unit(C.pose(run, k)[:3, :3].T@[0, 0, -1.])
print('gravity in the object frame: %s %s, %s %s, angle %.1f deg' % (ids[j], np.round(grav(j), 2), tname, np.round(grav(i), 2),
      np.degrees(np.arccos(np.clip(grav(j)@grav(i), -1, 1)))))
print('needle in the object frame : %s %s, %s %s' % (ids[j], np.round(C.needle(run, j), 2), tname, np.round(C.needle(run, i), 2)))
both = group['slots']+[C.slot(i, X, C.default_exit(run, i))]
for label, solid in (('B-seat whole', s), ('opened by the new keep-out', C.largest(s-C.clipped(C.move(X, C.body_and_keepout(run, i)), C.bounds(s)))[0])):
    mesh = G.mesh(solid)
    loss, held, rows = C.residual(run, both, mesh, {}, C.thin_cells(c, both, mesh))
    print('%-28s %.1f cm3  ' % (label, mesh.volume*1e6), ' | '.join('%s loss %.3f held %d/%d' % (r['task'], r['loss'], r['passed'], r['demands']) for r in rows))
