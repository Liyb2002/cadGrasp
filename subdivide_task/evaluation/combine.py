"""Evaluation of step4/combine.py and step5/combined.py, by bounding boxes (minimum-volume oriented boxes).

The object's own bbx is the reference.  For every distinct support: the bbx of Step4's membrane, of it after
Step5's trim (its peel), and of Step5's support with its feet, their material, the seats and how many joins were Juxtaposition and how many Rotational
reuse.  Sorted by Step5's bbx (a support Step5 could not stand comes last, by Step4's bbx).
"""
import argparse
import collections
import json

from subdivide_task.step1.common import ROOT, write_json
from subdivide_task.step4.combine import FOLDER as COMBINE, obb, plain, seats
from subdivide_task.step4.seats import context
from subdivide_task.step5.combined import FOLDER as STEP5, distinct


def run(object_name='B', top=10):
    c = context(object_name); data = ROOT/'subdivide_task/data'/object_name; reference = obb(c.base.source.vertices)
    step5 = {r['step4_file']: r for r in json.loads((data/STEP5/'report.json').read_text())['results']} if (data/STEP5/'report.json').exists() else {}
    rows = []
    for g in distinct(object_name):
        ops = collections.Counter(s['operation'] for s in g['chain']); r5 = step5.get(g['file'], {})
        stood = r5.get('status') == 'passed'
        rows.append(dict(samples=g['samples'], name=g['name'], step4_file=g['file'], step4_files=g['files'], step5=r5.get('name'),
                         juxtaposition=ops['juxtaposition'], rotational_reuse=ops['rotational_reuse'],
                         seats=[len(ks) for _, ks in seats(g)],
                         chain=' -> '.join([g['chain'][0]['parents'][0]]+[f"{s['parents'][1]}({s['operation'][0].upper()})" for s in g['chain']]),
                         step4_bbx_cm3=g['bbx_m3']*1e6, step4_volume_cm3=g['volume_cm3'],
                         trimmed_bbx_cm3=r5.get('peeled_bbx_m3', r5.get('trimmed_bbx_m3'))*1e6 if stood else None,
                         step5_bbx_cm3=r5['support_bbx_m3']*1e6 if stood else None, step5_volume_cm3=r5.get('volume_cm3') if stood else None,
                         step5_status=r5.get('status', 'not run'), step5_reason=r5.get('reason')))
    rows.sort(key=lambda r: (r['step5_bbx_cm3'] is None, r['step5_bbx_cm3'] if r['step5_bbx_cm3'] is not None else r['step4_bbx_cm3']))
    report = dict(object=object_name, object_bbx_cm3=reference*1e6, distinct_supports=len(rows), results=rows)
    write_json(data/'evaluation_combine.json', plain(report))
    print(f'object bbx {reference*1e6:.0f} cm3; {len(rows)} distinct supports; figures output/{object_name}/step5/<figure>.png')
    print(f"{'#':>2} {'step5 bbx':>9} {'x obj':>6} {'step4 bbx':>9} {'x obj':>6} {'trimmed':>9} {'x obj':>6} {'step5 cm3':>9} {'step4 cm3':>9} {'J':>2} {'R':>2}  seats  figure  chain")
    for k, r in enumerate(rows[:top]):
        s5 = f"{r['step5_bbx_cm3']:9.0f} {r['step5_bbx_cm3']/reference/1e6:6.2f}" if r['step5_bbx_cm3'] is not None else f"{'-':>9} {'-':>6}"
        v5 = f"{r['step5_volume_cm3']:9.1f}" if r['step5_volume_cm3'] is not None else f"{'-':>9}"
        t5 = f"{r['trimmed_bbx_cm3']:9.0f} {r['trimmed_bbx_cm3']/reference/1e6:6.2f}" if r['trimmed_bbx_cm3'] is not None else f"{'-':>9} {'-':>6}"
        print(f"{k+1:2d} {s5} {r['step4_bbx_cm3']:9.0f} {r['step4_bbx_cm3']/reference/1e6:6.2f} {t5} {v5} {r['step4_volume_cm3']:9.1f} "
              f"{r['juxtaposition']:2d} {r['rotational_reuse']:2d}  {r['seats']}  {r['step5']}  {r['chain']}")
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--top', type=int, default=10)
    a = p.parse_args(); run(a.object, a.top)
