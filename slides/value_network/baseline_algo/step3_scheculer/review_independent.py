"""Replay independent head geometry and publish a gallery without creating bodies."""
import argparse
import html
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.run_independent import root, SCHEMA
from step3_scheculer.pair_tasks import read_task
from step2_local_support import geometry as G, withdrawal as W, surface as S


def head_cells(task, contacts, metadata):
    cells = []
    for c in contacts:
        offsets = G.vertex_offsets(task.domain.mesh, metadata[c['candidate_id']]['normal_depth_m'])[0]
        for face in np.unique(c['source_faces']):
            polygon = c['triangles_m'][c['source_faces'] == face, 1]
            cells.append(SimpleNamespace(vertices=G.head_cell(task.domain.mesh, polygon, face, offsets)))
    return cells


def audit_pose(name, pose):
    folder = root(name)/pose
    path = folder/'step3_scheculer/schedule.json'
    report = I.check_report(path)
    if report['schema'] != SCHEMA or report['poses'] != [pose] or report['shared_heads'] or report['cross_pose_constraints']:
        raise ValueError('Not an independent single-pose result')
    task = read_task(name, pose, folder=folder/'step_1_needs')
    contacts = I.read_contacts(path.parent/f'final_contacts_{pose}.npz')
    ids = [c['candidate_id'] for c in contacts]
    if any(h['active_poses'] != [pose] for h in report['heads']) or ids != report['result']['selected_ids']:
        raise ValueError('Head ownership mismatch')
    candidate = folder/'step2_local_support'/f'candidates_{pose}.json'
    catalogue = json.loads(candidate.read_text())
    if catalogue['global_head_exclusions']:
        raise ValueError('Other pose surfaces were excluded')
    vectors = np.asarray(catalogue['direction_catalogue']['vectors'])
    geometry = report['result']['geometry']['per_pose'][0]
    directions = geometry.get('common_direction_ids', [])
    local_cells = head_cells(task, contacts, {h['id']:h for h in report['heads']})
    own_checks = []
    for c in contacts:
        overlap = np.intersect1d(c['source_faces'], task.domain.work_ids)
        height = float(c['triangles_m'][:, :, 2].min())
        area_fraction = float(c['triangle_areas_m2'].sum()/task.domain.mesh.area)
        if len(overlap) or height < S.FLOOR_CLEARANCE_M-1e-9 or abs(area_fraction/.01-1) > 1e-4+1e-12:
            raise ValueError('Own working surface, floor clearance or fixed area check failed')
        own_checks.append(dict(id=c['candidate_id'], overlapping_work_faces=overlap.tolist(),
                               min_contact_z_m=height, area_fraction=area_fraction))
    check = dict(clear=False, reason='no_selected_head_or_direction')
    if local_cells and directions:
        analyzer = W.Analyzer(task.domain.mesh, .01*task.domain.mesh.extents.max(), dict(vectors=vectors.tolist()))
        check = dict(direction_id=directions[0], direction=vectors[directions[0]].tolist(),
                     **analyzer.test(local_cells, vectors[directions[0]]))
    if report['result']['passed'] and not check['clear']:
        raise ValueError('Successful pose failed full active-head union withdrawal')
    chosen = path.parent/f'particle_{report["winner"]:03d}'
    saved = json.loads((chosen/'schedule.json').read_text())
    if saved != report['result']:
        raise ValueError('Selected particle mismatch')
    for filename, digest in saved['artifacts'].items():
        if I.sha256(chosen/filename) != digest:
            raise ValueError('Particle artifact mismatch')
    with np.load(chosen/'coverage.npz') as coverage:
        mask = coverage[pose]
    if mask.shape != (32768,) or int(mask.sum()) != saved['covered_counts'][0]:
        raise ValueError('Coverage does not match all original samples')
    result = dict(pose=pose, step3_passed=saved['passed'], heads=len(contacts),
        covered_count=int(mask.sum()), sample_count=len(mask),
        own_contact_checks=own_checks,
        own_head_union_withdrawal=check, original_sample_replay=saved['independent_sample_checks'],
        source_report_sha256=I.sha256(path), source_contacts_sha256=I.sha256(path.parent/f'final_contacts_{pose}.npz'),
        fixture_constructed=False, fixture_placement_solved=False)
    I.save(folder/'step3_scheculer/independent_review.json', result)
    return result


def publish(name):
    output = root(name)
    batch = json.loads((output/'batch.json').read_text())
    if not batch['complete']:
        raise ValueError('Finish every requested pose before publishing final review')
    rows = [audit_pose(name, pose) for pose in batch['poses'] if
            next(r for r in batch['results'] if r['pose'] == pose)['status'] != 'pipeline_error']
    # Existing diagrams show each active group in its own task coordinates.
    # A collage cannot be mistaken for a shared fixture registration.
    from PIL import Image, ImageDraw
    from step2_local_support.render import font
    cols, width, gap = 4, 370, 14
    height = 430
    sheet = Image.new('RGB', (cols*(width+gap)+gap,
        ((len(rows)+cols-1)//cols)*(height+gap)+100), '#f3f5f7')
    ink = ImageDraw.Draw(sheet)
    ink.text((gap,16), f'{name}: independent pose heads', font=font(32), fill='#25333e')
    ink.text((gap,58), 'Each card is a separate pose solve. No shared fixture or placement is shown.', font=font(18), fill='#65737e')
    cards = []
    for i, row in enumerate(rows):
        pose = row['pose']; source = output/pose/'heads.png'
        picture = Image.open(source).convert('RGB')
        pixels = np.asarray(picture)
        foreground = (pixels.min(axis=2) < 238)
        foreground[:int(picture.height*.18)] = False
        foreground[int(picture.height*.9):] = False
        ys, xs = np.where(foreground)
        if len(xs):
            picture = picture.crop((max(0,int(xs.min())-25), max(0,int(ys.min())-25),
                min(picture.width,int(xs.max())+26), min(picture.height,int(ys.max())+26)))
        picture.thumbnail((width,width))
        x, y = gap+(i%cols)*(width+gap), 96+(i//cols)*(height+gap)
        sheet.paste(picture,(x+(width-picture.width)//2,y+(width-picture.height)//2))
        label = f'{pose} | {row["heads"]} heads | {row["covered_count"]}/32768'
        ink.text((x+8,y+width+4), label, font=font(17), fill='#25333e')
        ink.text((x+8,y+width+29), 'Step3 PASS' if row['step3_passed'] else 'Step3 INCOMPLETE',
                 font=font(19), fill='#287754' if row['step3_passed'] else '#b5413d')
        cards.append(f'<article><h2>{html.escape(pose)}</h2><p>{row["heads"]} heads · '
            f'{row["covered_count"]:,}/32,768 · <b>{"PASS" if row["step3_passed"] else "INCOMPLETE"}</b></p>'
            f'<a href="{pose}/heads.png"><img loading="lazy" src="{pose}/heads.png" alt="{pose} heads"></a>'
            f'<a href="{pose}/step3_scheculer/schedule.json">Search record</a></article>')
    sheet.save(output/'heads_overview.png')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Independent pose heads</title>'
        '<style>body{font:16px system-ui;margin:28px;background:#f3f5f7;color:#25333e}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}'
        'article{background:white;padding:18px;border-radius:12px}img{width:100%}h2{margin:0}a{color:#2879b9}</style>'
        f'<h1>{name}：逐 pose 独立选头</h1><p>{sum(r["step3_passed"] for r in rows)}/{len(batch["poses"])} 个 pose 通过 Step3。'
        '各 pose 的头、受力和退出独立求解，不共享头。点击图片查看大图。</p>'
        '<p>这些是接触头结果；尚未构造或认证 Step4 的共同支撑。</p><main>'+''.join(cards)+'</main>')
    source = [output/'batch.json']+[output/r['pose']/'step3_scheculer/schedule.json' for r in rows]
    review = dict(complete=True, object=name, poses=batch['poses'], passed_count=sum(r['step3_passed'] for r in rows),
        results=rows, fixture_constructed=False,
        provenance=dict(inputs=I.hashes(source), code=I.hashes([Path(__file__),Path(G.__file__),Path(W.__file__)])),
        artifacts={p:I.sha256(output/p) for p in ('heads_overview.png','index.html')})
    I.save(output/'review.json', review)
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    args = parser.parse_args()
    report = publish(args.object)
    print('REVIEW COMPLETE', report['passed_count'], '/', len(report['poses']))
