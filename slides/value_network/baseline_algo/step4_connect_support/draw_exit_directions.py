"""Draw saved object-relative exits without changing any search or body result."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step2_local_support import geometry as G
from step4_connect_support import build_coupled_saddle as S
from step4_connect_support.codesign_port import build_local_bodies as L, visual_details as V
from step4_connect_support.fixture_view import cells_for, pack

COLORS = ['#d99530', '#9360b4', '#418eaa', '#508b64', '#ba647d']
RED = '#d3423e'


def font(size):
    return ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', size)


def read_data(output):
    report_path = output/'data/report.json'
    report = I.check_report(report_path)
    cache_path = output/'data/codesign_cache/sweep_cache.json'
    cache = json.loads(cache_path.read_text())
    I.check_hashes(cache['inputs'])
    bases = np.asarray(report['placement']['bases'])
    offsets = np.asarray(report['placement']['offsets'])
    directions = np.asarray(cache['directions'])
    np.testing.assert_allclose(cache['rotations'], bases, atol=1e-14, rtol=0)
    np.testing.assert_allclose(cache['offsets'], offsets, atol=1e-14, rtol=0)
    source = output/'data/codesign_inputs_surface/data/independent_input/schedule.json'
    schedule = json.loads(source.read_text())
    tasks, parts, inputs = [], [], [report_path, cache_path, source]
    for k, relative in enumerate(schedule['source_reports']):
        path = I.ROOT/relative
        own = I.check_report(path)
        pose = own['poses'][0]
        assert pose == report['poses'][k]
        task = read_task(report['object'], pose, folder=path.parent.parent/'step_1_needs')
        tasks.append(task)
        contacts_path = path.parent/f'final_contacts_{pose}.npz'
        catalogue_path = path.parent.parent/'step2_local_support'/f'candidates_{pose}.json'
        catalogue = json.loads(catalogue_path.read_text())['direction_catalogue']['vectors']
        np.testing.assert_allclose(directions[k], catalogue[report['attempts'][-1]['direction_ids'][k]], atol=1e-14, rtol=0)
        inputs.extend([path, contacts_path, catalogue_path, *task.inputs])
        for contact in I.read_contacts(contacts_path):
            points = contact['triangles_m'].reshape(-1, 3)@bases[k]+offsets[k]
            parts.append(dict(id=contact['candidate_id'], owner=pose, color=COLORS[k],
                              mesh=trimesh.Trimesh(points, np.arange(len(points)).reshape(-1, 3), process=False)))
    obj = tasks[0].domain.mesh.copy()
    obj.vertices = obj.vertices@bases[0]+offsets[0]
    for k, task in enumerate(tasks):
        np.testing.assert_allclose(task.domain.mesh.vertices@bases[k]+offsets[k], obj.vertices, atol=1e-12, rtol=0)
    # Keep the fixture/heads fixed. Object motion has the opposite sign to the
    # saved fixture withdrawal, exactly as used in the constructor's sweep.
    exits = -np.einsum('ki,kij->kj', directions, bases)
    chosen = next((h for h in report['generated_support_roots'] if h['candidate_id']=='pose_5_C020'), None)
    if chosen is None:
        raise ValueError('This explanatory view expects the recorded pose5+7 / C020 example')
    k = report['poses'].index(chosen['owner_pose'])
    folder = (I.ROOT/schedule['source_reports'][k]).parent
    contact = next(c for c in I.read_contacts(folder/f'final_contacts_{tasks[k].pose}.npz') if c['candidate_id']==chosen['candidate_id'])
    vertex_offsets = G.vertex_offsets(tasks[k].domain.mesh, chosen['constructor_root_normal_depth_m'])[0]
    root = L.union([S.solid(G.hull_mesh(v@bases[k]+offsets[k])) for v in cells_for(contact, tasks[k].domain, vertex_offsets)])
    root_mesh = S.unpack(root)
    samples = []
    for pose, direction in zip(report['poses'], exits):
        values = []
        for mm in [0, 1, 2, 5, 10, 20, 30, 50, 60]:
            moved = obj.copy(); moved.vertices += direction*mm/1000
            fraction = float((S.solid(moved)^root).volume()/root.volume())
            assert -1e-8 <= fraction <= 1+1e-8
            values.append(dict(distance_mm=mm, root_overlap_fraction=float(np.clip(fraction, 0, 1))))
        samples.append(dict(pose=pose, samples=values))
    return dict(report=report, object=obj, parts=parts, bases=bases, offsets=offsets,
                fixture_directions=directions, exits=exits, root=root_mesh, collision_samples=samples,
                root_id=chosen['candidate_id'], root_depth_m=chosen['constructor_root_normal_depth_m'], inputs=inputs)


def overlay(picture, mesh, color, view, size, opacity=1.):
    layer, ids = V.render((mesh.triangles, np.tile(ImageColor.getrgb(color), (len(mesh.faces), 1)), [], []), view, size)
    mask = Image.fromarray(np.uint8(ids>=0)*int(255*opacity))
    picture.paste(layer, mask=mask)


def scene(data, k, size=930):
    common = k is None
    basis = np.eye(3) if common else data['bases'][k]
    offset = np.zeros(3) if common else data['offsets'][k]
    convert = lambda mesh: trimesh.Trimesh((mesh.vertices-offset)@basis.T, mesh.faces, process=False)
    obj = convert(data['object'])
    parts = [(p, convert(p['mesh'])) for p in data['parts']]
    root = convert(data['root'])
    exits = data['exits']@basis.T
    center = obj.bounds.mean(0)
    arrow_ids = list(range(len(exits))) if common else [k]
    tails = [center.copy() for _ in arrow_ids]
    tips = [center+exits[j]*.145 for j in arrow_ids]
    # X-ray surface marks keep the fixed C020 visible behind the moving object.
    camera = V.R.axes([-.9, -1., .65])
    points = np.vstack([obj.vertices, *tips])
    if not common:
        points = np.vstack([points, obj.vertices+exits[k]*.02])
    view = V.fit(points, camera, margin=1.24)
    picture = Image.new('RGB', (size, size), 'white')
    if not common:
        floor = V.floor_triangles(obj.vertices, .015)
        m = trimesh.Trimesh(floor.reshape(-1,3), np.arange(len(floor)*3).reshape(-1,3), process=False)
        overlay(picture, m, '#e8edf0', view, size, .65)
    overlay(picture, obj, '#b1b9bf', view, size, .65 if common else .23)
    if not common:
        moved = obj.copy(); moved.vertices += exits[k]*.02
        overlay(picture, moved, '#929ca3', view, size, .68)
    for p, mesh in parts:
        overlay(picture, mesh, RED if p['id']==data['root_id'] else p['color'], view, size)
    overlay(picture, root, RED, view, size)
    ink = ImageDraw.Draw(picture)
    focus, width, axes = view
    project = lambda v: V.R.project(v, focus, axes, width, size)[:2]
    for j, tail, tip in zip(arrow_ids, tails, tips):
        a, b = project(tail), project(tip)
        direction = (b-a)/np.linalg.norm(b-a)
        side = np.array([-direction[1],direction[0]])
        ink.line([tuple(a),tuple(b-direction*18)],fill=COLORS[j],width=9)
        ink.polygon([tuple(b),tuple(b-direction*36+side*17),tuple(b-direction*36-side*17)],fill=COLORS[j])
        position = np.clip(.8*b+.2*a+side*45, [95,30], [size-95,size-40])
        ink.text(tuple(position), data['report']['poses'][j].replace('pose_','Pose '), font=font(28), fill=COLORS[j], stroke_width=1, anchor='mm')
    point = project(root.vertices.mean(0))
    ink.ellipse(tuple(point-12)+tuple(point+12), outline=RED, width=3)
    label = np.array([35., size-65])
    ink.line([tuple(point), tuple(label+[145,14])], fill=RED, width=2)
    ink.text(tuple(label), 'C020 根部', font=font(25), fill=RED)
    return picture


def publish(output):
    data = read_data(output)
    assert data['report']['poses']==['pose_5','pose_7']
    # A genuine collision witness, separate from complete-fixture feasibility.
    assert data['collision_samples'][0]['samples'][2]['root_overlap_fraction'] < 1e-7
    assert data['collision_samples'][1]['samples'][2]['root_overlap_fraction'] > .999
    size = 930
    page = Image.new('RGB', (3*size, 1160), 'white')
    ink = ImageDraw.Draw(page)
    ink.text((30,18), 'pose5+7：固定接触头，灰色物体沿箭头退出', font=font(39), fill=V.R.INK)
    ink.text((30,72), '橙色 = Pose 5　紫色 = Pose 7　红色 = C020 处生成的根部　箭头表示物体运动，彩色接触面透视显示', font=font(25), fill='#52616b')
    titles = ['同一物体参照：比较两个方向', 'Pose 5 的实际摆放', 'Pose 7 的实际摆放']
    footers = ['两个箭头都是真实方向；灰色物体未移动', '浅灰为起点，深灰为前进 20 mm 后；C020 固定', '前进 2 mm 已撞入 C020 根部；图示前进 20 mm']
    for col, k in enumerate([None, 0, 1]):
        page.paste(scene(data, k, size), (col*size, 160))
        ink.text((col*size+28,124), titles[col], font=font(29), fill=V.R.INK)
        ink.text((col*size+28,1105), footers[col], font=font(21), fill=RED if k==1 else '#52616b')
    page.save(output/'exit_directions.png')
    packed = dict(object=pack(data['object']), parts=[dict(id=p['id'], owner=p['owner'],color=p['color'],**pack(p['mesh'])) for p in data['parts']],
        root=pack(data['root']), root_id=data['root_id'], colors=COLORS[:2],
        poses=[dict(name=pose, rotation=b.tolist(), offset=o.tolist(), object_exit=d.tolist()) for pose,b,o,d in zip(data['report']['poses'],data['bases'],data['offsets'],data['exits'])],
        collision_samples=data['collision_samples'])
    template = Path(__file__).with_name('exit_directions_viewer.html')
    html = template.read_text().replace('__THREE__',(I.ROOT/'slides/reuse/vendor/three.min.js').read_text()).replace('__DATA__',json.dumps(packed,separators=(',',':')))
    (output/'exit_directions.html').write_text(html)
    I.save(output/'data/exit_directions.json', dict(complete=True, diagnostic_only=True,
        poses=data['report']['poses'], head_model='zero_thickness_contact_surface',
        arrow_definition='Object moves; support/heads fixed. Object exit = negative saved fixture direction transformed to common frame.',
        saved_fixture_directions_world=data['fixture_directions'].tolist(), object_exit_directions_common=data['exits'].tolist(),
        uses_last_saved_attempt=True, static_ghost_distance_m=.02, displayed_motion_range_m=.06,
        root_id=data['root_id'], root_normal_depth_m=data['root_depth_m'], collision_samples=data['collision_samples'],
        scope='Instantaneous C020-root overlap only; no all-head or complete-fixture acceptance claim',
        provenance=dict(inputs=I.hashes(data['inputs']),code=I.hashes([Path(__file__),template,Path(V.__file__),Path(S.__file__)])),
        artifacts={f'../{n}':I.sha256(output/n) for n in ['exit_directions.png','exit_directions.html']}))
    print('Exit-direction PNG and interactive 3D view saved:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    publish(parser.parse_args().output.resolve())
