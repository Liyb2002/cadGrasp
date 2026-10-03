"""Show every retained group/pose, including unsuccessful support withdrawals."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.exit_support_geometry import prepare
from step4_connect_support import exit_support_geometry as E, build_coupled_saddle as S
from step4_connect_support.video import mp4_writer
from step4_connect_support.fixture_view import pack
from step4_connect_support.codesign_port import visual_details as V

spec = importlib.util.spec_from_file_location('exit_cpu', I.ROOT/'slides/reuse/code/render_cpu.py')
CPU = importlib.util.module_from_spec(spec); spec.loader.exec_module(CPU)
BLUE, GREY, RED = '#8fc9e8', '#aeb8bf', '#e34236'
WIDTH, HEIGHT, PANEL, FPS = 1440, 880, 720, 12
DISTANCES = np.r_[0., np.arange(.5, 20.01, .5), np.linspace(32, 500, 40)]


def font(size):
    return ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', size)


def empty_mesh():
    return trimesh.Trimesh(np.empty((0, 3)), np.empty((0, 3), int), process=False)


def piece(mesh, color, bias=0):
    return CPU.piece(dict(v=mesh.vertices.ravel(), f=mesh.faces.ravel()), color, bias=bias)


def collision_data(case, k):
    folder = case.output/'data/exit_replay'
    source = folder/f'collision_{case.poses[k]}.json'
    sweep_path = case.output/f'data/codesign_cache/sweep{k}.npz'
    inputs = I.hashes([folder/'support.json', sweep_path])
    code = I.hashes([Path(__file__), Path(S.__file__)])
    array_path = source.with_suffix('.npz')
    if source.exists():
        saved = json.loads(source.read_text())
        if saved['provenance'] == dict(inputs=inputs, code=code):
            I.check_report(source)
            with np.load(array_path) as z:
                meshes = [trimesh.Trimesh(z['vertices'][a:b], z['faces'][c:d], process=False)
                    for (a,b),(c,d) in zip(zip(z['vo'][:-1],z['vo'][1:]), zip(z['fo'][:-1],z['fo'][1:]))]
            return saved, meshes
    body, obj = S.solid(case.support), S.solid(case.object)
    meshes, volumes = [], []
    for distance in DISTANCES:
        moved = obj.translate((case.exits[k]*distance/1000/S.SCALE).tolist())
        intersection = body^moved
        volume = max(0., float(intersection.volume())*S.SCALE**3)
        volumes.append(volume)
        meshes.append(S.unpack(intersection) if volume > 8e-14 else empty_mesh())
    hits = np.flatnonzero(np.asarray(volumes) > 8e-14)
    # Prefer an early, visible witness. First sampled overlap is recorded too.
    candidates = [i for i in hits if 2 <= DISTANCES[i] <= 5]
    witness = candidates[0] if candidates else int(hits[0]) if len(hits) else 0
    moved = obj.translate((case.exits[k]*DISTANCES[witness]/1000/S.SCALE).tolist())
    root_hits = [p['id'] for p in case.patches if float((p['root']^moved).volume())*S.SCALE**3 > 8e-14]
    with np.load(sweep_path) as z:
        sweep = S.solid(trimesh.Trimesh(z['v'], z['f'], process=False))
    continuous = abs(float((body^sweep).volume()))*S.SCALE**3
    assert continuous+1e-11 >= max(volumes)
    np.savez_compressed(array_path,
        vertices=np.vstack([m.vertices for m in meshes]), faces=np.vstack([m.faces for m in meshes]),
        vo=np.cumsum([0]+[len(m.vertices) for m in meshes]), fo=np.cumsum([0]+[len(m.faces) for m in meshes]))
    saved = dict(complete=True, diagnostic_only=True, pose=case.poses[k],
        distances_mm=DISTANCES.tolist(), overlap_volumes_m3=volumes,
        first_sampled_collision_mm=float(DISTANCES[hits[0]]) if len(hits) else None,
        witness_index=int(witness), witness_mm=float(DISTANCES[witness]), root_ids_at_witness=root_hits,
        continuous_sweep_overlap_m3=continuous, continuous_sweep_length_m=S.SWEEP_LENGTH,
        continuous_sweep_collision=bool(continuous > 8e-14), volume_tolerance_m3=8e-14,
        fixture_direction_world=case.directions[k].tolist(), object_direction_world=(-case.directions[k]).tolist(),
        object_direction_common=case.exits[k].tolist(),
        original_step3_covered=case.report['covered_counts'][k], original_fixture_passed=case.report['passed'],
        provenance=dict(inputs=inputs, code=code), artifacts={array_path.name:I.sha256(array_path)})
    I.save(source, saved)
    return saved, meshes


def project(points, camera):
    focus, basis, span = camera
    p = (np.asarray(points)-focus)@basis.T
    return np.stack([PANEL/2+p[...,0]*PANEL/span, PANEL/2-p[...,1]*PANEL/span], axis=-1)


def arrow(image, start, end, camera, label=True):
    ink = ImageDraw.Draw(image)
    a, b = project(np.array([start,end]), camera)
    V.arrow(ink, a, b, color='#9059b5', width=5)
    if label:
        midpoint = (a+b)/2
        ink.text((midpoint[0], midpoint[1]-28), '物体退出方向', font=font(21), fill='#79479c', anchor='mm')


def render_pose(case, k, renderer):
    record, collisions = collision_data(case, k)
    basis, offset = case.bases[k], case.offsets[k]
    rotation, translation = basis, -basis@offset
    world = lambda vertices:(vertices-offset)@basis.T
    obj, support = piece(case.object, GREY), piece(case.support, BLUE)
    world_object, world_support = world(case.object.vertices), world(case.support.vertices)
    direction = -case.directions[k]
    np.testing.assert_allclose(basis@case.exits[k], direction, atol=1e-12, rtol=0)
    center = world_object.mean(0)
    # The camera sees motion sideways; two fixed cameras are used throughout.
    side = np.array([-direction[1], direction[0], .55])
    points = np.vstack([world_object, world_object+direction*.5, world_support])
    full_camera = CPU.fit(points, side, 1., padding=1.12)
    witness = record['witness_index']
    if len(collisions[witness].faces):
        collision_points = world(collisions[witness].vertices)
        focus = collision_points.mean(0)
        close_span = max(.085, float(np.ptp(collision_points@CPU.axes(side).T, axis=0)[:2].max())*1.7)
    else:
        focus, close_span = center, float(np.ptp(world_object, axis=0).max())*1.3
    close_camera = (focus, CPU.axes(side), close_span)
    floor = CPU.box(np.r_[np.ptp(points[:,:2],axis=0)+.05,.001],
                    np.r_[(points[:,:2].min(0)+points[:,:2].max(0))/2,-.0006], '#f0f3f5')
    floor['unlit'] = True
    fixed = [CPU.placed(floor), CPU.placed(support, rotation, translation)]
    fixed_close = renderer.render(fixed, close_camera, PANEL, PANEL)
    # Pre-render actual intersecting material as an explicitly labeled X-ray.
    red_pieces = [piece(m, RED, .00001) if len(m.faces) else None for m in collisions]
    frames = {}

    def image_at(i, replay=False):
        key = (i,replay)
        if key in frames:return frames[key]
        distance = DISTANCES[i]
        object_part = CPU.placed(obj, rotation, translation+direction*distance/1000)
        left = renderer.render(fixed+[object_part], full_camera, PANEL, PANEL)
        # Full path arrow denotes the object's center trajectory, 0--500 mm.
        arrow(left, center, center+direction*.5, full_camera)
        # Close-up follows the first 20 mm, then keeps the witness visible.
        ci = i if distance <= 20 else witness
        cd = DISTANCES[ci]
        close_object = renderer.render([CPU.placed(obj, rotation, translation+direction*cd/1000)], close_camera, PANEL, PANEL)
        right = fixed_close.copy()
        mask = np.any(np.asarray(close_object)<250, axis=2)
        blended = Image.blend(right, close_object, .52)
        right.paste(blended, mask=Image.fromarray(mask.astype(np.uint8)*255))
        if red_pieces[ci] is not None:
            red = renderer.render([CPU.placed(red_pieces[ci],rotation,translation)],close_camera,PANEL,PANEL)
            pixels = np.asarray(red)
            red_mask = (pixels[:,:,0] > pixels[:,:,1]*1.4) & (pixels[:,:,0] > pixels[:,:,2]*1.4)
            right.paste(red, mask=Image.fromarray(red_mask.astype(np.uint8)*255))
        arrow(right, focus-direction*.025, focus+direction*.025, close_camera, False)
        frame = Image.new('RGB',(WIDTH,HEIGHT),'white'); frame.paste(left,(0,110)); frame.paste(right,(PANEL,110))
        ink = ImageDraw.Draw(frame)
        pose = case.poses[k].replace('pose_','Pose ')
        ink.text((24,16),f'{case.output.parent.name} · {pose} · 支撑固定，物体退出',font=font(29),fill='#253e4b')
        kind = '退出失败候选支撑' if case.support_record['diagnostic_candidate'] else '原 Step4 实体（验收失败）'
        ink.text((24,58),f'{kind}　｜　浅蓝：支撑　灰：物体　红：实际相交材料（右图透视显示）',font=font(21),fill='#586d78')
        ink.text((24,101),f'全程 0–500 mm　当前 {distance:g} mm'+('　回看碰撞' if replay else ''),font=font(20),fill='#344e5c')
        label = '碰撞定格' if distance > 20 or replay else '退出近景'
        ink.text((PANEL+24,101),f'{label}　{cd:g} mm',font=font(20),fill='#b23530')
        collision = record['overlap_volumes_m3'][i] > record['volume_tolerance_m3']
        ink.text((24,HEIGHT-42),'当前相交' if collision else '当前帧未相交',font=font(22),fill=RED if collision else '#425e6d')
        ink.text((PANEL+24,HEIGHT-42),'红色仅标出此时物体和支撑重叠的部分',font=font(20),fill='#a33c35')
        frames[key]=frame
        return frame

    png = case.output/f'exit_{case.poses[k]}.png'
    image_at(witness).save(png)
    video = case.output/f'exit_{case.poses[k]}.mp4'
    sequence = [0]*12+list(range(1,len(DISTANCES)))+[len(DISTANCES)-1]*12+[witness]*18
    with mp4_writer(video, fps=FPS) as writer:
        for j,i in enumerate(sequence):writer.append_data(np.asarray(image_at(i,j>=len(sequence)-18)))
    # The actual world support stays fixed through every video frame.
    frame_points = np.stack([center+direction*d/1000 for d in DISTANCES])
    np.testing.assert_allclose(np.linalg.norm(frame_points-center,axis=1),DISTANCES/1000,atol=1e-12,rtol=0)
    record = dict(record, png=png.name, video=video.name, fps=FPS, frame_count=len(sequence),
        duration_seconds=len(sequence)/FPS, video_size=[WIDTH,HEIGHT],
        path_world_m=frame_points.tolist(), frames_are_collision_demonstration=True,
        support_stationary=True, right_view_freezes_witness_after_mm=20.,
        full_camera=[x.tolist() if hasattr(x,'tolist') else x for x in full_camera],
        close_camera=[x.tolist() if hasattr(x,'tolist') else x for x in close_camera])
    print('EXIT VIDEO', case.output.parent.name, case.poses[k], 'first sampled collision',record['first_sampled_collision_mm'],'mm',flush=True)
    return record


def viewer(case, rows):
    # Reuse the repo's offline support viewer for orbiting the actual candidate.
    from step4_connect_support.codesign_port.fixture_view import export_viewer
    folder=case.output/'data/exit_replay/viewer';folder.mkdir(exist_ok=True)
    report=dict(object=case.report['object'],poses=case.poses,passed=False,complete=True,diagnostic_only=True,
        constructed=True,presentation_description=case.support_record['description'],
        source_schedule=str((case.output/'data/codesign_inputs_surface/data/independent_input/schedule.json').relative_to(I.ROOT)))
    colors={p['id']:BLUE for p in case.patches}
    export_viewer(folder,case.support,[(p['id'],p['mesh']) for p in case.patches],report,case.tasks,case.bases,case.offsets,colors)
    text=(folder/'index.html').read_text().replace('href="fixture.obj"','href="exit_support.obj"').replace('<a href="fixture_mm.stl" download>STL · mm</a>','')
    # This viewer also serves head-only previews. Our diagnostic contains a
    # complete support, so do not inherit its "no complete support" UI branch.
    text=text.replace('if(DATA.report.diagnostic_only){','if(DATA.report.diagnostic_only && !DATA.report.constructed){')
    text=text.replace('=DATA.report.diagnostic_only\n','=DATA.report.diagnostic_only && !DATA.report.constructed\n')
    (case.output/'exit_support.html').write_text(text)


STYLE='<style>body{font:16px system-ui;color:#294450;background:#f4f7f9;max-width:1500px;margin:26px auto;padding:0 20px}h1{font-size:28px}p{line-height:1.7}a{color:#216e9a}section{background:white;border-radius:12px;padding:22px;margin:22px 0}video,img{width:100%;border-radius:8px}nav{display:flex;gap:14px;flex-wrap:wrap}small{color:#6d7d86} .warning{color:#ad4138}</style>'


def page(case, rows):
    intro=f'<h1>{case.output.parent.name}：所有 pose 的支撑与退出视频</h1><p>{case.support_record["description"]}。每个视频左侧显示完整 500 mm 路径，右侧放大起始退出；超过 20 mm 后右侧保留碰撞定格。</p><p>浅蓝色支撑固定，灰色物体沿紫色箭头移动，红色为实际相交材料。发生碰撞后仍继续播放，是为了展示冲突位置。</p><p><a href="exit_support.html">旋转查看支撑</a> · <a href="exit_support.obj">下载候选 OBJ</a></p>'
    cards=[]
    for row in rows:
        ident=row['pose']; first=row['first_sampled_collision_mm']
        finding=f'首个采样碰撞：{first:g} mm' if first is not None else '显示的采样帧未发现相交'
        detail='；碰撞定格涉及根部：'+', '.join(row['root_ids_at_witness']) if row['root_ids_at_witness'] else ''
        coverage=f'原 Step3 覆盖 {row["original_step3_covered"]:,}/32,768'
        cards.append(f'<section id="{ident}"><h2>{ident.replace("pose_","Pose ")}</h2><p class="warning">{finding}{detail}</p><p><small>{coverage}；此展示不改变原完整验收结果。</small></p><video controls preload="metadata" poster="{row["png"]}" src="{row["video"]}"></video><p><a href="{row["video"]}">下载视频</a> · <a href="{row["png"]}">打开方向与碰撞图</a></p></section>')
    (case.output/'exit_paths.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+STYLE+intro+''.join(cards))


def run(output):
    case=prepare(output)
    renderer=CPU.Renderer()
    rows=[render_pose(case,k,renderer) for k in range(len(case.poses))]
    viewer(case,rows);page(case,rows)
    support_json=case.output/'data/exit_replay/support.json'
    files=['exit_support.obj','exit_support.html','exit_paths.html']+[r[key] for r in rows for key in ('png','video')]
    inputs=[support_json]+[case.output/f'data/exit_replay/collision_{p}.json' for p in case.poses]
    I.save(case.output/'data/exit_replay.json',dict(complete=True,diagnostic_only=True,poses=case.poses,
        group=case.output.parent.name,support=case.support_record,per_pose=rows,
        all_poses_included=len(rows)==len(case.poses),original_acceptance_unchanged=True,
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(E.__file__),Path(CPU.__file__),Path(CPU.__file__).with_name('raster.cpp')])),
        artifacts={'../'+n:I.sha256(case.output/n) for n in files}))
    # Recheck physical source reports after publication: no search inputs or
    # accepted/rejected outcomes may have been changed by this illustration.
    I.check_report(case.output/'data/report.json')
    print('EXIT GROUP COMPLETE',case.output.parent.name,len(rows),flush=True)


def gallery(name):
    base=I.OUTPUTS/name
    groups=sorted((p for p in base.glob('pose*+*') if (p/'step4/data/report.json').exists()),key=lambda p:(len(p.name.split('+')),p.name))
    cards=[];count=0
    for g in groups:
        report=I.check_report(g/'step4/data/exit_replay.json')
        count+=len(report['per_pose'])
        path=g.name+'/step4/'
        links=' · '.join(f'<a href="{path}exit_paths.html#{r["pose"]}">{r["pose"].replace("pose_","Pose ")}</a>' for r in report['per_pose'])
        cards.append(f'<section><h2><a href="{path}exit_paths.html">{g.name}</a></h2><p>{report["support"]["description"]}</p><nav>{links}</nav><a href="{path}exit_paths.html"><img loading="lazy" src="{path}{report["per_pose"][0]["png"]}"></a></section>')
    intro=f'<h1>B：8 组、{count} 个组内 pose 的退出图与视频</h1><p>包含全部失败组。每组都有同一物体周围的支撑实体和每个 pose 的实际尝试方向；浅蓝支撑固定、灰色物体移动、红色标出相交。</p><p>pose3+6 使用原 Step4 实体；其他 7 组为本次补建的失败候选，保留会挡住退出的材料。原来的搜索与验收结果不变。</p>'
    (base/'exit_paths.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+STYLE+intro+''.join(cards))
    print('ALL EXIT PATHS',len(groups),count,base/'exit_paths.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs',type=Path,nargs='*')
    parser.add_argument('--gallery',metavar='OBJECT')
    args=parser.parse_args()
    for output in args.outputs:run(output.resolve())
    if args.gallery:gallery(args.gallery)
