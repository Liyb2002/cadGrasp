"""Publish checked layouts in the requested Co-optimize/B/step4.2 directory."""
import argparse
import hashlib
import json
import os
import shutil
import time

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/pose_set_search_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pose_set_search_matplotlib')
from common import *
from PIL import Image, ImageDraw, ImageFont, ImageOps
from step41_render import draw_pose
from reuse_first import registered
from model import Layout
from case_sets import CASES


def sheet(tiles, labels, path):
    columns = min(5, len(tiles));size = 760
    canvas = Image.new('RGB', (columns*size, ((len(tiles)+columns-1)//columns)*(size+45)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    for k, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (k % columns)*size, (k//columns)*(size+45)
        thumb = ImageOps.contain(tile, (size-30, size-30), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x+(size-thumb.width)//2, y+45+(size-thumb.height)//2))
        draw.text((x+25, y+16), label, fill='#303840', font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    source = args.result.resolve()
    report = json.loads((source/'report.json').read_text())
    case, mode = source.parent.name, source.name
    assert report['pose_count'] == len(CASES[case])
    assert report['force_exit_work_passed'] and report.get('every_insertion_passed', True)
    proof = json.loads((source/'pressure_audit.json').read_text())
    assert proof['complete'] and proof['passed']
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert proof['support_sha256'] == digest(source/'support.obj')
    out = args.out or CO/'output/B/step4.2'/case/mode
    out.mkdir(parents=True, exist_ok=False)
    data = out/'data'
    shutil.copytree(source, data)
    saved = np.load(source/'layout.npz')
    layout = Layout(saved['placements'], saved['directions'], saved['hosts'], tuple(map(int, saved['active'])))
    native = saved['native_world']
    support = C.trimesh.load(source/'support.obj', process=False)
    _, _, mesh = C.state('B', report['poses'][0])
    final, sweeps, labels, states = [], [], [], []
    began = time.monotonic()
    for k, pose in enumerate(report['poses']):
        host = int(layout.hosts[k])
        frame = native[host]
        body = transform_mesh(mesh, layout.placements[k])
        direction = layout.directions[k]
        world = frame @ layout.placements[k]
        np.testing.assert_allclose(world[:3, :3], native[k, :3, :3], atol=1e-10, rtol=0)
        assert abs(world[2, 3]-native[k, 2, 3]) < 1e-9
        final.append(draw_pose(body, support, C.trimesh.Trimesh(), frame, arrow=True, direction=direction))
        display = C.S.swept_solid(body, min(.10, report['full_exit_length_m'])*direction)
        sweeps.append(draw_pose(body, support, C.trimesh.Trimesh(), frame, arrow=True, direction=direction, sweep=display))
        is_native = registered(layout, k)
        labels.append(f'{pose.replace("_", " ")} | rotate fixture' if is_native else
                      f'{pose.replace("_", " ")} | juxtapose into {report["poses"][host].replace("_", " ")}')
        states.append(dict(pose=pose, mode='single_use_rotating_fixture' if is_native else 'juxtaposed',
                           host=report['poses'][host], fixture_to_world=frame.tolist(),
                           object_to_fixture=layout.placements[k].tolist(),
                           direction_world=(frame[:3, :3] @ direction).tolist(),
                           support_minimum_world_z_m=float(C.transform_points(support.vertices, frame)[:, 2].min())))
        print('PUBLISHED POSE', case, mode, pose, labels[-1], flush=True)
    sheet(final, labels, out/'final_result.png')
    sheet(sweeps, labels, out/'exit_sweeps.png')
    C.save(out/'render.json', dict(complete=True, source_result=str(source.relative_to(ROOT)),
        source_support_sha256=digest(source/'support.obj'), source_layout_sha256=digest(source/'layout.npz'),
        same_rigid_support_shape_in_every_pose=True, support_not_clipped_or_changed_for_display=True,
        style_source='slides/Co-optimize/vis_func/step41_render.py:draw_pose',
        full_exit_length_m=report['full_exit_length_m'], display_sweep_length_m=min(.10, report['full_exit_length_m']),
        states=states, seconds=time.monotonic()-began,
        artifacts={name:digest(out/name) for name in ['final_result.png', 'exit_sweeps.png']}))
    (out/'README.md').write_text(
        '# 转动复用优先，再选择性 Juxtapose\n\n'
        f'集合：{", ".join(report["poses"])}。方法：{mode}。材料体积 **{report["volume_cm3"]:.3f} cm³**。'
        f'保留转动复用 {report["rotating_reuse_pose_count"]} 个 pose；Juxtapose {report["juxtaposed_pose_count"]} 个 pose。\n\n'
        '目标是在全部原始力／力矩、退出和工作面约束满足后减少实体支撑体积；支撑摆放数不作惩罚。'
        '每格都是同一份蓝色刚性支撑在保存的支撑摆放下、灰色物体在保存的相对配置中，没有为了图像删掉材料。\n\n'
        '[最终全部 pose](final_result.png) · [退出路径](exit_sweeps.png) · [原始结果](data/report.json) · '
        '[实体支撑](data/support.obj) · [全部非负反力证书](data/pressure_audit.json)。\n\n'
        '路径图按 Step4.1 风格显示前 100 mm；实际验收完整路径为 '
        f'{report["full_exit_length_m"]*1000:.1f} mm。每个 pose 验收全部 32,768 个原始载荷，含第七维不上抬条件。'
        '实体连通、安装接地和强度尚未验收。\n')
    print('PUBLISHED', out, flush=True)


if __name__ == '__main__':
    main()
