"""Render the five operation demonstrations with their recorded design timelines.

Run as: python -m whole_task_2.operations.render_videos
This is a sequence of design states, not a robot motion/assembly certificate.
"""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.needle_render import sector_lines
from whole_task_2.step3.render_independent import Renderer, head_context, BLUE
from whole_task_2.step5.video import raster
from .geometry import OperationModel, moved_mesh
from .state import transform
from .pose_change import pose_change
from .wrapped_shell import wrapped_state
from whole_task_2.step3.run_independent import make_domain
from whole_task_2.step3.independent_physics import demands, certificate
from whole_task_2.step1.needle_geometry import geometric_mask

HERE = Path(__file__).parent
DATA = ROOT / 'whole_task_2/data/B/operation_videos'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def cached_state(model, layout, tag, index):
    folder = DATA / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{index:03}.npz'
    record = path.with_suffix('.json')
    input_binding=sha256(model.folder/'report.json')
    if path.exists() and record.exists():
        meta = json.loads(record.read_text())
        with np.load(path) as z:
            if (meta.get('input_step32_report_sha256')==input_binding
                    and meta['mesh_sha256'] == sha256(path)
                    and all(np.array_equal(z[k], v) for k, v in layout.arrays().items())):
                return dict(layout=layout.copy(), support=trimesh.Trimesh(z['vertices'], z['faces'], process=False),
                            volume_cm3=meta['volume_cm3'], audit=meta['audit'])
    candidate = model.accept(layout)
    np.savez_compressed(path, **layout.arrays(), vertices=candidate['support'].vertices,
                        faces=candidate['support'].faces)
    write_json(record, dict(mesh_sha256=sha256(path), volume_cm3=candidate['volume_cm3'], audit=candidate['audit'],
                           input_step32_report_sha256=input_binding))
    print(tag, index, 'PASS', round(candidate['volume_cm3'], 3), flush=True)
    return candidate


def pose_states():
    model = OperationModel(task_ids=['W02'])
    q = model.initial.copy()
    q.material[0] = np.eye(4)
    base = wrapped_state(model, cached_state(model, q, 'pose', 0), 'pose_shell', 0)
    states = []
    folder = DATA / 'pose90'
    folder.mkdir(parents=True, exist_ok=True)
    binding = sha256(DATA / 'pose_shell/000.npz')
    for i, angle in enumerate(np.linspace(0., 90., 61)):
        p = pose_change(q, 0, angle, model.axis, model.head)
        delta = p.poses[0] @ np.linalg.inv(q.poses[0])
        support = moved_mesh(base['support'], delta)
        path = folder / f'{i:03}.json'
        old = json.loads(path.read_text()) if path.exists() else {}
        pose = p.poses[0]
        head = (model.head - pose[:3, 3]) @ pose[:3, :3]
        assert geometric_mask(model.source, head, pose[:3, :3].T @ model.axis,
            model.workspace)[model.records[0]['working_face_ids']].all()
        if old.get('source_shell_sha256') == binding and old.get('angle_deg') == float(angle):
            proof = old['force']
        else:
            domain = make_domain(model.source, pose, model.records[0]['working_face_ids'],
                model.ids[0], model.workspace, model.physical, model.policy)
            endpoints, samples = demands(domain, model.policy)
            proof, _ = certificate(moved_mesh(model.source, pose), support,
                domain, endpoints, samples, search=True)
            if not proof['passed']:
                raise RuntimeError(f'Pose {angle:g} degrees fails the shell contact loads')
            write_json(path, dict(angle_deg=float(angle), source_shell_sha256=binding,
                force=proof, world_from_object=pose.tolist(), working_area_inside_sector=True))
        states.append(dict(layout=p, support=support, volume_cm3=float(support.volume*1e6),
            audit=dict(passed=True, force=proof), support_representation='wrapped_shell'))
        if i % 10 == 0:
            print('pose90 CHECK', angle, 'degrees', flush=True)
    return model, states


def video_size(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    return stream['width'], stream['height']


def make_video(model, states, filename, title=None, size=(1280, 800)):
    width, height = size
    fps, count = 24, 192
    panels = len(model.ids)
    tile_width, tile_height = width // panels, height - 110 if title else height
    renderer = Renderer(tile_width, tile_height)
    contexts = head_context(model.workspace)
    lines = sector_lines(model.workspace)
    body_masks = []
    for record in model.records:
        mask = np.zeros(len(model.source.faces), bool)
        mask[record['working_face_ids']] = True
        body_masks.append(mask)
    clouds = []
    for state in states:
        for k in range(panels):
            clouds.append(transform(state['support'].vertices, np.linalg.inv(state['layout'].material[k])))
            clouds.append(transform(model.source.vertices, state['layout'].poses[k]))
    points = np.vstack([*clouds, lines, *[m.vertices for m, _, _ in contexts]])
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(HERE / filename)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    rendered = []
    try:
        for index, state in enumerate(states):
            canvas = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(canvas)
            if title:
                draw.text((30, 18), title, font=ImageFont.truetype(FONT, 29), fill='#29475c')
                draw.text((30, 60), 'Fully checked design states  |  fixed working head and original pink region',
                          font=ImageFont.truetype(FONT, 18), fill='#617481')
            for k, name in enumerate(model.ids):
                support = moved_mesh(state['support'], np.linalg.inv(state['layout'].material[k]))
                body = moved_mesh(model.source, state['layout'].poses[k])
                image = renderer.draw([(support, BLUE, None),
                    (body, raster.GRAY, body_masks[k]), *contexts], points, [1.1, 1.4, .95], lines)
                canvas.paste(image, (tile_width*k, 95 if title else 0))
                if title:
                    draw.text((tile_width*k+24, height-55), name, font=ImageFont.truetype(FONT, 23), fill='#29475c')
            if title:
                draw.text((width-305, height-55), f"Material {state['volume_cm3']:,.0f} cm3", font=ImageFont.truetype(FONT, 23), fill='#29475c')
            rendered.append(canvas)
            print(filename, 'render', index+1, '/', len(states), flush=True)
        for frame in range(count):
            alpha = np.clip((frame-24)/(count-49), 0, 1)
            index = min(len(rendered)-1, round(alpha*(len(rendered)-1)))
            process.stdin.write(rendered[index].tobytes())
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')
    sheet = Image.new('RGB', (width*3, height*2), 'white')
    for j, index in enumerate(np.linspace(0, len(rendered)-1, 6).astype(int)):
        sheet.paste(rendered[index], ((j%3)*width, (j//3)*height))
    sheet.save(DATA / (Path(filename).stem + '_review.jpg'))
    return dict(video_sha256=sha256(HERE / filename), states=len(states),
        all_displayed_states_geometry_checked=True, continuous_robot_motion_claimed=False,
        support_representation='wrapped_shell', no_base_plate=True, no_vertical_backing=True,
        initial_volume_cm3=states[0]['volume_cm3'], final_volume_cm3=states[-1]['volume_cm3'],
        width=width, height=height, text_overlays=bool(title), frames=count, fps=fps)


def retire_old_videos():
    manifest = DATA / 'videos.json'
    metadata = json.loads(manifest.read_text())
    if all((HERE / name).exists() and name in metadata for name in ('juxtaposition.mp4', 'rotational_reuse.mp4')):
        for name in ('support_merge.mp4', 'support_turn.mp4'):
            (HERE / name).unlink(missing_ok=True)
            metadata.pop(name, None)
        write_json(manifest, metadata)


def main(only=None):
    only = {'pose_rotation': 'pose', 'juxtaposition': 'juxtapose',
            'rotational_reuse': 'merge', 'exit_direction': 'exit_redirection'}.get(only, only)
    if only in (None, 'exit_redirection', 'translation'):
        from .render_refinements import main as render_refinements
        render_refinements(only)
    if only in (None, 'pose'):
        from .render_pose_rotation import main as render_pose_rotation
        render_pose_rotation()
    if only in (None, 'juxtapose'):
        from .render_juxtapose import main as render_juxtapose
        render_juxtapose()
    if only in (None, 'merge'):
        from .render_rotational_reuse import main as render_merge
        render_merge()
    retire_old_videos()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['pose_rotation', 'juxtaposition', 'rotational_reuse',
                                         'exit_redirection', 'exit_direction', 'translation',
                                         'pose', 'juxtapose', 'merge'])
    main(parser.parse_args().only)
