"""Render fixed-camera columns: each column uses one real support installation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from whole_task_2.step1.common import write_json, sha256
from whole_task_2.common.render import camera_fit
from .angular_model import AngularTaskModel, AngularLayout


def render(folder):
    from whole_task_2.step5.video import raster as r
    import moderngl
    import trimesh
    folder = Path(folder); report = json.loads((folder/'report.json').read_text())
    model = AngularTaskModel(name=report['object'], output_root=report['input_root'])
    with np.load(folder/'step4.2/layout.npz') as z:
        layout = AngularLayout.from_arrays(z)
    model.activate(layout)
    support_path = folder/'step4.2/complete_support.obj'
    support = trimesh.load(support_path, force='mesh', process=False)
    groups = report['support_installations']['groups']
    width, height, label, header = 640, 580, 38, 52
    board = Image.new('RGB', (width*len(groups), header+(height+label)*max(len(g['working_areas']) for g in groups)), 'white')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    ctx = r.context(); studio = r.Studio(ctx, width, height)
    fragment = r.FRAGMENT.replace('float s=shadow(n,key);', 'float s=mix(.75,1.,shadow(n,key));')
    shader = ctx.program(vertex_shader=r.VERTEX, fragment_shader=fragment)
    shadow_shader = ctx.program(vertex_shader=r.SHADOW_VERTEX, fragment_shader=r.SHADOW_FRAGMENT)
    body = r.Shape(ctx, shader, model.mesh.vertices, model.mesh.faces, r.GRAY, dynamic=True, roughness=.65)
    fixture = r.Shape(ctx, shader, support.vertices, support.faces, r.BLUE, smooth=False, roughness=.55)
    key = np.array([-.65, -.9, 1.45]); key /= np.linalg.norm(key)
    shader['shadow_map'].value = 3
    execution = {row['pose']: row for row in [model.execution(layout, k) for k in layout.active]}
    scenes = []
    try:
        for column, group in enumerate(groups):
            t = np.asarray(group['world_from_support'])
            clouds = [support.vertices @ t[:3, :3].T+t[:3, 3]]
            for pose in group['working_areas']:
                u = np.asarray(execution[pose]['world_from_object'])
                clouds.append(model.mesh.vertices @ u[:3, :3].T+u[:3, 3])
                np.testing.assert_allclose(execution[pose]['world_from_support'], t, atol=1e-10, rtol=0)
            cloud = np.vstack(clouds)
            vp, target, direction = camera_fit(r, [.1, 1., 1.15], cloud, width, height, 1.12)
            light_vp = r.orthographic(2., 2.) @ r.view_matrix(key, cloud.mean(axis=0))
            for program in [shader, shadow_shader]:
                program['light_projection'].write(np.asarray(light_vp, dtype='f4').T.tobytes())
            shader['view_projection'].write(np.asarray(vp, dtype='f4').T.tobytes())
            shader['eye'].value = tuple(target+direction*4)
            pen.text((column*width+16, 13), f"Support {column+1}: "+' / '.join(group['working_areas']), font=font, fill='#434b53')
            for row_index, pose in enumerate(group['working_areas']):
                k = model.poses.index(pose); u = np.asarray(execution[pose]['world_from_object'])
                colors = np.tile(r.GRAY, (len(model.mesh.faces), 1)); colors[model.tasks[k].domain.work_ids] = r.PINK
                body.paint(colors)
                draws = [(fixture, t), (body, u)]
                ctx.enable(moderngl.DEPTH_TEST)
                studio.shadow_buffer.use(); ctx.viewport = (0, 0, 1536, 1536); studio.shadow_buffer.clear(depth=1)
                for shape, transform in draws:
                    shape.shadow(ctx, shadow_shader, transform)
                studio.gbuffer.use(); ctx.viewport = (0, 0, *studio.size); studio.gbuffer.clear(1, 1, 1, 1, depth=1)
                studio.shadow.use(3)
                for shape, transform in draws:
                    shape.draw(shader, transform)
                image = Image.frombytes('RGB', (width, height), studio.finish(vp, direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                x = column*width; y = header+row_index*(height+label)
                board.paste(image, (x, y))
                pen.text((x+16, y+height+5), pose, font=font, fill='#434b53')
            scenes.append(dict(working_areas=group['working_areas'], world_from_support=t.tolist(),
                camera_target_world=np.asarray(target).tolist(), camera_direction_world=np.asarray(direction).tolist(),
                camera_fixed_within_group=True, support_fixed_within_group=True))
        board.save(folder/'grouped_overview.png')
    finally:
        ctx.release()
    write_json(folder/'grouped_visualization.json', dict(renderer='OpenGL rasterization',
        actual_complete_fixture_sha256=sha256(support_path), source_sha256=sha256(__file__),
        report_sha256=sha256(folder/'report.json'), figure_sha256=sha256(folder/'grouped_overview.png'),
        full_fixture_in_every_view=True, camera_fixed_within_each_column=True, groups=scenes,
        region_color='pink', support_color='blue'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    render(parser.parse_args().folder)
