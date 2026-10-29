"""Rasterize the complete fixture and all 13 actual centered executions."""
import json
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from subdivide_task.step1.common import sha256, write_json
from subdivide_task.common.render import camera_fit, cone_segments


def render(model, layout, support, folder):
    from subdivide_task.step5.video import raster as r
    import moderngl
    model.activate(layout); folder = Path(folder)
    width, height, label = 720, 620, 46
    columns = 4; count = len(layout.active)+1
    canvas = Image.new('RGB', (columns*width, math.ceil(count/columns)*(height+label)), 'white')
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    ctx = r.context(); studio = r.Studio(ctx, width, height)
    fragment = r.FRAGMENT.replace('float s=shadow(n,key);', 'float s=mix(.75,1.,shadow(n,key));')
    shader = ctx.program(vertex_shader=r.VERTEX, fragment_shader=fragment)
    shadow_shader = ctx.program(vertex_shader=r.SHADOW_VERTEX, fragment_shader=r.SHADOW_FRAGMENT)
    body = r.Shape(ctx, shader, model.mesh.vertices, model.mesh.faces, r.GRAY, dynamic=True, roughness=.65)
    fixture = r.Shape(ctx, shader, support.vertices, support.faces, r.BLUE, smooth=False, roughness=.55)
    key = np.array([-.65, -.9, 1.45]); key /= np.linalg.norm(key)
    shader['shadow_map'].value = 3

    def picture(draws, view, target_points):
        vp, target, direction = camera_fit(r, view, target_points, width, height, 1.12)
        light_vp = r.orthographic(2., 2.) @ r.view_matrix(key, np.asarray(target_points).mean(axis=0))
        for program in [shader, shadow_shader]:
            program['light_projection'].write(np.asarray(light_vp, dtype='f4').T.tobytes())
        shader['view_projection'].write(np.asarray(vp, dtype='f4').T.tobytes())
        shader['eye'].value = tuple(target+direction*4)
        ctx.enable(moderngl.DEPTH_TEST)
        studio.shadow_buffer.use(); ctx.viewport = (0, 0, 1536, 1536); studio.shadow_buffer.clear(depth=1)
        for shape, transform in draws:
            shape.shadow(ctx, shadow_shader, transform)
        studio.gbuffer.use(); ctx.viewport = (0, 0, *studio.size); studio.gbuffer.clear(1, 1, 1, 1, depth=1)
        studio.shadow.use(3)
        for shape, transform in draws:
            shape.draw(shader, transform)
        return Image.frombytes('RGB', (width, height), studio.finish(vp, direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    scenes = []
    try:
        initial = model.execution(layout, layout.active[0])
        t = np.asarray(initial['world_from_support'])
        cloud = support.vertices @ t[:3, :3].T+t[:3, 3]
        canvas.paste(picture([(fixture, t)], [-.8, -1.2, 1.2], cloud), (0, 0))
        pen.text((18, height+5), 'Shared fixture / 13 working areas', font=font, fill='#434b53')
        for tile, k in enumerate(layout.active, 1):
            row = model.execution(layout, k)
            u, t = np.asarray(row['world_from_object']), np.asarray(row['world_from_support'])
            colors = np.tile(r.GRAY, (len(model.mesh.faces), 1)); colors[model.tasks[k].domain.work_ids] = r.PINK
            body.paint(colors)
            cloud = model.mesh.vertices @ u[:3, :3].T+u[:3, 3]
            # Closeup shows the bunny and its real local fixture contacts;
            # the first tile separately shows the entire shared solid.
            image = picture([(fixture, t), (body, u)], [-1.1, .8, 1.0], cloud)
            x, y = (tile % columns)*width, (tile//columns)*(height+label)
            canvas.paste(image, (x, y))
            pen.text((x+18, y+height+5), f"{model.poses[k]}   {layout.angles_deg[k] % 360.:.2f}°", font=font, fill='#434b53')
            scenes.append(dict(pose=model.poses[k], axial_angle_deg=float(layout.angles_deg[k] % 360.),
                world_from_object=u.tolist(), world_from_support=t.tolist(), complete_support_used=True))
        canvas.save(folder/'execution_overview.png')
    finally:
        ctx.release()
    result = dict(renderer='OpenGL rasterization', complete_actual_fixture=True,
        region_color='pink', support_color='blue', scenes=scenes,
        report_sha256=sha256(folder/'report.json'), figure_sha256=sha256(folder/'execution_overview.png'),
        source_sha256=sha256(__file__), per_pose_view='bunny_contact_closeup',
        first_tile='entire_shared_fixture')
    write_json(folder/'visualization.json', result)
    return result
