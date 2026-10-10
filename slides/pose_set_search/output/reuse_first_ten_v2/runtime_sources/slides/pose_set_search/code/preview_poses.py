"""Ten alternative object configurations in ONE fixed final fixture placement."""
import hashlib
import importlib.util
import json
import os

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/pose_set_search_numba')
from common import *
from PIL import Image, ImageDraw


def arrow(origin, direction):
    transform = C.trimesh.geometry.align_vectors([0., 0., 1.], direction)
    shaft = C.trimesh.creation.cylinder(radius=.0013, height=.028, sections=16)
    shaft.apply_transform(transform)
    shaft.apply_translation(origin + .014*direction)
    tip = C.trimesh.creation.cone(radius=.0038, height=.009, sections=16)
    tip.apply_transform(transform)
    tip.apply_translation(origin + .028*direction)
    return C.trimesh.util.concatenate([shaft, tip])


def main():
    folder = HERE / 'output/ten_v10/pose1-10/joint'
    report = json.loads((folder/'report.json').read_text())
    assert report['force_exit_work_passed']
    with np.load(folder/'layout.npz') as data:
        native = data['native_world'].copy()
        placements = data['placements'].copy()
        hosts = data['hosts'].copy()
        directions = data['directions'].copy()
    assert np.all(hosts == 0)
    path = CO / 'operation_demo/combined/code/geometry.py'
    spec = importlib.util.spec_from_file_location('pose_set_pose_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    R = module.R
    R.COLORS['exit'] = '#efb622'
    support = transform_mesh(C.trimesh.load(folder/'support.obj', process=False), native[0])
    task, _, mesh = C.state('B', 'pose_1')
    bodies, arrows, shifts = [], [], []
    for k in range(10):
        world = native[hosts[k]] @ placements[k]
        body = transform_mesh(mesh, world)
        direction = native[hosts[k], :3, :3] @ directions[k]
        np.testing.assert_allclose(direction, [0., 0., 1.], atol=1e-10)
        top = np.max((body.vertices-body.center_mass) @ direction)
        origin = body.center_mass + (top+.004)*direction
        bodies.append(body)
        arrows.append(arrow(origin, direction))
        shifts.append(float(np.linalg.norm((world[:3, 3]-native[k, :3, 3])[:2])*1000))
    points = np.vstack([support.vertices]+[body.vertices for body in bodies]+[a.vertices for a in arrows])
    azimuth, elevation = np.radians(-55.), np.arctan(1/np.sqrt(2))
    camera = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    up = np.cross(camera, right)
    center = sum(axis*((points @ axis).min()+(points @ axis).max())/2
                 for axis in [right, up, camera])
    scale = min(405/np.ptp(points @ right), 290/np.ptp(points @ up))*.98
    canvas = Image.new('RGB', (2250, 960), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), 'Joint result | pose 1–10 | one fixed fixture placement',
              font=R.font(27, True), fill='#273340')
    draw.text((30, 66), 'Blue: same final support. Gray: each alternative object pose. Yellow: final exit direction (+Z). Same camera and scale.',
              font=R.font(19), fill='#46505a')
    for k, (body, exit_arrow) in enumerate(zip(bodies, arrows)):
        x, y = (k % 5)*450, 110+(k//5)*400
        renderer = R.Renderer(dict(initial=support, final=support, host=body, guest=body),
                              dict(guest_direction_world=[0., 0., 1.]), width=450, height=350,
                              annotations=False, guest_only=True)
        renderer.center, renderer.scale = center, scale
        renderer.camera, renderer.right, renderer.up = camera, right, up
        frame = renderer.frame([(support, 'support'), (exit_arrow, 'exit')], 0., '', guest_visible=True)
        canvas.paste(Image.fromarray(frame), (x, y+30))
        draw.text((x+18, y), f'Pose {k+1}  |  lateral shift {shifts[k]:.1f} mm',
                  font=R.font(18, True), fill='#273340')
    draw.text((30, 929), 'Each pose: 32,768 original loads verified. Force / exit / work accepted; connectivity, installation and strength not certified.',
              font=R.font(17), fill='#46505a')
    out = HERE / 'vis'
    out.mkdir(exist_ok=True)
    canvas.save(out/'pose1-10_joint.png')
    C.save(out/'pose1-10_sources.json', dict(
        source_result=str(folder.relative_to(ROOT)),
        support_sha256=hashlib.sha256((folder/'support.obj').read_bytes()).hexdigest(),
        layout_sha256=hashlib.sha256((folder/'layout.npz').read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
        body_inputs_sha256=C.I.hashes(task.inputs),
        fixture_world_transform=native[0].tolist(), lateral_displacements_mm=shifts,
        support_geometry_identical_in_all_panels=True,
        azimuth_degrees=-55., elevation_degrees=float(np.degrees(elevation)),
        renderer=str((CO/'operation_demo/juxtapose/code/render.py').relative_to(ROOT))))
    print(out/'pose1-10_joint.png')


if __name__ == '__main__':
    main()
