"""Show the saved blue wrap and gray object in each pose's native placement."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps


def render_pose(name, group, pose, base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = base / 'step3.2'
    support_path = base / 'step3.2/wrapped_support.obj'
    object_path = base / 'step3.1/registered_object.obj'
    task, transform, _ = state(name, pose)
    support = trimesh.load(support_path, force='mesh', process=False)
    obj = trimesh.load(object_path, force='mesh', process=False)
    support.apply_transform(transform)
    obj.apply_transform(transform)
    np.testing.assert_allclose(obj.vertices, task.domain.mesh.vertices, atol=1e-10, rtol=0)

    # Fixed native-world orthographic isometric camera.
    azimuth = -45.
    elevation = float(np.degrees(np.arctan(1/np.sqrt(2))))
    camera = np.array([np.cos(np.radians(elevation))*np.cos(np.radians(azimuth)),
                       np.cos(np.radians(elevation))*np.sin(np.radians(azimuth)),
                       np.sin(np.radians(elevation))])
    light = camera + np.array([-.15, -.2, .65])
    light /= np.linalg.norm(light)

    def colors(mesh, color):
        rgb = np.array(matplotlib.colors.to_rgb(color))
        brightness = .58 + .42*np.maximum(mesh.face_normals @ light, 0)
        return np.c_[brightness[:, None]*rgb, np.ones(len(mesh.faces))]

    # One opaque collection sorts object and support faces together.
    # The gray object is visible through actual work openings.
    triangles = np.concatenate([obj.triangles, support.triangles])*1000
    facecolors = np.concatenate([colors(obj, '#a4a8ac'), colors(support, '#319cd7')])
    fig = plt.figure(figsize=(9, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=facecolors,
                                        edgecolors='none', zsort='average'))
    points = np.vstack([obj.vertices, support.vertices])*1000
    center = (points.min(0)+points.max(0))/2
    radius = float(np.ptp(points, axis=0).max())*.53
    for axis, value in zip('xyz', center):
        getattr(ax, 'set_'+axis+'lim')(value-radius, value+radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elevation, azim=azimuth, roll=0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    buffer = BytesIO()
    fig.savefig(buffer, dpi=180, facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    tile = Image.open(buffer).convert('RGB')
    pixels = np.asarray(tile)
    ys, xs = np.where(np.any(pixels < 245, axis=2))
    if len(xs):
        tile = tile.crop((max(0, xs.min()-20), max(0, ys.min()-20),
                          min(tile.width, xs.max()+21), min(tile.height, ys.max()+21)))
    inputs = [object_path, support_path, *task.inputs]
    record = dict(complete=True, presentation_only=True, pose_set=group['id'],
                  displayed_pose=pose, object_instances=1,
                  T_fixture_to_world=transform.tolist(),
                  support_source='step3.2/wrapped_support.obj',
                  support_color='blue', object_color='gray', opacity=1.,
                  working_faces_exposed_by_actual_support_openings=True,
                  view_elevation_deg=elevation, view_azimuth_deg=azimuth,
                  projection='orthographic', camera_frame='native world', view_kind='isometric',
                  geometry_changed=False, mechanics_rerun=False,
                  provenance=provenance(inputs, [Path(__file__)]))
    return tile, record


def render(name, group, base=None):
    base = base or (HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name) / group['id'] / 'step3'
    out = base / 'step3.2'
    panels = [render_pose(name, group, pose, base) for pose in group['poses']]
    columns = 2 if len(panels) == 4 else min(3, len(panels))
    rows = (len(panels)+columns-1)//columns
    width, height = 900, 760
    overview = Image.new('RGB', (columns*width, rows*height), 'white')
    draw = ImageDraw.Draw(overview)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    for index, (tile, record) in enumerate(panels):
        x, y = (index%columns)*width, (index//columns)*height
        fitted = ImageOps.contain(tile, (width-140, height-40), Image.Resampling.LANCZOS)
        overview.paste(fitted, (x+130+(width-140-fitted.width)//2,
                               y+(height-fitted.height)//2))
        label = record['displayed_pose'].replace('_', ' ')
        draw.text((x+12, y+height//2-20), label, fill='#303840', font=font)
    image_path = out / 'overview.png'
    overview.save(image_path)
    # Retain only the overview as the public Step3.2 image.
    for image in out.glob('*.png'):
        if image.name != 'overview.png':
            image.unlink()
    for image in (base / 'step3.1').glob('*.png'):
        image.unlink()
    (base / 'step3.1/data/render.json').unlink(missing_ok=True)
    for record_path in (out / 'data').glob('render_pose_*.json'):
        record_path.unlink()
    records = [record for tile, record in panels]
    inputs = sorted({ROOT / path for record in records for path in record['provenance']['inputs']})
    save(out / 'data/render.json', dict(complete=True, presentation_only=True,
         pose_set=group['id'], poses=group['poses'], states=records,
         panel_columns=columns, panel_rows=rows, pose_labels=True,
         image_size_px=list(overview.size), geometry_changed=False, mechanics_rerun=False,
         provenance=provenance(inputs, [Path(__file__)]),
         artifacts={'../overview.png': I.sha256(image_path)}))
    I.check_report(out / 'data/render.json')
    summary_path = base / 'data/report.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary['artifacts'] = {path: digest for path, digest in summary['artifacts'].items()
                                if not path.startswith('../step3.2/') and path != '../step3.1/overview.png'}
        summary['artifacts']['../step3.2/overview.png'] = I.sha256(image_path)
        save(summary_path, summary)
    readme = base / 'README.md'
    if readme.exists():
        text = readme.read_text().split('Step3.1 只保存注册模型与变换，不发布图片。')[0]
        lines = text.rstrip().splitlines()
        lines = [line for line in lines if '`step3.1/overview.png`' not in line and not line.startswith('Step3.2 图片：')]
        lines += ['', 'Step3.1 只保存注册模型与变换，不发布图片。',
                  'Step3.2 图片：[overview.png](step3.2/overview.png) 汇总本组全部 pose，按保存顺序排列并在旁边标注 pose 编号。各视图采用统一等轴测正交视角，以该姿态的原生摆放显示蓝色不透明支撑和灰色物体，工作面从真实开口露出。']
        readme.write_text('\n'.join(lines)+'\n')
    print('STEP3.2 OVERVIEW', group['id'], len(records), 'poses', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--scope', choices=['all', 'legal', 'illegal'], default='all')
    args = parser.parse_args()
    groups = []
    if args.scope in ('all', 'legal'):
        groups += read_sets(args.object)['sets']
    if args.scope in ('all', 'illegal'):
        groups += [dict(g, id='illegal/'+g['id']) for g in
                   json.loads((ROOT/'objects'/args.object/'illegal_pose_sets.json').read_text())['sets']]
    if args.sets:
        groups = [g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
        if not groups:
            parser.error('No matching pose sets')
    count = 0
    for group in groups:
        count += len(render(args.object, group))
    print('STEP3.2 TOTAL', len(groups), 'overviews', count, 'poses', flush=True)


if __name__ == '__main__':
    main()
