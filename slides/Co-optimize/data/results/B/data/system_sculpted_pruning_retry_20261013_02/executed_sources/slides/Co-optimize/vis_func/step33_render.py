"""Text-free Step3.3 overviews using the Step3.2 wrap and saved ring recipes."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import argparse
from io import BytesIO
from PIL import Image, ImageOps


RING_COLORS = ['#319cd7'] * 6

def render_pose(name, group, pose, base, rings, ring_colors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = base / 'step3.3'
    support_path = initialized_wrap(base)
    object_path = base / 'step3.1/registered_object.obj'
    task, transform, _ = state(name, pose)
    support = trimesh.load(support_path, force='mesh', process=False)
    obj = trimesh.load(object_path, force='mesh', process=False)
    support.apply_transform(transform)
    world_rings = [ring.copy() for ring in rings]
    for ring in world_rings:
        ring.apply_transform(transform)
    obj.apply_transform(transform)
    np.testing.assert_allclose(obj.vertices, task.domain.mesh.vertices, atol=1e-10, rtol=0)

    # Fixed world-space isometric camera: Z stays up and the native floor
    # stays horizontal, independent of the pose's working-face direction.
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
    triangles = np.concatenate([obj.triangles, support.triangles]+[ring.triangles for ring in world_rings])*1000
    facecolors = np.concatenate([colors(obj, '#a4a8ac'), colors(support, '#319cd7')]+[colors(ring, color) for ring, color in zip(world_rings, ring_colors)])
    fig = plt.figure(figsize=(9, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=facecolors,
                                        edgecolors='none', zsort='average'))
    points = np.vstack([obj.vertices, support.vertices]+[ring.vertices for ring in world_rings])*1000
    center = (points.min(0)+points.max(0))/2
    radius = float(np.ptp(points, axis=0).max())*.53
    for axis, value in zip('xyz', center):
        getattr(ax, 'set_'+axis+'lim')(value-radius, value+radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elevation, azim=azimuth, roll=0)
    xy = points[:,:2]
    lo, hi = xy.min(0)-8, xy.max(0)+8
    boundary = np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
    ax.plot(*boundary.T, color='#c7c7c7', linewidth=.7)
    saved_points = np.load(out/'data'/f'{pose}.npz')['saved_demands_world_xy_m']*1000
    index = group['poses'].index(pose)
    ax.scatter(saved_points[:,0], saved_points[:,1], np.zeros(len(saved_points)),
               s=.1, alpha=.12, color=ring_colors[index], rasterized=True)
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
                  support_source=str(initialized_wrap(base).relative_to(base)),
                  support_color='blue', object_color='gray', opacity=1.,
                  working_faces_exposed_by_actual_support_openings=True,
                  view_elevation_deg=elevation, view_azimuth_deg=azimuth,
                  projection='orthographic', camera_frame='native world', view_kind='isometric',
                  geometry_changed=False, mechanics_rerun=False,
                  provenance=provenance(inputs, [Path(__file__), HERE/'step3.3/step33.py']))
    return tile, record


def draw_group(name, group, base, rings, rows):
    out = base/'step3.3'
    palette = RING_COLORS[:len(group['poses'])]
    panels = [render_pose(name, group, pose, base, rings, palette) for pose in group['poses']]
    cols = 2 if len(panels)==4 else min(3,len(panels))
    nr = (len(panels)+cols-1)//cols
    size = 760
    overview = Image.new('RGB',(cols*size,nr*size),'white')
    for index,(tile,record) in enumerate(panels):
        tile = ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS)
        overview.paste(tile,((index%cols)*size+(size-tile.width)//2,(index//cols)*size+(size-tile.height)//2))
    overview.save(out/'overview.png')
    for extra in out.glob('*.png'):
        if extra.name!='overview.png':extra.unlink()
    inputs=[initialized_wrap(base),base/'step3.1/registered_object.obj',out/'support_with_rings.obj']
    inputs += [out/'data'/f'{pose}.npz' for pose in group['poses']]
    inputs += [out/'data'/f'{pose}_perimeter.obj' for pose in group['poses']]
    save(out/'data/render.json',dict(complete=True,presentation_only=True,poses=group['poses'],
         pose_set=group['id'],states=[r for tile,r in panels],text_or_labels=False,
         shell_source=str(initialized_wrap(base).relative_to(base)),object_color='gray',shell_color='blue',opacity=1,
         boundary_kind='expanded_convex_hull',ring_count=len(rings),rings=[dict(pose=pose,color=color,recipe=row) for pose,color,row in zip(group['poses'],palette,rows)],
         same_ring_colors_in_all_panels=True,geometry_changed=False,mechanics_rerun=False,
         image_size_px=list(overview.size),provenance=provenance(inputs,[Path(__file__),HERE/'step3.3/step33.py']),
         artifacts={'../overview.png':I.sha256(out/'overview.png')}))
    I.check_report(out/'data/render.json')
    print('STEP3.3 OVERVIEW',group['id'],len(rings),'distinct rings',flush=True)


def render(name,group):
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3';out=base/'step3.3'
    report=json.loads((out/'data/report.json').read_text());rows=report['state_results']
    assert [row['pose'] for row in rows]==group['poses']
    rings=[]
    for row in rows:
        if row.get('boundary_kind') not in ('minimum_convex_hull','expanded_convex_hull'):
            raise RuntimeError('Step3.3 still uses circles; regenerate with step33.py first')
        rings.append(trimesh.load(out/'data'/f"{row['pose']}_perimeter.obj",force='mesh',process=False))
    draw_group(name,group,base,rings,rows)
    readme=out/'README.md'
    text=readme.read_text().replace('灰色是壳子，圈的颜色标识来源 pose。','灰色不透明壳子直接使用 Step3.2 的包裹体，蓝色物体从工作面开口露出；每个 pose 一个接地圈，各圈颜色不同，并在全部视图中保持一致。图内没有标题、文字或图例。')
    text = text.replace('灰色不透明壳子', '蓝色不透明壳子').replace('蓝色物体', '灰色物体').replace('各圈颜色不同', '所有围边统一为支撑的蓝色')
    text = text.replace('灰色是壳子，围边的颜色标识来源 pose。', '灰色是物体，蓝色是壳子与全部凸包围边；各视图使用统一正交等轴测视角，图内无文字。')
    readme.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sets',nargs='+');parser.add_argument('--scope',choices=['all','legal','illegal'],default='all');args=parser.parse_args()
    groups=[]
    if args.scope in ('all','legal'):groups+=read_sets('B')['sets']
    if args.scope in ('all','illegal'):groups+=[dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    for group in groups:render('B',group)
    print('STEP3.3 TOTAL',len(groups),'overviews',flush=True)

if __name__=='__main__':main()
