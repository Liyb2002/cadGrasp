"""English PNG figures of pose sets and native absolute-world pose overlays."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import trimesh

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / 'assets/data'
COOPT = HERE.parents[1]
ROOT = COOPT.parents[1]
COLORS = ['#D86456', '#C39231', '#398CA6', '#8A6CB4']
GREEN = '#65A779'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'axes.unicode_minus': False})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_renderer():
    source = COOPT / 'vis_func/cpu_mesh.py'
    spec = importlib.util.spec_from_file_location('representation_cpu_mesh', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(label, experiment):
    base = COOPT / 'data/results/B' / label
    initial_report = base / 'step4/step4.1/data/report.json'
    initial_layout = base / 'step4/step4.1/layout.npz'
    object_path = base / 'step3/step3.1/registered_object.obj'
    final = base / 'step4/step4.2' / experiment
    final_report = final / 'report.json'
    final_layout = final / 'layout.npz'
    report = json.loads(initial_report.read_text())
    accepted = json.loads(final_report.read_text())
    with np.load(initial_layout) as z:
        placements = z['placements'].copy()
        material_directions = z['directions'].copy()
        native = z['native_world'].copy()
    with np.load(final_layout) as z:
        guests = [int(k) for k in z['active'] if int(z['hosts'][k]) != int(k)
                  or not np.allclose(z['placements'][k], np.eye(4), atol=1e-10, rtol=0)]
    if len(report['poses']) != 4 or not np.allclose(placements, np.eye(4), atol=1e-12):
        raise ValueError('Select a four-task saved registered initial layout')
    if not accepted['passed'] or not guests or accepted['poses'] != report['poses']:
        raise ValueError('Select an actual certified four-pose result using Juxtapose')
    mesh = trimesh.load(object_path, force='mesh', process=False)
    work, worlds = [], []
    sources = [initial_report, initial_layout, object_path, final_report, final_layout]
    for k, pose in enumerate(report['poses']):
        snapshot = ROOT / 'objects/B/poses' / pose / 'setup.npz'
        needs_path = snapshot.with_name('needs.json')
        with np.load(snapshot) as z:
            ids = z['work_faces'].copy()
        if ids.dtype == bool:
            ids = np.flatnonzero(ids)
        needs = json.loads(needs_path.read_text())
        geometry = needs['geometry']
        transform = np.asarray(needs['frame']['T_world_mesh'])
        np.testing.assert_allclose(native[k], transform, atol=1e-12, rtol=0)
        world = mesh.copy()
        world.apply_transform(native[k])
        np.testing.assert_allclose(world.vertices, geometry['vertices_m'], atol=1e-12, rtol=0)
        if (not np.array_equal(mesh.faces, geometry['faces'])
                or not np.array_equal(np.sort(ids), np.sort(geometry['work_face_ids']))):
            raise ValueError('Working area does not match the original physical input')
        work.append(ids.astype(int))
        worlds.append(world)
        sources.extend([snapshot, needs_path])

    # Step4.1 makes exits common or close in material coordinates F, subject to
    # every task's floor hemisphere. Each task has its OWN rotation into W.
    actual_world_directions = np.einsum('kij,kj->ki', native[:, :3, :3], material_directions)
    np.testing.assert_allclose(material_directions, report['directions'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(actual_world_directions, axis=1), 1., atol=1e-12)
    angles = np.degrees(np.arccos(np.clip(material_directions @ material_directions.T, -1., 1.)))
    world_angles = np.degrees(np.arccos(np.clip(actual_world_directions @ actual_world_directions.T, -1., 1.)))
    if actual_world_directions[:, 2].min() < -1e-12:
        raise ValueError('Saved initial exit crosses an original floor hemisphere')
    sources.extend([COOPT/'helper_func/optimization/initial_directions.py',
                    COOPT/'helper_func/whole_search/model.py', COOPT/'helper_func/whole_pipeline.py',
                    COOPT/'step3.1/step31.py'])
    return dict(mesh=mesh, worlds=worlds, poses=report['poses'], work=work,
                directions=actual_world_directions, positions=native[:, :3, 3], native=native,
                material_directions=material_directions,
                angles=angles, world_angles=world_angles, guests=guests, sources=sources,
                saved_initialization_metadata=report['initialization'])


def draw_definition():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200, facecolor='white')
    fig.text(.075, .90, 'Pose-set representation', fontsize=26, color='#1D2934')
    fig.text(.5, .735, r'$\mathcal{S}=\{\pi_i\}_{i=1}^{n}$', fontsize=43,
             ha='center', va='center', color='#1D2934')
    fig.text(.5, .65, 'One set contains n task poses', fontsize=16,
             ha='center', color='#63717D')
    fig.text(.5, .515,
             r'$\pi_i=\left(\Theta_i,\;\mathcal{A}_i^{W},\;\hat{\mathbf{d}}_i^{W},\;\mathbf{p}_i^{W}\right)$',
             fontsize=32, ha='center', va='center', color='#1D2934')
    labels = [(.20, r'$\Theta_i\in SO(3)$', 'Pose', '#63717D'),
              (.40, r'$\mathcal{A}_i^W\subseteq\partial O_i^W$', 'Working surface', GREEN),
              (.62, r'$\hat{\mathbf{d}}_i^W\in\mathbb{S}^{2}$', 'World exit direction', '#398CA6'),
              (.83, r'$\mathbf{p}_i^W\in\mathbb{R}^{3}$', 'Absolute world position', '#8A6CB4')]
    for x, math, description, color in labels:
        fig.add_artist(plt.Line2D([x-.074, x+.074], [.395, .395], color=color, lw=3))
        fig.text(x, .333, math, fontsize=20, ha='center', color='#1D2934')
        fig.text(x, .265, description, fontsize=14, ha='center', color=color)
    fig.text(.5, .16,
             r'$\mathbf{x}^{W}=\Theta_i\mathbf{x}^{O}+\mathbf{p}_i^{W}'
             r'\qquad |\mathcal{A}_i^W|=\int_{\mathcal{A}_i^W}dA$',
             fontsize=19, ha='center', color='#425466')
    fig.text(.5, .068, 'Every pose has its own absolute transform in the same world frame W.',
             fontsize=15, ha='center', color='#1D2934')
    fig.savefig(HERE / 'definition.png', facecolor='white')
    plt.close(fig)


def body_piece(module, mesh, working, color):
    colors = np.broadcast_to(np.asarray(module.rgb(color)), (len(mesh.faces), 3, 3)).copy()
    colors[np.asarray(working, int)] = module.rgb(GREEN)
    return module.piece(dict(v=mesh.vertices.ravel(), f=mesh.faces.ravel()), colors)


def project(point, camera, width, height):
    focus, basis, span = camera
    q = (np.asarray(point)-focus) @ basis.T
    return np.array([width/2+q[0]*height/span, height/2-q[1]*height/span])


def ghost_overlay(images, module, background):
    canvas = background.convert('RGBA')
    for k, image in enumerate(images):
        pixels = np.asarray(image)
        mask = np.min(pixels, axis=2) < 250
        layer = np.dstack([pixels, np.uint8(mask)*78])
        canvas = Image.alpha_composite(canvas, Image.fromarray(layer))
        silhouette = Image.fromarray(np.uint8(mask)*255)
        edge = np.asarray(silhouette.filter(ImageFilter.MaxFilter(3))).astype(int)
        edge -= np.asarray(silhouette.filter(ImageFilter.MinFilter(3))).astype(int)
        outline = np.zeros((*mask.shape, 4), dtype=np.uint8)
        outline[:, :, :3] = module.rgb(COLORS[k]).astype(np.uint8)
        outline[:, :, 3] = np.uint8(edge > 0)*145
        canvas = Image.alpha_composite(canvas, Image.fromarray(outline))
    return canvas.convert('RGB')


def draw_arrows(ax, data, camera, width, height, length, indices, labels=True):
    # Exit directions are free vectors; each arrow is drawn at its object's COM.
    offsets = [(-45, -18), (-45, 5), (45, -15), (40, 12)]
    for k in indices:
        origin = data['worlds'][k].center_mass
        start = project(origin, camera, width, height)
        end = project(origin+length*data['directions'][k], camera, width, height)
        annotation = ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='-|>', color=COLORS[k],
                                                lw=2.4, mutation_scale=18))
        annotation.arrow_patch.set_path_effects([
            patheffects.Stroke(linewidth=4.2, foreground='white', alpha=.8),
            patheffects.Normal()])
        ax.plot(*start, 'o', color=COLORS[k], markersize=4,
                markeredgecolor='white', markeredgewidth=.6)
        if labels:
            ax.annotate(data['poses'][k].replace('_', ' '), xy=end,
                        xytext=offsets[k], textcoords='offset points',
                        ha='center', fontsize=13, color=COLORS[k],
                        bbox=dict(boxstyle='round,pad=.25', facecolor='white', edgecolor='none', alpha=.9),
                        arrowprops=dict(arrowstyle='-', color=COLORS[k], lw=.9))


def floor_piece(module, vertices):
    part = module.piece(dict(v=vertices, f=np.array([[0, 1, 2], [0, 2, 3]])), '#EDF0F3')
    part['unlit'] = True
    return module.placed(part)


def floor_background(module, raster, vertices, camera, width, height):
    image = raster.render([floor_piece(module, vertices)], camera, width, height, scale=2.)
    corners = [tuple(project(p, camera, width, height)) for p in vertices]
    ImageDraw.Draw(image).line(corners+[corners[0]], fill='#CCD4DC', width=2)
    return image


def draw_relative_exits(fig, data):
    """Orthographic fan of the actual saved exits in their common material F."""
    directions = data['material_directions']
    mean = directions.sum(axis=0)
    mean /= np.linalg.norm(mean)
    residual = directions-(directions@mean)[:, None]*mean
    _, _, axes = np.linalg.svd(residual, full_matrices=False)
    tangent = axes[0]
    projected = np.c_[directions@mean, directions@tangent]
    ax = fig.add_axes([.65, .075, .14, .115])
    ax.set_xlim(-.06, 1.1)
    ax.set_ylim(-.40, .40)
    ax.set_aspect('equal')
    ax.axis('off')
    for k, end in enumerate(projected):
        ax.annotate('', xy=end, xytext=(0., 0.),
                    arrowprops=dict(arrowstyle='-|>', color=COLORS[k], lw=1.8, mutation_scale=12))
    pairs = data['angles'][np.triu_indices(4, 1)]
    fig.text(.65, .202, 'Similar relative exits in material frame', fontsize=15, color='#1D2934')
    fig.text(.80, .14, f'{pairs.min():.2f}'+chr(176)+' - '+f'{pairs.max():.2f}'+chr(176)+' apart',
             fontsize=13, color='#63717D')
    fig.text(.80, .10, 'Encourage overlapping cuts', fontsize=12, color='#63717D')
    return dict(projection='orthographic onto mean-exit and first tangent principal direction',
                horizontal_basis_material=mean.tolist(), vertical_basis_material=tangent.tolist(),
                material_direction_projection=projected.tolist())


def draw_initialization(data, label):
    module = load_renderer()
    raster = module.Renderer()
    points = np.vstack([mesh.vertices for mesh in data['worlds']])
    extent = float(np.ptp(points, axis=0).max())
    low, high = points[:, :2].min(0)-extent*.16, points[:, :2].max(0)+extent*.16
    floor = np.array([[low[0], low[1], 0.], [high[0], low[1], 0.],
                      [high[0], high[1], 0.], [low[0], high[1], 0.]])
    length = extent*.90
    tips = np.array([mesh.center_mass+length*d for mesh, d in zip(data['worlds'], data['directions'])])
    view = np.array([1., -1., .80])
    view /= np.linalg.norm(view)
    width, height = 1760, 1370
    camera = module.fit(np.vstack([points, tips, floor]), view, width/height, padding=1.24)
    images = []
    for k, mesh in enumerate(data['worlds']):
        color = np.asarray(module.rgb(COLORS[k]))*.42+np.array([180., 187., 192.])*.58
        parts = [module.placed(body_piece(module, mesh, data['work'][k], color/255.))]
        image = raster.render(parts, camera, width, height, scale=2.)
        images.append(image)
    background = floor_background(module, raster, floor, camera, width, height)
    combined = ghost_overlay(images, module, background)
    combined.save(DATA / 'combined_initialization.png')

    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='white')
    fig.text(.035, .944, 'Step4.1 initialization', fontsize=27, color='#1D2934')
    fig.text(.035, .891, 'B / '+label+'  |  Original absolute poses and saved world exits',
             fontsize=16, color='#63717D')
    ax = fig.add_axes([.025, .15, .59, .70])
    ax.imshow(combined)
    ax.axis('off')
    draw_arrows(ax, data, camera, width, height, length, range(4))
    fig.text(.66, .824, 'Each pose keeps its own world transform', fontsize=15, color='#1D2934')
    body_camera = module.fit(np.vstack([points, floor]), view, 1.28, padding=1.16)
    for k, pose in enumerate(data['poses']):
        x, y = .655+.167*(k%2), .515-.275*(k//2)
        tile_ax = fig.add_axes([x, y, .152, .24])
        mesh = data['worlds'][k]
        parts = [module.placed(body_piece(module, mesh, data['work'][k], '#BDC4CA'))]
        tile = raster.render(parts, body_camera, 640, 500, scale=2.)
        background = floor_background(module, raster, floor, body_camera, 640, 500)
        pixels = np.asarray(tile)
        layer = np.dstack([pixels, np.uint8(np.min(pixels, axis=2) < 250)*255])
        tile = Image.alpha_composite(background.convert('RGBA'), Image.fromarray(layer)).convert('RGB')
        tile.save(DATA / (pose+'_initialization.png'))
        tile_ax.imshow(tile)
        tile_ax.axis('off')
        tile_ax.set_title(pose.replace('_', ' '), fontsize=14, color=COLORS[k], pad=2)
    relative_fan = draw_relative_exits(fig, data)
    fig.text(.035, .097, r'$\hat{\mathbf{d}}_i^F=\Theta_i^{\mathsf{T}}\hat{\mathbf{d}}_i^W$',
             fontsize=21, color='#1D2934')
    fig.text(.035, .041,
             'Green: working surfaces. Arrows: saved Step4.1 exits. Similarity is measured relative to the object, in material frame F.',
             fontsize=12, color='#63717D')
    fig.savefig(HERE / 'initialization.png', facecolor='white')
    plt.close(fig)
    return dict(camera_direction_world=view.tolist(), orthographic=True,
                world_transforms_preserved_per_pose=True, coincident_registration_used=False,
                translucent_pose_layers=True, pose_layer_alpha=78/255,
                world_coordinate_axes_drawn=False, per_pose_position_labels_drawn=False,
                floor_drawn_in_overlay_and_individual_views=True,
                floor_z_m=0., floor_vertices_world_m=floor.tolist(),
                arrow_display_length_m=length, arrow_origins='each object world COM',
                world_arrows_use_saved_step41_execution_directions=True,
                relative_exit_fan=relative_fan,
                similarity_frame='common material frame F; not world W',
                arbitrary_world_exit_seeds_used=False,
                new_physical_exit_acceptance_run=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', default='pose1+2+3+27')
    parser.add_argument('--experiment', default='pso_accelerated_all52_step4_20261014_01')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    data = read_inputs(args.set, args.experiment)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in data['sources']}
    draw_definition()
    rendering = draw_initialization(data, args.set)
    changed = [p for p, h in protected.items() if digest(ROOT/p) != h]
    if changed:
        raise ValueError('Saved inputs changed: '+str(changed))
    for path in HERE.rglob('*.svg'):
        path.unlink()
    record = dict(complete=True, object='B', pose_set=args.set, poses=data['poses'],
                  language='English', output_format='PNG',
                  tuple_order=['pose', 'world_working_surface',
                               'absolute_world_exit_direction', 'absolute_world_position'],
                  frame='original shared world W from each task input; no registration of poses',
                  absolute_world_transforms=data['native'].tolist(),
                  absolute_world_positions_m=data['positions'].tolist(),
                  pose_symbol='Theta_i', absolute_world_position_symbol='p_i^W',
                  pose_contains_translation=False,
                  pose_matrices=data['native'][:, :3, :3].tolist(),
                  initialization_world_exit_directions=data['directions'].tolist(),
                  exit_seed_policy='actual saved Step4.1 directions, transformed by each own original task rotation',
                  initialization_exits_are_saved_step41_exits=True,
                  saved_step41_common_material_directions=data['material_directions'].tolist(),
                  relative_similarity_frame='registered common material frame F',
                  direction_conversion='d_i^W = Theta_i d_i^F; d_i^F = transpose(Theta_i) d_i^W',
                  material_pairwise_exit_angles_degrees=data['angles'].tolist(),
                  world_pairwise_exit_angles_degrees=data['world_angles'].tolist(),
                  saved_initialization_metadata=data['saved_initialization_metadata'],
                  initialization_objective='angular dispersion sum_i(1-dot(d_i^F,reference^F)), with unit vectors and native floor hemispheres',
                  shared_exit_cut_overlap_is_design_intent=True,
                  direct_cut_volume_minimization_in_initializer=False,
                  global_minimum_cut_volume_claimed=False,
                  work_face_ids=[ids.tolist() for ids in data['work']],
                  working_areas_m2=[float(data['mesh'].area_faces[a].sum()) for a in data['work']],
                  selected_final_experiment=args.experiment, saved_final_solution_certified=True,
                  saved_final_juxtaposed_poses=[data['poses'][k] for k in data['guests']],
                  saved_final_juxtaposed_count=len(data['guests']),
                  global_juxtapose_necessity_claimed=False,
                  physical_search_run=False, force_solver_run=False, initialization_rerun=False,
                  protected_inputs=protected, saved_inputs_unchanged=True, rendering=rendering,
                  outputs={name:digest(HERE/name) for name in ['definition.png','initialization.png']},
                  rendering_source_sha256=digest(Path(__file__)))
    (DATA/'manifest.json').write_text(json.dumps(record, ensure_ascii=True, indent=2)+'\n')
    print(json.dumps(dict(complete=True, poses=data['poses'],
                         saved_final_juxtaposed_count=len(data['guests']),
                         world_positions_m=data['positions'].tolist(),
                         material_exit_angle_range_deg=[float(data['angles'][np.triu_indices(4,1)].min()),
                                                       float(data['angles'][np.triu_indices(4,1)].max())],
                         saved_inputs_unchanged=True), indent=2), flush=True)


if __name__ == '__main__':
    main()
