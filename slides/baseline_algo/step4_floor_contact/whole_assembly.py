"""Whole-assembly floor demand from current Step 1 loads, independent of heads.

The mapped tangent-cap vertices conservatively enclose continuous reachable
loads. Only pressure-center samples and conservative continuous enclosure points are output.
Thickness, connections and installation are deferred to Step 5.
"""
from step1.needs import COORD
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from step1.needs import ContinuousNeeds, OUTPUTS, ROOT, OBJECTS, demand, sha256
from step1.cases import pose_name
from step3_scheculer import contacts as I, enclosure as E

STAGE = 'step4_floor_contact'
SCHEMA = 'whole_assembly_floor_demand_points_v3'


def output_folder(name):
    return OUTPUTS/name/pose_name()/STAGE


def pressure_centers(loads, origin):
    """Required ground wrenches about COM -> ground CoP in world coordinates."""
    loads = np.asarray(loads, float).reshape(-1, 6)
    normal = loads[:, 1]
    if not np.isfinite(loads).all() or np.any(normal <= 0):
        raise ValueError('Finite floor demand requires strictly positive normal reaction')
    moments = loads[:, 3:]+np.cross(origin, loads[:, :3])
    return np.c_[moments[:, 2], -moments[:, 0]]/normal[:, None], normal


def outer_loads(domain):
    # Step3 appends a solver-only no-uplift coordinate. Floor demands are the
    # physical force and moment; the auxiliary equality is not an external load.
    return E.targets(SimpleNamespace(domain=domain, scale=np.ones(6)), sides=8, bands=1)[:, :6]


def search_load_paths(name):
    return sorted((OUTPUTS/name/pose_name()/'step3_scheculer/search_loads').glob('after_round_*.json'))


def inputs(name):
    root = OUTPUTS/name/pose_name()
    source = root/'step_1_needs/needs.json'
    domain = ContinuousNeeds.read(source)
    samples = json.loads((source.parent/'samples.json').read_text())
    assert samples['provenance']['physical_domain_sha256'] == sha256(source)
    sampled = np.asarray(samples['need_wrench'])
    np.testing.assert_allclose(sampled, demand(samples['pt_m'], samples['force_push_mg'], domain.com, domain.gravity), atol=1e-13, rtol=0)
    supplemental = []
    extra_paths = search_load_paths(name)
    for path in extra_paths:
        for case in json.loads(path.read_text())['counterexamples']:
            value = domain.evaluate(case['work_face_index'], case['u'], case['v'],
                case['theta_rad'], case['phi_rad'], magnitude_mg=case['magnitude_mg'])
            assert value['reachable']
            np.testing.assert_allclose(value['need_wrench'], case['need_wrench'], atol=1e-13, rtol=0)
            supplemental.append(value['need_wrench'])
    loads = np.vstack([np.r_[-domain.gravity, [0., 0., 0.]], sampled,
                       np.asarray(supplemental).reshape(-1, 6)])
    snapshot = ROOT/domain.data['provenance']['setup_snapshot']
    with np.load(snapshot) as z:
        pivot = z['floor_contact_m'].copy()
    paths = [source, source.parent/'samples.json', snapshot]+extra_paths
    return domain, loads, pivot, samples, paths


def build(name):
    domain, loads, pivot, samples, paths = inputs(name)
    out = output_folder(name); out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(complete=False, status='computing_whole_assembly_floor_demand'))
    cloud, normal = pressure_centers(loads, domain.com)
    outer = outer_loads(domain)
    enclosure, outer_normal = pressure_centers(outer, domain.com)
    # Positive-denominator linear-fractional maps preserve convex containment:
    # p(sum a_i w_i) = sum (a_i N_i / sum a_i N_i) p(w_i).
    arrays = dict(load_wrenches=loads, floor_demands_xz_m=cloud,
        total_floor_normal_mg=normal,
        continuous_outer_load_wrenches=outer, continuous_floor_enclosure_xz_m=enclosure,
        continuous_outer_normal_mg=outer_normal,
        original_pivot_m=pivot, moment_origin_m=domain.com)
    np.savez_compressed(out/'floor_contact.npz', **arrays)
    report = dict(object=name, pose=pose_name(), stage=STAGE, schema=SCHEMA, complete=True,
        status='floor_demand_points_ready', sample_count=samples['count'], load_count=len(loads),
        zero_process_force_included=True, support_body_count=1, support_mass_ignored=True,
        ground_footprints=[], shape_designed=False, connections_constructed=False,
        trajectory_selected=False, foot_positions_frozen_for_step5=False,
        connectors_may_move_feet=True, continuous_domain_coverage_proved=False,
        continuous_demand_enclosure_proved=True, ground_bearing_equilibrium_verified=False,
        physical_supports_verified=False, arrays_file='floor_contact.npz',
        floor_polygon_designed=False, original_pivot_m=pivot.tolist(),
        base_design_stage=5,
        minimum_floor_normal_mg=float(normal.min()),
        supplemental_load_sources=[str(p.relative_to(ROOT)) for p in search_load_paths(name)],
        continuous_enclosure=dict(method='work_triangle_vertices_times_tangent_cap_outer_polytopes',
            sides=8, bands=1, includes_zero_force=True, includes_occluded_directions=True,
            outer_vertex_count=len(outer), minimum_outer_normal_mg=float(outer_normal.min()),
            cap_padding=E.CAP_PADDING, mapped_outer_points_are_physical_samples=False,
            argument='Positive total normal force makes each mapped convex combination a positive weighted combination of vertex pressure centers.'),
        ground_rule='required_hull inside convex hull of actual support-floor contact plus original object-floor contact',
        scope='Floor pressure-center samples and conservative continuous enclosure points only. Step5 constructs the demand hull, actual footprint, belt, connectors and trajectory and verifies bearing.',
        provenance=dict(inputs=I.hashes(paths), code=I.hashes([Path(__file__), Path(E.__file__)])),
        artifacts={'floor_contact.npz': sha256(out/'floor_contact.npz')})
    I.save(out/'floor_contact.json', report)
    I.save(out/'status.json', dict(complete=True, status=report['status'], floor_contact_sha256=sha256(out/'floor_contact.json')))
    print(name, pose_name(), 'Step 4:', len(cloud), 'physical loads;', len(enclosure), 'continuous enclosure points', flush=True)
    return report


def read(name):
    out = output_folder(name)
    status = json.loads((out/'status.json').read_text())
    assert status['complete'] and status['floor_contact_sha256'] == sha256(out/'floor_contact.json')
    report = I.check_report(out/'floor_contact.json')
    if report['schema'] != SCHEMA:
        raise RuntimeError('Rebuild Step 4 for one connected rigid support')
    assert report['supplemental_load_sources'] == [str(p.relative_to(ROOT)) for p in search_load_paths(name)]
    return report


def audit(name):
    report = read(name); out = output_folder(name)
    arrays = I.load_npz(out/report['arrays_file'])
    domain, loads, pivot, samples, _ = inputs(name)
    np.testing.assert_array_equal(arrays['load_wrenches'], loads)
    np.testing.assert_array_equal(arrays['original_pivot_m'], pivot)
    np.testing.assert_array_equal(arrays['continuous_outer_load_wrenches'], outer_loads(domain))
    # Independent external-load formula, including q_y * F_horizontal.
    f = np.asarray(samples['force_push_mg']); q = np.asarray(samples['pt_m'])
    expected = (COORD.floor(domain.com)-COORD.floor(q)*f[:, 1, None]+q[:, 1, None]*COORD.floor(f))/(1-f[:, 1, None])
    np.testing.assert_allclose(arrays['floor_demands_xz_m'][1:1+len(f)], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(arrays['floor_demands_xz_m'][0], COORD.floor(domain.com), atol=1e-14)
    for load_key, point_key, normal_key in [
        ('load_wrenches', 'floor_demands_xz_m', 'total_floor_normal_mg'),
        ('continuous_outer_load_wrenches', 'continuous_floor_enclosure_xz_m', 'continuous_outer_normal_mg')]:
        w = arrays[load_key]; p = arrays[point_key]; n = arrays[normal_key]
        np.testing.assert_array_equal(n, w[:, 1]); assert np.all(n > 0)
        floor_moment = np.cross(COORD.lift_floor(p), COORD.lift_floor(np.zeros((len(p), 2)), n))
        about_world = w[:, 3:]+np.cross(domain.com, w[:, :3])
        np.testing.assert_allclose(COORD.floor(floor_moment), COORD.floor(about_world), atol=1e-12, rtol=1e-12)
    assert not report['floor_polygon_designed']
    assert not any(key in arrays for key in ('support_polygon_xz_m', 'required_hull_xz_m', 'support_boundary_closed_m'))
    result = dict(object=name, pose=pose_name(), complete=True, passed=True,
        demand_formula_independently_checked=True, continuous_outer_domain_rebuilt=True,
        demand_points_only_verified=True, floor_shape_deferred_to_step5=True,
        actual_feet_not_yet_constructed=True, bearing_success_claimed=False,
        provenance=dict(inputs=I.hashes([out/'floor_contact.json', out/'floor_contact.npz']), code=I.hashes([Path(__file__)])))
    I.save(out/'audit.json', result)
    print(name, pose_name(), 'Step 4 demand audit passed', flush=True)
    return result


def draw(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from step5_connect_support import visual_details as V
    report = read(name); out = output_folder(name)
    a = I.load_npz(out/'floor_contact.npz')
    domain, _, _, _, _ = inputs(name)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
    from matplotlib.collections import PolyCollection
    # Draw the actual projected triangles, without constructing any convex hull.
    ax.add_collection(PolyCollection(COORD.floor(domain.mesh.triangles)*1000,
                                    facecolors='#e3e4df', edgecolors='none'))
    ax.autoscale_view()
    p = a['floor_demands_xz_m']*1000
    ax.scatter(*p.T, s=2, alpha=.23, c='#bd8236', rasterized=True)
    outer = a['continuous_floor_enclosure_xz_m']*1000
    ax.scatter(*outer.T, s=3, alpha=.3, c='#298b93', rasterized=True)
    ax.scatter(*(COORD.floor(a['original_pivot_m'])*1000), marker='x', s=55, c='#33424a')
    ax.set_aspect('equal'); ax.set_axis_off(); ax.margins(.12)
    fig.subplots_adjust(.025, .025, .975, .975)
    fig.savefig(out/'floor_contact.png'); plt.close(fig)
    floor_points = COORD.lift_floor(a['floor_demands_xz_m'])
    points = np.vstack([domain.mesh.vertices, floor_points])
    view = V.camera(domain, [], points=points)
    body = V.scene(domain, [], [], 1100, view=view, xray=True, labels=False)
    focus, width, basis = view
    projected = V.R.project(floor_points[::max(1,len(floor_points)//3000)], focus, basis, width, 1100)
    ink_body = ImageDraw.Draw(body)
    for x,y,*_ in projected: ink_body.ellipse((x-1,y-1,x+1,y+1), fill='#bd8236')
    body.save(out/'floor_contact_object.png')
    page = Image.new('RGB', (2200, 1260), V.R.PAPER); ink = ImageDraw.Draw(page)
    ink.text((30, 20), f'{name} / {pose_name()} / Floor demand points only', font=V.R.font(38), fill=V.R.INK)
    page.paste(body, (0, 85))
    page.paste(Image.open(out/'floor_contact.png').resize((1100, 1100)), (1100, 85))
    ink.text((30, 1180), 'Orange: physical load landings. Teal: conservative continuous enclosure points.', font=V.R.font(26), fill=V.MUTED)
    ink.text((30, 1215), 'No support boundary is designed here. All support geometry and installation belong to Step5.', font=V.R.font(23), fill=V.MUTED)
    page.save(out/'floor_diagnostic.png')
    images = ['floor_contact.png', 'floor_contact_object.png', 'floor_diagnostic.png']
    I.save(out/'views.json', dict(complete=True, whole_assembly_demand_cloud_drawn=True,
        closed_floor_polygon_drawn=False, floor_polygon_is_solid_plate=False,
        actual_feet_drawn=False, continuous_enclosure_separate_from_samples=True,
        contact_interfaces_drawn=False, contact_labels_drawn=False,
        provenance=dict(inputs=I.hashes([out/'floor_contact.json', out/'floor_contact.npz']),
                        code=I.hashes([Path(__file__), Path(V.__file__), Path(V.R.__file__)])),
        artifacts={p: sha256(out/p) for p in images}))
