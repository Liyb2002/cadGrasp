"""Describe the sampled demand input and draw its physical mapping.

Run from the repository root in the cadgrasp environment:
    python slides/baseline_algo/step1/domain.py

Writes output/<object>/<pose>/step_1_needs/domain.json and domain.png. Each JSON references
the same object's samples.json and physical needs.json by relative path and hash.
This script does not integrate coverage or generate additional load samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from needs import ContinuousNeeds, OBJECTS, OUTPUTS, ROOT, sha256
from step1.cases import pose_name

INK, MUTED, PAPER = '#1b1b1a', '#6b6b66', '#ffffff'


def specification(object_name):
    folder = OUTPUTS / object_name/pose_name() / 'step_1_needs'
    samples_path = folder / 'samples.json'
    samples = json.loads(samples_path.read_text())
    if samples['provenance']['physical_domain_sha256'] != sha256(folder / 'needs.json'):
        raise ValueError('Samples refer to a different physical domain; regenerate samples.json')
    path = folder / 'needs.json'
    data = json.loads(path.read_text())
    first = data['load']
    if (samples['object'] != object_name or data['object'] != object_name
            or first['gravity_force_mg'] != [0., 0., -1.]
            or first.get('magnitude_range_mg') != [0., first['K']]):
        raise ValueError('Object identity or physical load range does not match the sampled domain')
    alpha = float(np.deg2rad(first['cone_half_deg']))
    objects = {
        object_name: {
            'source_file': 'needs.json',
            'source_sha256': sha256(path),
            'work_face_count': len(data['geometry']['work_face_ids']),
            'work_area_m2': data['geometry']['work_area_m2'],
            'position_direction_measure_upper_bound_m2_sr': float(
                data['geometry']['work_area_m2'] * 2*np.pi*(1-np.cos(alpha))),
            'measure_role': 'Physical position/direction size only; sample count is the scoring denominator',
            'reachability': data['reachability'],
        }
    }
    return {
        'schema_version': 3,
        'object': object_name,
        'representation': 'sampled_demand_approximation_with_physical_domain_reference',
        'scope': 'Finite paired demands drawn from continuous physical positions, directions and force magnitudes at setup tip 1',
        'sample_set': {
            'source_file': 'samples.json', 'source_sha256': sha256(samples_path),
            'count': samples['count'], 'seed': samples['seed'],
            'method': samples['method'], 'weight_per_sample': samples['weight_per_sample'],
            'sampled_100_percent_proves_continuous_coverage': False,
        },
        'geometry_references_required': True,
        'objects': objects,
        'geometry_bindings': {
            'f': 'each mesh face ID in source.geometry.work_face_ids',
            'a_f_b_f_h_f': 'source.geometry.vertices_m indexed by source.geometry.faces[f]',
            'n_f_e1_f_e2_f': 'inward_normals, tangent1, tangent2 at the corresponding work_face_index',
            'A_f': 'source.geometry.work_face_areas_m2 at the corresponding work_face_index',
            'c': 'source.frame.moment_origin_m',
            'occluder': 'entire source.geometry mesh, including non-work faces',
        },
        'load': {'magnitude_K': first['K'], 'magnitude_varies': True,
                 'magnitude_range_mg': [0., first['K']],
                 'gravity_force_mg': [0., 0., -1.], 'cone_half_deg': first['cone_half_deg']},
        'parameter_domain': {
            'coordinates': ['u', 'v', 'theta', 'phi', 'magnitude_mg'],
            'dimension': 5,
            'surface_triangle': {
                'variables': ['u', 'v'],
                'inequality_convention': 'matrix @ [u,v] <= upper',
                'matrix': [[-1., 0.], [0., -1.], [1., 1.]],
                'upper': [0., 0., 1.],
                'nested_intervals': ['0 <= u <= 1', '0 <= v <= 1-u'],
                'boundary_included': True,
            },
            'direction_intervals_rad': {
                'theta': {'lower': 0., 'upper': alpha, 'lower_closed': True, 'upper_closed': True},
                'phi': {'lower': 0., 'upper': float(2*np.pi),
                        'lower_closed': True, 'upper_closed': False, 'periodic': True},
            },
            'magnitude_interval_mg': {'lower': 0., 'upper': first['K'],
                                      'lower_closed': True, 'upper_closed': True},
            'pt_formula': 'a_f + u*(b_f-a_f) + v*(h_f-a_f)',
            'd_formula': 'cos(theta)*n_f + sin(theta)*(cos(phi)*e1_f + sin(phi)*e2_f)',
            'F_push_formula': 'magnitude_mg*d_f(x)',
            'visibility': 'V_f(x) = 1 iff the object-specific reachability predicate holds at pt_f(x), d_f(x)',
            'U_f': '{x=(u,v,theta,phi,magnitude_mg) satisfying the triangle, angular and magnitude intervals, and V_f(x)=1}',
            'U': 'tagged union of {f} x U_f over every work face f',
            'boundary_convention': 'union of face families; shared edges and angular poles have zero integration measure',
        },
        'demand_image': {
            'pair_order': ['Fx', 'Fy', 'Fz', 'tau_x', 'tau_y', 'tau_z'],
            'force_formula': '-([0,0,-1] + F_push)',
            'moment_formula': '-cross(pt_f(x)-c,F_push)',
            'D': 'union_f {Psi_f(x) : x in U_f}',
            'ambient_dimension': 6,
            'image_dimension_at_most': 5,
            'force_unit': 'mg', 'moment_unit': 'mg*m',
            'representation_is_componentwise_box': False,
            'pairing': 'The same pt and complete F_push determine all six components together',
        },
        'measure': {
            'status': 'estimated_by_equal_weight_physical_load_samples',
            'space': 'physical load parameter domain U, with face labels',
            'unit': 'm^2 sr mg; the estimated coverage ratio is dimensionless',
            'position_jacobian': '2*A_f',
            'direction_jacobian': 'sin(theta)',
            'magnitude_weight': 'uniform on [0,K]',
            'density': '2*A_f*sin(theta); defines the sampling distribution, no polygon integration is performed',
            'differentials': ['du', 'dv', 'dtheta', 'dphi', 'dmagnitude_mg'],
            'denominator': 'Number of reachable samples',
            'repeated_wrenches': 'Different physical loads mapping to the same wrench retain their load-domain weight',
            'six_dimensional_lebesgue_volume': False,
        },
        'step3_coverage': {
            'status': 'sampled_scoring_implemented_in_step3',
            'design': 'one fixed contact design S, including the original object-floor contact',
            'supply_cone': 'K_S = {G_S*lambda : lambda >= 0}; lambda may depend on the load',
            'feasible_subdomain': 'E_S = {(f,x) in U : Psi_f(x) in K_S}',
            'ratio': 'Approximately count(satisfied paired samples)/sample_set.count',
            'denominator_must_be_positive': True,
            'method': 'Batch joint six-dimensional feasibility on the same stored samples for every candidate; no work-surface integration',
            'numerical_status': 'Finite-sample approximation; use an independent larger sample set to recheck finalists',
            'all_load_success': 'Passing all samples is not a proof over the continuous domain',
            'existing_implementation': 'slides/baseline_algo/step3.1_score_candidate/contribution.py',
            'existing_implementation_scope': 'Scores accepted unselected Step 2 circles jointly with the current fixed contacts; Step 3.2 chooses the maximum',
            'full_interval_feasibility': 'For each allowed position and direction, affine dependence on magnitude and convexity of the supply cone make zero and upper-endpoint feasibility sufficient and necessary for the entire interval',
        },
        'figure_notation': {
            'source': 'slides/obj_supp/demand/demand_equation.py, demand pair formula and layout',
            'units': 'Physical force and moment, retaining mg as in equations; numerical exports remain normalized by mg',
            'range': 'Paired right-hand sides as pt and the complete F_push vary over their allowed continuous range',
        },
        'figure_math': {
            'pair': (
                r'$\mathrm{demand}(F_{\rm push},\,\mathrm{pt})'
                r'\;=\;\left(F_D,\;\tau_D\right)$'),
            'pair_expanded': (
                r'$\left(F_D,\;\tau_D\right)'
                r'\;=\;\left(mg\,\hat{z}-F_{\rm push},\;'
                r'-(\mathrm{pt}-c)\times F_{\rm push}\right)$'),
            'definitions': (
                r'$\mathrm{pt}$: push location   ·   $F_{\rm push}$: applied force'
                r'   ·   $r_{\rm push}=\mathrm{pt}-c$: arm from center of mass $c$'),
            'allowed_range': (
                r'$\mathrm{pt}\in\mathrm{work\ region},\qquad'
                rf'0\leq |F_{{\rm push}}|\leq {first["K"]:g}\,mg$'),
        },
        'provenance': {'generator': str(Path(__file__).relative_to(ROOT)),
                       'generator_sha256': sha256(__file__), 'example_files_used': []},
    }


def read_object(name, path=None):
    """Resolve the physical geometry and verify the referenced sample input."""
    path = Path(path) if path is not None else OUTPUTS / name/pose_name() / 'step_1_needs/domain.json'
    spec = json.loads(path.read_text())
    record = spec['objects'][name]
    source = path.parent / record['source_file']
    if sha256(source) != record['source_sha256']:
        raise ValueError(f'{name}: continuous domain references changed geometry; regenerate domain.json')
    sample_record = spec['sample_set']
    if sha256(path.parent / sample_record['source_file']) != sample_record['source_sha256']:
        raise ValueError(f'{name}: samples changed; regenerate domain.json')
    return ContinuousNeeds.read(source), spec


def draw(path):
    """Draw formulas from the saved JSON, without evaluating any load samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spec = json.loads(Path(path).read_text())
    formula = spec['figure_math']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(16, 6.6), dpi=200, facecolor=PAPER)
    labels = []

    def text(y, value, size=22, color=INK):
        labels.append(fig.text(.5, y, value, fontsize=size, color=color,
                               ha='center', va='center'))

    text(.905, f'{spec["object"]}  /  Step 1: sampled demand pairs', 36)
    text(.805, f'{spec["sample_set"]["count"]:,} reachable samples  ·  equal weights  ·  local cone = {spec["load"]["cone_half_deg"]:g}°', 22, MUTED)
    text(.620, formula['pair'], 36)
    text(.420, formula['pair_expanded'], 36)
    text(.205, formula['definitions'], 20, MUTED)
    text(.090, formula['allowed_range'], 22)
    # Catch a clipped equation when fonts, notation or figure dimensions change.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        bounds = label.get_window_extent(renderer)
        if not (bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.width
                and bounds.y1 <= fig.bbox.height):
            raise RuntimeError(f'Text extends beyond the figure: {label.get_text()}')
    image = Path(path).with_suffix('.png')
    fig.savefig(image, facecolor=PAPER)
    plt.close(fig)
    return image


def build(name):
    """Refresh the per-object index and formula figure from saved samples."""
    path = OUTPUTS / name/pose_name() / 'step_1_needs/domain.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(specification(name), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    read_object(name, path)
    print(path)
    print(draw(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    args = parser.parse_args()
    names = args.objects or OBJECTS
    if any(name not in OBJECTS for name in names):
        parser.error('objects must be A1-f, B or C5')
    for name in names:
        build(name)


if __name__ == '__main__':
    main()
