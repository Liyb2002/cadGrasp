"""Sample paired six-dimensional demands at the existing setup pose.

The physical domain and geometry remain in needs.json for downstream geometry
readers. samples.json is the finite, equally weighted search input. No coverage
integration, contact candidates, or old demand samples are used.

Running this script refreshes samples, examples, domain metadata and both figures
for each requested object. build() exposes the data-only export for library use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent
ROOT = HERE.parents[2]
OUTPUTS = BASELINE / 'output'
sys.path.insert(0, str(BASELINE))
from step1.cases import pose_name
sys.path.insert(0, str(ROOT / 'slides/tools'))
import coordinates as COORD
from cone_model import CONE_HALF_DEG, frame

OBJECTS = ('A1-f', 'B', 'C5')
K = 0.5
GRAVITY = np.array([0., 0., -1.])
RAY_OFFSET_M = 1e-5
DEFAULT_SAMPLE_COUNT = 32768
DEFAULT_SAMPLE_SEED = 20260907
SAMPLE_BATCH_SIZE = 4096


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'),
                               allow_nan=False) + '\n')


def demand(q_m, force_push_mg, com_m, gravity_mg=GRAVITY):
    """Return (..., 6), ordered Fx,Fy,Fz,taux,tauy,tauz; mg and mg*m.

    This map accepts a complete force vector. Admissibility (surface point,
    K, cap and reachability) is checked separately by ContinuousNeeds.evaluate.
    """
    q, force = np.broadcast_arrays(np.asarray(q_m, float),
                                  np.asarray(force_push_mg, float))
    if q.shape[-1] != 3 or not (np.isfinite(q).all() and np.isfinite(force).all()):
        raise ValueError('Expected finite 3D positions and force vectors')
    return np.concatenate((-(force + gravity_mg),
                           -np.cross(q - com_m, force)), axis=-1)


def setup_geometry(name):
    """Replay only uniform subdivision; never rerun setup selection/rendering."""
    pose = pose_name()
    source = ROOT / 'slides/setup/poses' / name / pose / 'setup.npz'
    if not source.is_file():
        raise FileNotFoundError(f'{name}/{pose}: create the setup pose first ({source})')
    with np.load(source) as z:
        meta = {k: z[k].copy() for k in (
            'T_world_mesh', 'com_m', 'work_faces', 'mesh_sha256',
            'poses_sha256', 'K', 'cone_half_deg', 'tip')}
        if 'pose_id' in z and str(z['pose_id']) != pose:
            raise ValueError(f'{name}/{pose}: setup snapshot belongs to a different pose')
    for file, key in [('mesh.stl', 'mesh_sha256'), ('poses.json', 'poses_sha256')]:
        if sha256(ROOT / 'objects' / name / file) != str(meta[key]):
            raise ValueError(f'{name}: stale setup snapshot ({file})')
    if float(meta['K']) != K:
        raise ValueError('Setup snapshot uses a different force magnitude limit')
    raw = trimesh.load(ROOT / 'objects' / name / 'mesh.stl', force='mesh')
    vertices, faces = np.asarray(raw.vertices), np.asarray(raw.faces)
    rounds = 0
    while len(faces) < len(meta['work_faces']) and rounds < 6:
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
        rounds += 1
    if len(faces) != len(meta['work_faces']):
        raise ValueError('Cannot reproduce setup work-face indexing')
    local = trimesh.Trimesh(vertices, faces, process=False)
    np.testing.assert_allclose(local.area, raw.area, rtol=1e-12)
    np.testing.assert_allclose(local.volume, raw.volume, rtol=1e-12)
    T = meta['T_world_mesh']
    R, t = T[:3, :3], T[:3, 3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1., atol=1e-12)
    com = R @ local.center_mass + t
    np.testing.assert_allclose(com, meta['com_m'], atol=1e-12)
    world = trimesh.Trimesh(vertices @ R.T + t, faces, process=False)
    inward = -(local.face_normals @ R.T)[meta['work_faces']]
    tangent1, tangent2 = frame(inward)
    return {
        'schema_version': 2,
        'coordinate_system': 'z_up_xy_floor',
        'object': name,
        'pose_id': pose,
        'representation': 'continuous_parametric_wrench_set',
        'finite_enumeration': False,
        'scope': f'Current triangle mesh, setup {pose}, every process magnitude from zero to K',
        'units': {'position': 'm', 'force': 'mg', 'moment': 'mg*m',
                  'mg': 'object weight; multiply both wrench blocks by weight in N for SI'},
        'frame': {'name': 'world', 'up': [0, 0, 1], 'moment_origin_m': com.tolist(),
                  'T_world_mesh': T.tolist(),
                  'tip': int(meta['tip']) if int(meta['tip']) >= 0 else None},
        'load': {'K': K, 'magnitude_range_mg': [0., K],
                 'magnitude_endpoints_included': True,
                 'gravity_force_mg': GRAVITY.tolist(),
                 'gravity_application_point_m': com.tolist(),
                 'cone_half_deg': CONE_HALF_DEG},
        'parameters': {
            'work_face_index': 'integer index into geometry.work_face_ids',
            'u_v': 'u >= 0, v >= 0, u+v <= 1 (entire closed triangle)',
            'theta_rad': [0, float(np.deg2rad(CONE_HALF_DEG))],
            'phi_rad': [0, float(2*np.pi)],
            'magnitude_mg': [0., K],
            'phi_endpoint': 'periodic; 0 and 2*pi denote the same direction',
            'boundary_normals': 'Face-wise normals; the set is the union of face families',
            'q_m': 'a + u*(b-a) + v*(c-a); a,b,c are the indexed world triangle',
            'd': 'cos(theta)*inward + sin(theta)*(cos(phi)*tangent1 + sin(phi)*tangent2)',
            'F_push_mg': 'magnitude_mg*d',
        },
        'mapping': {
            'pair_order': ['Fx', 'Fy', 'Fz', 'tau_x', 'tau_y', 'tau_z'],
            'push_wrench': '[F_push, cross(q-COM,F_push)]',
            'external_wrench': 'push_wrench + [gravity_force,0,0,0]',
            'need_wrench': '-external_wrench',
            'need_force_mg': '-gravity_force_mg - F_push_mg',
            'need_moment_mgm': '-cross(q_m-COM_m,F_push_mg)',
            'set': '{ need(q,magnitude_mg*d) : every work face, every u,v,theta,phi in domain, 0 <= magnitude_mg <= K, reachable(q,d) }',
            'pairing': 'Both blocks share exactly the same q and complete F_push',
        },
        'reachability': {
            'obstacle': 'complete exported object mesh',
            'escape_direction': '-d',
            'ray_origin': 'q + ray_offset_m*outward_face_normal',
            'ray_offset_m': RAY_OFFSET_M,
            'predicate': 'no object-mesh intersection along the outward half-ray',
            'implementation': 'trimesh.ray.intersects_any, float64; numerical geometry test',
            'force_application_point': 'q itself, NEVER the offset ray origin',
            'scope': 'Object self-occlusion only; the inherited tool-ray convention',
        },
        'sampling_measure': {
            'domain': 'reachable surface positions x allowed force directions x force-magnitude interval',
            'position_direction_density': '2*face_area_m2*sin(theta) du dv dtheta dphi',
            'magnitude_weighting': 'uniform on [0,K]',
            'six_dimensional_volume': False,
            'note': 'Defines the physical sampling distribution; samples.json contains the finite scoring input. No coverage is computed here.'
        },
        'geometry': {
            'vertices_m': world.vertices.tolist(), 'faces': world.faces.tolist(),
            'work_face_ids': np.flatnonzero(meta['work_faces']).tolist(),
            'inward_normals': inward.tolist(),
            'tangent1': tangent1.tolist(), 'tangent2': tangent2.tolist(),
            'work_face_areas_m2': world.area_faces[meta['work_faces']].tolist(),
            'total_area_m2': float(world.area),
            'work_area_m2': float(world.area_faces[meta['work_faces']].sum()),
        },
        'provenance': {
            'setup_snapshot': str(source.relative_to(ROOT)),
            'setup_snapshot_sha256': sha256(source),
            'setup_snapshot_cone_half_deg': float(meta['cone_half_deg']),
            'force_cone_override': ('The setup snapshot fixes pose and work faces. '
                                    'Step 1 applies the current declared process cone.'),
            'fields_used': ['T_world_mesh', 'com_m', 'work_faces', 'K', 'cone_half_deg', 'tip',
                            'mesh_sha256', 'poses_sha256'],
            'sampled_push_or_coverage_fields_used': [],
            'mesh_sha256': str(meta['mesh_sha256']),
            'poses_sha256': str(meta['poses_sha256']),
            'uniform_subdivision_rounds': rounds,
            'generator_sha256': sha256(__file__),
            'numpy_version': np.__version__, 'trimesh_version': trimesh.__version__,
        },
    }


class ContinuousNeeds:
    """Evaluate any member of the exported continuous family, not a lookup table."""

    def __init__(self, data):
        self.data = data
        g = data['geometry']
        self.mesh = trimesh.Trimesh(g['vertices_m'], g['faces'], process=False)
        self.work_ids = np.asarray(g['work_face_ids'], int)
        self.normals = np.asarray(g['inward_normals'])
        self.e1, self.e2 = np.asarray(g['tangent1']), np.asarray(g['tangent2'])
        self.com = np.asarray(data['frame']['moment_origin_m'])
        self.gravity = np.asarray(data['load']['gravity_force_mg'])
        self.k = float(data['load']['K'])
        self.magnitude_range = np.asarray(data['load'].get('magnitude_range_mg', [self.k, self.k]), float)
        self.half_angle = np.deg2rad(data['load']['cone_half_deg'])
        self.ray_offset = float(data['reachability']['ray_offset_m'])

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def evaluate(self, work_face_index, u, v, theta_rad, phi_rad, magnitude_mg=None):
        """Evaluate any allowed magnitude; omission selects the upper endpoint.

        Tool reachability is evaluated for every parameter point, including the
        zero-magnitude boundary.  The finite scoring set contains no separately
        injected pure-gravity condition.
        """
        magnitude = self.k if magnitude_mg is None else magnitude_mg
        i, u, v, theta, phi, magnitude = np.broadcast_arrays(
            np.asarray(work_face_index), u, v, theta_rad, phi_rad, magnitude)
        if not (np.isfinite(i).all() and np.isfinite(u).all() and np.isfinite(v).all()
                and np.isfinite(theta).all() and np.isfinite(phi).all()
                and np.isfinite(magnitude).all()):
            raise ValueError('Parameters must be finite')
        if np.any((i != i.astype(int)) | (i < 0) | (i >= len(self.work_ids))):
            raise ValueError('Invalid work-face index')
        if np.any((u < 0) | (v < 0) | (u+v > 1) | (theta < 0) | (theta > self.half_angle)):
            raise ValueError('Parameters outside the work triangle or direction cap')
        if np.any((magnitude < self.magnitude_range[0]) | (magnitude > self.magnitude_range[1])):
            raise ValueError('Force magnitude outside the declared interval')
        i = i.astype(int)
        tri = self.mesh.triangles[self.work_ids[i]]
        q = tri[..., 0, :] + u[..., None]*(tri[..., 1, :]-tri[..., 0, :])
        q += v[..., None]*(tri[..., 2, :]-tri[..., 0, :])
        d = np.cos(theta)[..., None]*self.normals[i] + np.sin(theta)[..., None]*(
            np.cos(phi)[..., None]*self.e1[i] + np.sin(phi)[..., None]*self.e2[i])
        origins = q - self.ray_offset*self.normals[i]
        tool_reachable = ~self.mesh.ray.intersects_any(origins.reshape(-1, 3), (-d).reshape(-1, 3))
        tool_reachable = tool_reachable.reshape(i.shape)
        reachable = tool_reachable
        f = magnitude[..., None]*d
        push = np.concatenate((f, np.cross(q-self.com, f)), axis=-1)
        external = push + np.r_[self.gravity, [0., 0., 0.]]
        need = demand(q, f, self.com, self.gravity)
        return dict(q_m=q, d=d, force_push_mg=f, push_wrench=push,
                    external_wrench=external, need_wrench=need,
                    magnitude_mg=magnitude, tool_reachable=tool_reachable,
                    reachable=reachable)


def example_cases(domain):
    """Three reproducible arbitrary loads; no support/coverage-based selection."""
    rng = np.random.default_rng(20260907)
    areas = np.asarray(domain.data['geometry']['work_face_areas_m2'])
    cases = []
    used_faces = set()
    for attempt in range(300):
        i = int(rng.choice(len(areas), p=areas/areas.sum()))
        u, v = rng.random(2)
        if u+v > 1:
            u, v = 1-u, 1-v
        theta = float(np.arccos(rng.uniform(np.cos(domain.half_angle), 1)))
        phi = float(rng.uniform(0, 2*np.pi))
        magnitude = [domain.k/4, domain.k/2, domain.k][len(cases)]
        result = domain.evaluate(i, u, v, theta, phi, magnitude_mg=magnitude)
        if not result['tool_reachable'] or i in used_faces:
            continue
        used_faces.add(i)
        record = {'id': len(cases)+1, 'work_face_index': i,
                  'mesh_face_id': int(domain.work_ids[i]),
                  'parameters': {'u': float(u), 'v': float(v), 'theta_rad': theta,
                                 'phi_rad': phi, 'magnitude_mg': magnitude}}
        record.update({('pt_m' if key == 'q_m' else key): value.tolist()
                       for key, value in result.items()})
        # Independent component expansion; no np.cross or demand() for this check.
        x, y, z = result['q_m']-domain.com
        fx, fy, fz = result['force_push_mg']
        tau = np.array([y*fz-z*fy, z*fx-x*fz, x*fy-y*fx])
        residual = np.r_[result['need_wrench'][:3]+result['force_push_mg']+domain.gravity,
                         result['need_wrench'][3:]+tau]
        record['verification'] = {
            'moment_from_component_expansion_mgm': tau.tolist(),
            'force_balance_residual_mg': residual[:3].tolist(),
            'moment_balance_residual_mgm': residual[3:].tolist(),
            'norm_F_push_mg': float(np.linalg.norm(result['force_push_mg'])),
            'cap_angle_deg': float(np.rad2deg(theta)),
            'surface_point_is_barycentric': True,
        }
        np.testing.assert_allclose(residual, 0, atol=1e-13)
        cases.append(record)
        if len(cases) == 3:
            return {'schema_version': 2, 'object': domain.data['object'], 'seed': 20260907,
                    'purpose': 'Three individual illustrative loads; never applied simultaneously',
                    'selection': 'Area-weighted random faces and uniform solid angle; reject occluded rays; illustrative magnitudes K/4, K/2, K',
                    'continuous_set_file': 'needs.json', 'cases': cases}
    raise RuntimeError('Could not find three distinct reachable work faces')


def sample_needs(domain, count=DEFAULT_SAMPLE_COUNT, seed=DEFAULT_SAMPLE_SEED):
    """IID physical loads conditioned on tool visibility, with equal weights.

    Draw six random numbers per proposal: face, two surface coordinates, solid
    angle cosine, azimuth, and magnitude. Fixed proposal batches make larger
    counts preserve the exact accepted prefix for the same seed. Do not sample
    the six wrench components independently or deduplicate repeated wrenches.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Sample count must be a positive integer')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Sample seed must be a nonnegative integer')
    if not np.array_equal(domain.magnitude_range, [0., domain.k]) or domain.k <= 0:
        raise ValueError('Sampling requires the magnitude interval [0,K], K > 0')
    rng = np.random.Generator(np.random.PCG64(seed))
    areas = np.asarray(domain.data['geometry']['work_face_areas_m2'])
    if not np.isfinite(areas).all() or (areas <= 0).any() or not len(areas):
        raise ValueError('Sampling requires positive finite work-face areas')
    cdf = np.cumsum(areas/areas.sum())
    cdf[-1] = 1.
    chunks = {key: [] for key in ('work_face_index', 'parameters', 'pt_m',
                                  'force_push_mg', 'need_wrench')}
    accepted = proposals = evaluated = 0
    budget = max(SAMPLE_BATCH_SIZE, 100*count)
    while accepted < count and evaluated < budget:
        random = rng.random((min(SAMPLE_BATCH_SIZE, budget-evaluated), 6))
        indices = np.searchsorted(cdf, random[:, 0], side='right')
        u, v = random[:, 1].copy(), random[:, 2].copy()
        reflect = u+v > 1
        u[reflect], v[reflect] = 1-u[reflect], 1-v[reflect]
        theta = np.arccos(np.cos(domain.half_angle) +
                          random[:, 3]*(1-np.cos(domain.half_angle)))
        phi = 2*np.pi*random[:, 4]
        magnitude = domain.k*random[:, 5]
        values = domain.evaluate(indices, u, v, theta, phi, magnitude_mg=magnitude)
        hits = np.flatnonzero(values['tool_reachable'])
        take = hits[:count-accepted]
        used = int(take[-1])+1 if len(take) == count-accepted else len(random)
        proposals += used
        evaluated += len(random)
        chunks['work_face_index'].append(indices[take])
        chunks['parameters'].append(np.c_[u, v, theta, phi, magnitude][take])
        for source, target in [('q_m', 'pt_m'), ('force_push_mg', 'force_push_mg'),
                               ('need_wrench', 'need_wrench')]:
            chunks[target].append(values[source][take])
        accepted += len(take)
    if accepted != count:
        raise RuntimeError(f'Only {accepted} of {count} reachable samples after {evaluated} proposals')
    arrays = {key: np.concatenate(parts).tolist() for key, parts in chunks.items()}
    return {
        'schema_version': 1, 'object': domain.data['object'],
        'representation': 'sampled_paired_wrench_set',
        'count': int(count), 'seed': int(seed), 'random_generator': 'PCG64',
        'method': 'iid_area_solid_angle_uniform_magnitude_with_visibility_rejection',
        'approximation': 'Finite samples of the physical demand set; not a six-dimensional volume or a proof of full coverage',
        'sampling_distribution': {
            'position': 'Face probability proportional to physical area; uniform barycentric area within each face',
            'direction': 'Uniform solid angle in the local inward cone; uniform cos(theta) and phi',
            'magnitude': 'Uniform magnitude in [0,K]; not uniform force-vector volume',
            'conditioning': 'Reject occluded tool rays and continue until count reachable demands are collected',
            'joint_measure': 'reachable surface area x solid angle x uniform magnitude',
        },
        'weight_per_sample': 1./count,
        'coverage_estimator': 'sum(satisfied_i)/count, using the same fixed samples for every candidate',
        'score_increment_percentage_points': 100./count,
        'score_increment_is_error_bound': False,
        'pair_order': ['Fx', 'Fy', 'Fz', 'tau_x', 'tau_y', 'tau_z'],
        'parameter_order': ['u', 'v', 'theta_rad', 'phi_rad', 'magnitude_mg'],
        'units': {'pt_m': 'm', 'force_push_mg': 'mg', 'need_force': 'mg', 'need_moment': 'mg*m'},
        'moment_origin_m': domain.com.tolist(),
        'magnitude_range_mg': domain.magnitude_range.tolist(),
        'proposal_count_until_last_sample': proposals,
        'rejected_proposal_count': proposals-count,
        'evaluated_proposals_including_unused_batch_tail': evaluated,
        **arrays,
    }


def build(name, count=DEFAULT_SAMPLE_COUNT, seed=DEFAULT_SAMPLE_SEED):
    folder = OUTPUTS/name/pose_name()/'step_1_needs'
    path = folder/'needs.json'
    stored = json.loads(path.read_text()) if path.exists() else None
    if stored is not None and stored.get('pose_id', 'pose_1') != pose_name():
        raise ValueError(f'{name}: saved needs belong to a different pose')
    if (stored is None or 'coordinate_migration' in stored or 'pose_id' not in stored or
            float(stored['load']['cone_half_deg']) != CONE_HALF_DEG):
        save_json(path, setup_geometry(name))
    domain = ContinuousNeeds.read(path)
    if domain.data.get('pose_id', 'pose_1') != pose_name():
        raise ValueError(f'{name}: saved needs belong to a different pose')
    # Sampling alone must not invalidate Step 2's unchanged geometry hashes.
    provenance = domain.data['provenance']
    for source, digest in [(ROOT/provenance['setup_snapshot'], provenance['setup_snapshot_sha256']),
                            (ROOT/'objects'/name/'mesh.stl', provenance['mesh_sha256']),
                            (ROOT/'objects'/name/'poses.json', provenance['poses_sha256'])]:
        if sha256(source) != digest:
            raise ValueError(f'{name}: rebuild the stale physical domain before sampling ({source})')
    samples = sample_needs(domain, count, seed)
    samples['provenance'] = {
        'physical_domain_file': 'needs.json', 'physical_domain_sha256': sha256(path),
        'generator': str(Path(__file__).relative_to(ROOT)), 'generator_sha256': sha256(__file__),
        'numpy_version': np.__version__, 'trimesh_version': trimesh.__version__,
        'support_or_coverage_inputs_used': [],
    }
    save_json(folder/'samples.json', samples)
    examples = example_cases(domain)
    examples['needs_sha256'] = sha256(path)
    save_json(folder/'examples.json', examples)
    print(f'{name}: {count:,} reachable paired demands in a {domain.data["load"]["cone_half_deg"]:g}-degree cone '
          f'({samples["rejected_proposal_count"]} rejected proposals) -> {folder.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--count', type=int, default=DEFAULT_SAMPLE_COUNT,
                        help='Number of reachable demands per object (default: 32768)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SAMPLE_SEED)
    args = parser.parse_args()
    if args.count < 1 or args.seed < 0 or any(name not in OBJECTS for name in args.objects):
        parser.error('objects must be A1-f, B or C5; count > 0 and seed >= 0')
    from domain import build as build_domain
    from draw_proof import draw as draw_examples
    for name in args.objects or OBJECTS:
        build(name, args.count, args.seed)
        build_domain(name)
        draw_examples(name)
