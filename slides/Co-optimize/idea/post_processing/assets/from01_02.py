"""Continue video 01's exact connected shape with text-free nested deletions.

The movie is a same-shape deletion illustration. It never switches to the
different published round-stock reconstruction. Complete hollow rims and
their existing connectors are retained; cuts that disconnect material are
rejected. Numerical result files and the first movie are read-only.
"""
from pathlib import Path
import itertools
import json
import shutil

import prune02 as B
import prepare02 as P
from deterministic_base import box
import numpy as np
import trimesh
import manifold3d as md

R = B.R
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/video02_from01'
ORIGINAL = HERE / 'data/illustrative_support.obj'
BODY_TARGET = R.BASE / 'step5/step5.2/pso_all52_20261013_01_rounded/pruned_support.obj'
BaseMovie = B.Movie
COPLANAR_TOL = 1e-7  # Manifold coordinates: 20 nm in the displayed metre frame.
DISPLAY_VOLUME_TOL = 1e-9  # Near-coincident Boolean residual: 0.001 cm³.


def components(value):
    return [part for part in value.decompose() if R.volume(part) > 1e-12]


def narrow_rim(ring, fraction):
    outer = ring.inner + (ring.outer - ring.inner) * fraction
    pieces = []
    for i in range(len(outer)):
        j = (i + 1) % len(outer)
        pieces.append(R.prism(np.array([ring.inner[i], ring.inner[j], outer[j], outer[i]])))
    return R.moved(R.union(pieces), ring.inverse)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    first, rows = P.load_rows()
    ref = np.asarray(first['fixed_absolute_fixture_transform'])
    if not np.array_equal(ref, rows[0]['T']):
        raise ValueError('Video 01 installation changed')
    protected = dict(B.read(HERE / 'data/video02/stages.json')['protected_inputs'])
    protected.update(B.read(HERE / 'data/inputs.json'))
    for path in [ORIGINAL, HERE / 'data/manifest.json', HERE / 'data/geometry.npz', BODY_TARGET]:
        protected[R.relative(path)] = R.digest(path)
    previous_manifest = B.read(HERE / 'data/video02_minimal/manifest.json')
    original_mesh = trimesh.load(ORIGINAL, force='mesh', process=False)
    core_mesh = trimesh.load(R.STEP4 / 'support.obj', force='mesh', process=False)
    original, core = R.solid(original_mesh), R.solid(core_mesh)
    if len(components(original)) != 1:
        raise ValueError('The first movie endpoint must already be connected')
    rings = [R.Ring(row, core_mesh) for row in rows]
    rods = R.union([R.solid(ring.link) for ring in rings])
    rim_values = [ring.solid for ring in rings]
    reconstructed = R.union([core, rods, *rim_values])
    if R.volume(original - reconstructed) + R.volume(reconstructed - original) > 1e-11:
        raise ValueError('Reconstructed feet do not match the first movie')
    # This saved same-layout body is only an interior retention mask. The final
    # illustrative object also keeps the old feet and any necessary old links.
    body_target = (core ^ R.solid(trimesh.load(BODY_TARGET, force='mesh', process=False))).simplify(COPLANAR_TOL)
    protected_feet = R.union([rods, *[narrow_rim(ring, .75) for ring in rings]]).simplify(COPLANAR_TOL)
    keep = (body_target + protected_feet).simplify(COPLANAR_TOL) ^ original
    current = original
    records = []
    rejected_disconnected = 0

    def save(value, phase, detail, exact_original=False):
        mesh = original_mesh if exact_original else R.unpack(value)
        if not mesh.is_watertight or len(components(value)) != 1:
            raise ValueError('Every displayed shape must be closed and connected')
        bounds = B.mesh_bounds(mesh, ref)
        material = abs(float(mesh.volume)) * 1e6
        if records:
            old_bounds = np.asarray(records[-1]['bounds_world_m'])
            if material > records[-1]['material_cm3'] + 1e-5:
                raise ValueError('Deletion added material')
            if np.any(bounds[0] < old_bounds[0] - 1e-9) or np.any(bounds[1] > old_bounds[1] + 1e-9):
                raise ValueError('Deletion expanded the box')
        missing = R.volume(protected_feet - value)
        if missing > DISPLAY_VOLUME_TOL:
            raise ValueError(f'A complete rim or old connector was cut: {missing} m³')
        index = len(records)
        path = DATA / 'meshes' / f'{index:03d}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
        records.append(dict(index=index, phase=phase, detail=detail,
            mesh=R.relative(path), mesh_sha256=R.digest(path), material_cm3=material,
            bounds_world_m=bounds.tolist(), bbx_l=float(np.prod(bounds[1] - bounds[0]) * 1000),
            connected_components=1, watertight=True, protected_rim_and_connector_missing_m3=missing,
            source_stage=None, certified_optimizer_intermediate=False, exact_final_source_sha256=None))
        print(json.dumps(dict(stage=index, phase=phase, cm3=round(material, 3),
            bbx_l=round(records[-1]['bbx_l'], 6))), flush=True)

    save(current, 'video01_exact_endpoint', {}, exact_original=True)
    # Shave only the outside width; each ring keeps its entire original closed
    # inner contour. No arcs, original rods, or support connections disappear.
    center = np.mean(np.asarray(records[0]['bounds_world_m']), axis=0)
    rim_order = sorted(range(5), key=lambda i: -float(np.max(np.linalg.norm(
        R.transform(rings[i].mesh.vertices, ref) - center, axis=1))))
    for fraction in [.875, .75]:
        for i in rim_order:
            proposed = list(rim_values)
            proposed[i] = narrow_rim(rings[i], fraction)
            candidate = R.union([core, rods, *proposed]).simplify(COPLANAR_TOL) ^ original
            if R.volume(current) - R.volume(candidate) < 1e-10:
                continue
            if len(components(candidate)) != 1:
                rejected_disconnected += 1
                continue
            current, rim_values = candidate, proposed
            save(current, 'outer_rim_width', dict(pose=rows[i]['pose'], full_closed_contour=True,
                remaining_width_mm=4 * fraction))
    # Recompute the exact protected feet after any conservative ring rejection.
    keep = R.union([body_target, rods, *rim_values]).simplify(COPLANAR_TOL) ^ original
    core_world = R.transform(core_mesh.vertices, ref)
    low, high = core_world.min(0) - 1e-6, core_world.max(0) + 1e-6
    edges = [np.linspace(low[j], high[j], 6) for j in range(3)]
    center, extent = (low + high) / 2, (high - low) / 2
    cells = []
    for ijk in itertools.product(range(5), repeat=3):
        a = np.array([edges[j][ijk[j]] for j in range(3)])
        b = np.array([edges[j][ijk[j] + 1] for j in range(3)])
        score = float(np.linalg.norm(((a + b) / 2 - center) / extent))
        cells.append((-score, ijk, a, b))
    cells.sort(key=lambda row: (row[0], row[1]))
    inverse = np.linalg.inv(ref)
    cut_mask = md.Manifold()

    def cut(a, b):
        nonlocal current, cut_mask, rejected_disconnected
        cutter = R.moved(box(a, b), inverse)
        proposed_mask = (cut_mask + cutter).simplify(COPLANAR_TOL)
        # Rebuild from the same original body and accumulated cuts. Reusing an
        # already re-triangulated union would repeatedly subdivide coincident
        # retained surfaces without changing the actual displayed solid.
        candidate = (keep + (core - proposed_mask)).simplify(COPLANAR_TOL) ^ original
        if R.volume(current) - R.volume(candidate) < 2.5e-8:
            return False
        if len(components(candidate)) == 1:
            current, cut_mask = candidate, proposed_mask
            return True
        rejected_disconnected += 1
        return False

    for _, ijk, a, b in cells:
        if cut(a, b):
            save(current, 'connected_body_cut', dict(cell=list(ijk)))
    if len(records) < 12 or records[-1]['material_cm3'] > records[0]['material_cm3'] * .85:
        raise ValueError('Insufficient visible deletion')
    final_path = DATA / 'final_support.obj'
    R.unpack(current).export(final_path, digits=17)
    final_mesh = trimesh.load(final_path, force='mesh', process=False)
    final_array_path = R.ROOT / records[-1]['mesh']
    np.savez_compressed(final_array_path, vertices=np.asarray(final_mesh.vertices), faces=np.asarray(final_mesh.faces))
    records[-1]['mesh_sha256'] = R.digest(final_array_path)
    records[-1]['phase'] = 'same_shape_final'
    records[-1]['exact_final_source_sha256'] = R.digest(final_path)
    protected[R.relative(final_path)] = R.digest(final_path)
    for path, sha in protected.items():
        if R.digest(R.ROOT / path) != sha:
            raise ValueError('Protected original changed: ' + path)
    R.save(DATA / 'previous_version.json', dict(manifest=previous_manifest,
        reason='Starting shape did not match the endpoint of video 01'))
    R.save(DATA / 'inputs.json', protected)
    R.save(DATA / 'geometry.json', dict(complete=True, records=records,
        exterior_cut_end_index=len(records)-1, reference_transform=ref.tolist(), protected_inputs=protected,
        source_video01_mesh=R.relative(ORIGINAL), source_video01_mesh_sha256=R.digest(ORIGINAL),
        final_obj_source=R.relative(final_path), final_obj_sha256=R.digest(final_path),
        initial_geometry_exactly_matches_video01=True, camera_basis_matches_video01_endpoint=True,
        all_five_hollow_rims_retained=True, original_connectors_retained=True,
        all_displayed_shapes_single_connected_component=True, disconnected_cut_rejections=rejected_disconnected,
        display_coplanar_cleanup_tolerance_m=COPLANAR_TOL*.2,
        protected_geometry_boolean_volume_tolerance_m3=DISPLAY_VOLUME_TOL,
        every_displayed_cut_intersected_with_original_video01_solid=True,
        new_material_added=False, illustrative_geometry_only=True, optimizer_replay=False,
        published_step5_round_reconstruction_used=False, new_physical_certification_claimed=False,
        numerical_models_modified=False, force_solver_rerun=False))
    B.freeze()


class Movie(BaseMovie):
    def __init__(self):
        super().__init__()
        points = B.M.box_vertices(np.asarray(self.records[0]['bounds_world_m']))
        basis = R.basis(2 * np.pi)
        projected = points @ basis.T
        low, high = projected.min(0), projected.max(0)
        focus = ((low + high) / 2) @ basis
        size = high - low
        self.camera = (focus, basis, max(size[1] / .84, size[0] / (B.WIDTH / B.HEIGHT) / .84))

    def at(self, t):
        if t < B.START:
            return 0
        if t >= B.PRUNE_END:
            return len(self.records) - 1
        return min(len(self.records)-1, 1 + int((t-B.START)/(B.PRUNE_END-B.START)*(len(self.records)-1)))

    def verify(self):
        result = super().verify()
        first = trimesh.load(ORIGINAL, force='mesh', process=False)
        with np.load(R.ROOT / self.records[0]['mesh']) as arrays:
            if not np.array_equal(arrays['vertices'], first.vertices) or not np.array_equal(arrays['faces'], first.faces):
                raise ValueError('The first frame is not the exact video 01 endpoint')
        result.update(first_model_exact_video01_vertices_and_faces=True,
            source_video01_mesh_sha256=R.digest(ORIGINAL),
            same_fixed_absolute_pose_as_video01=True, same_camera_direction_as_video01_endpoint=True,
            all_five_rims_complete_and_closed=True, original_connectors_retained=True,
            every_displayed_stock_connected=True, different_round_stock_reconstruction_used=False,
            final_is_same_shape_illustration=True, final_is_published_step5_result=False,
            bbx_shrink_steps=sum(b['bbx_l'] < a['bbx_l']-1e-8 for a,b in zip(self.records,self.records[1:])))
        return result


if __name__ == '__main__':
    import sys
    prior_version = B.read(DATA / 'previous_version.json') if (DATA / 'previous_version.json').exists() else None
    B.DATA = DATA
    B.prepare = prepare
    B.Movie = Movie
    B.main()
    if '--encode' in sys.argv:
        if prior_version:
            R.save(DATA / 'previous_version.json', prior_version)
        manifest = B.read(DATA / 'manifest.json')
        manifest.update(initial_model_exact_video01=True, source_video01_mesh_sha256=R.digest(ORIGINAL),
            five_complete_hollow_rims_retained=True, existing_connectors_retained=True,
            all_displayed_models_connected=True, final_is_published_step5_result=False,
            different_round_stock_reconstruction_used=False)
        R.save(DATA / 'manifest.json', manifest)
