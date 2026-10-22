"""Baseline: cover the entire source surface with connected working regions.

The Step1.3 geometry sampler supplies feasible regions and witness alignments.
Step1.2 selects from that pool; its numbering describes its planning role, not
the execution order of these two modules. No region count or fixture objective
is supplied. Each operation keeps its complete region, including overlaps.

    .venv/bin/python -B -m whole_task_2.step1.step1_2
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix

from .common import DEFAULT_OUTPUT, ROOT, load_shape, sha256, stage_directory, write_json


class IncompleteCandidatePool(ValueError):
    def __init__(self, face_ids, area_m2):
        self.face_ids = np.asarray(face_ids, dtype=int)
        self.area_m2 = float(area_m2)
        super().__init__(f"Candidate pool cannot cover {len(self.face_ids)} source faces "
                         f"({self.area_m2*1e4:.6g} cm2): {self.face_ids.tolist()}")


def greedy_cover(membership, face_area_m2, candidate_indices=None):
    """Area-weighted greedy set cover, with stable source-index tie breaking.

    Return source candidate indices, never residual fragments. Strict coverage
    uses face membership rather than an area tolerance: even the last small
    triangle must have a feasible covering candidate.
    """
    membership = np.asarray(membership, dtype=bool)
    area = np.asarray(face_area_m2, dtype=float)
    if membership.ndim != 2 or area.shape != (membership.shape[1],) or not len(area):
        raise ValueError("Expected a candidate-by-face matrix and one area per face")
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Source face areas must be finite and positive")
    ids = (np.arange(len(membership)) if candidate_indices is None
           else np.asarray(candidate_indices, dtype=int))
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(membership))):
        raise ValueError("Expected unique valid candidate indices")
    ids = np.sort(ids)
    eligible = membership[ids]
    missing = np.flatnonzero(~eligible.any(axis=0))
    if len(missing):
        raise IncompleteCandidatePool(missing, area[missing].sum())
    sparse = csr_matrix(eligible, dtype=float)
    remaining = np.ones(len(area), dtype=bool)
    selected, trace = [], []
    total = float(area.sum())
    while remaining.any():
        gain = np.asarray(sparse @ (area*remaining)).ravel()
        index = int(ids[int(np.argmax(gain))])
        new = remaining & membership[index]
        if not new.any():
            raise RuntimeError("A coverable face remained without positive greedy gain")
        remaining[new] = False
        selected.append(index)
        full_area = float(area[membership[index]].sum())
        new_area = float(area[new].sum())
        trace.append(dict(candidate_index=index, face_count=int(membership[index].sum()),
            new_face_count=int(new.sum()), area_m2=full_area, area_fraction=full_area/total,
            newly_covered_area_m2=new_area, newly_covered_area_fraction=new_area/total,
            repeated_area_m2=full_area-new_area, remaining_face_count=int(remaining.sum()),
            cumulative_coverage_fraction=float(area[~remaining].sum()/total)))
    return np.asarray(selected, dtype=int), trace


def load_candidate_pool(output_root=DEFAULT_OUTPUT, object_name="B"):
    """Reject stale geometry/config/code before reusing the cached sampler."""
    stage = Path(output_root)/object_name/"step1"
    folder = stage/"step1.3"
    path = folder/"pose_coverage.json"
    source = json.loads(path.read_text())
    bindings = {str(path.resolve()): sha256(path)}
    config = stage/"step1.1/config.json"
    shape = ROOT/"objects"/object_name/"mesh.stl"
    for p, expected in [(config, source['config_sha256']), (shape, source['shape_sha256'])]:
        if sha256(p) != expected:
            raise ValueError(f"Stale candidate pool input: {p}")
        bindings[str(p.resolve())] = expected
    for name, expected in source['input_sha256'].items():
        p = ROOT/name
        if sha256(p) != expected:
            raise ValueError(f"Stale candidate pool policy: {p}")
        bindings[str(p.resolve())] = expected
    for name, expected in source['computation_source_sha256'].items():
        p = Path(__file__).with_name(name)
        if sha256(p) != expected:
            raise ValueError(f"Stale candidate pool computation: {p}")
        bindings[str(p.resolve())] = expected
    archive = folder/source['arrays']
    if sha256(archive) != source['artifact_sha256'][source['arrays']]:
        raise ValueError(f"Changed candidate archive: {archive}")
    bindings[str(archive.resolve())] = sha256(archive)
    keys = ['vertices_object', 'faces', 'source_face_ids', 'face_area_m2',
            'candidate_face_membership', 'representative_candidate_indices',
            'candidate_rotation_world_object', 'candidate_translation_world_object_m',
            'candidate_outward_axis_object']
    with np.load(archive) as z:
        arrays = {key: z[key] for key in keys}
    mesh = load_shape(shape)
    np.testing.assert_array_equal(arrays['vertices_object'], mesh.vertices)
    np.testing.assert_array_equal(arrays['faces'], mesh.faces)
    np.testing.assert_array_equal(arrays['source_face_ids'], np.arange(len(mesh.faces)))
    np.testing.assert_allclose(arrays['face_area_m2'], mesh.area_faces, rtol=1e-12)
    if len(arrays['candidate_face_membership']) != source['candidate_count']:
        raise ValueError("Candidate count does not match archive")
    # The display examples are appended separately by select_pose_coverage.
    # Exclude them, so no manually aimed ear/mouth region influences planning.
    eligible = np.setdiff1d(np.arange(source['candidate_count']),
                           arrays['representative_candidate_indices'])
    return mesh, source, arrays, eligible, bindings


def plan_coverage(output_root=DEFAULT_OUTPUT, object_name="B"):
    # Preserve this public entry point when the selected task uses the new
    # rigid-needle model. Its candidates require perspective geometry.
    config_path = Path(output_root)/object_name/'step1/step1.1/config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get('working_tool', {}).get('model') in ('fixed_length_needle_in_bounded_spherical_sector', 'needle_spherical_sector_surface_intersection'):
            from .run_needle import run
            selection = json.loads((config_path.parent/'selection.json').read_text())
            return run(output_root, object_name, selection['source'], selection['workspace'].get('proposal_inset_fraction', .05))
    start = time.perf_counter()
    mesh, source, pool, eligible, bindings = load_candidate_pool(output_root, object_name)
    selected, trace = greedy_cover(pool['candidate_face_membership'], mesh.area_faces, eligible)
    masks = pool['candidate_face_membership'][selected]
    transforms = np.tile(np.eye(4), (len(selected), 1, 1))
    transforms[:, :3, :3] = pool['candidate_rotation_world_object'][selected]
    transforms[:, :3, 3] = pool['candidate_translation_world_object_m'][selected]
    counts = masks.sum(axis=0)
    # Ownership is bookkeeping for future paint scheduling, not a clipped region.
    owner = masks.argmax(axis=0)
    ids = [f'W{i+1:02d}' for i in range(len(selected))]
    output = stage_directory(output_root, object_name, 'step1.2')
    np.savez_compressed(output/'coverage_plan.npz', vertices_object=mesh.vertices,
        faces=mesh.faces, source_face_ids=np.arange(len(mesh.faces)), face_area_m2=mesh.area_faces,
        candidate_indices=selected, working_area_ids=np.asarray(ids),
        working_area_face_membership=masks, coverage_count=counts,
        first_covering_area_index=owner, screening_world_from_object=transforms,
        outward_axis_object=pool['candidate_outward_axis_object'][selected])
    for i, row in enumerate(trace):
        row.update(id=ids[i], face_ids=np.flatnonzero(masks[i]).tolist(),
                   screening_world_from_object=transforms[i].tolist(),
                   outward_axis_object=pool['candidate_outward_axis_object'][selected[i]].tolist())
    total = float(mesh.area)
    report = dict(schema_version=1, object=object_name,
        algorithm='area_weighted_greedy_set_cover', coverage_target='entire_original_surface',
        scope='Geometric full-surface baseline; downstream four-pose demo is a separate result.',
        candidate_count=len(eligible), excluded_display_candidate_indices=pool['representative_candidate_indices'].tolist(),
        working_area_count=len(selected), requested_working_area_count=None,
        overlap_allowed=True, complete_connected_candidate_regions_preserved=True,
        strict_all_source_faces_required=True, source_face_count=len(mesh.faces),
        surface_area_m2=total, covered_face_count=int((counts > 0).sum()),
        coverage_fraction=float(mesh.area_faces[counts > 0].sum()/total),
        uncovered_source_face_ids=np.flatnonzero(counts == 0).tolist(),
        summed_working_area_m2=float(mesh.area_faces @ counts),
        mean_area_weighted_multiplicity=float(mesh.area_faces @ counts/total),
        maximum_face_multiplicity=int(counts.max()),
        repeated_coverage_area_m2=float(mesh.area_faces @ (counts-1)),
        region_count_globally_minimized=False, fixture_aware_optimization=False,
        screening_alignments_preserved=True, pose_alternatives_optimized=False,
        native_robot_motion_verified=False, support_verified=False,
        workspace=source['workspace'], hole_filling=source['hole_filling'],
        fixed_tool_direction_world=source['fixed_tool_direction_world'],
        fixed_tcp_rotation_world=source['fixed_tcp_rotation_world'],
        working_areas=trace, arrays='coverage_plan.npz',
        computation_elapsed_s=time.perf_counter()-start, input_sha256=bindings,
        computation_source_sha256={'step1_2.py': sha256(__file__)})
    from .coverage_render import render
    report['visualization'] = render(mesh, masks, transforms, report, output)
    report['figures'] = ['working_areas_overview.png', 'coverage_progress.png']
    report['artifact_sha256'] = {name: sha256(output/name)
                               for name in [report['arrays'], *report['figures']]}
    write_json(output/'coverage_plan.json', report)
    from .validate_coverage_plan import validate
    validate(output_root, object_name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = plan_coverage(args.output_root, args.object)
    print(json.dumps(dict(working_area_count=report['working_area_count'],
        coverage_fraction=report['coverage_fraction'],
        candidate_count=report['candidate_count'],
        output=str(Path(args.output_root)/args.object/'step1/step1.2')), indent=2))


if __name__ == '__main__':
    main()
