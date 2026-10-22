"""Recheck full coverage and each selected whole region in its witness pose."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from robots_config.constraints import load_config

from .common import DEFAULT_OUTPUT, ROOT, sha256, write_json
from .first_layer import first_layer_faces
from .pose_coverage import sphere_workspace, spatial_workspace_mask, face_components
from .step1_3 import _adjacency
from .step1_2 import load_candidate_pool, greedy_cover


def validate(output_root=DEFAULT_OUTPUT, object_name='B'):
    folder = Path(output_root)/object_name/'step1/step1.2'
    path = folder/'coverage_plan.json'
    report = json.loads(path.read_text())
    if report.get('geometry_model') in ('fixed_length_needle_in_bounded_spherical_sector', 'needle_spherical_sector_surface_intersection'):
        from .validate_needle import validate as validate_needle
        return validate_needle(output_root, object_name)
    for name, digest in report['input_sha256'].items():
        assert sha256(name) == digest, name
    for name, digest in report['computation_source_sha256'].items():
        assert sha256(Path(__file__).with_name(name)) == digest, name
    for name, digest in report['visualization']['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    for name, digest in report['artifact_sha256'].items():
        assert sha256(folder/name) == digest, name
    mesh, source, pool, eligible, _ = load_candidate_pool(output_root, object_name)
    with np.load(folder/report['arrays']) as z:
        arrays = {key: z[key] for key in z.files}
    selected, replay = greedy_cover(pool['candidate_face_membership'], mesh.area_faces, eligible)
    np.testing.assert_array_equal(selected, arrays['candidate_indices'])
    masks = arrays['working_area_face_membership']
    np.testing.assert_array_equal(masks, pool['candidate_face_membership'][selected])
    np.testing.assert_array_equal(arrays['vertices_object'], mesh.vertices)
    np.testing.assert_array_equal(arrays['faces'], mesh.faces)
    np.testing.assert_array_equal(arrays['source_face_ids'], np.arange(len(mesh.faces)))
    np.testing.assert_array_equal(arrays['face_area_m2'], mesh.area_faces)
    np.testing.assert_array_equal(arrays['outward_axis_object'], pool['candidate_outward_axis_object'][selected])
    assert report['candidate_count'] == len(eligible)
    assert not np.isin(selected, pool['representative_candidate_indices']).any()
    assert len(selected) == report['working_area_count'] == len(report['working_areas'])
    assert report['requested_working_area_count'] is None
    counts = masks.sum(axis=0)
    np.testing.assert_array_equal(arrays['coverage_count'], counts)
    np.testing.assert_array_equal(arrays['first_covering_area_index'], masks.argmax(axis=0))
    assert (counts > 0).all()
    assert report['coverage_fraction'] == 1. and report['uncovered_source_face_ids'] == []
    assert report['covered_face_count'] == report['source_face_count'] == len(mesh.faces)
    np.testing.assert_allclose(report['surface_area_m2'], mesh.area, rtol=1e-12)
    np.testing.assert_allclose(report['summed_working_area_m2'], mesh.area_faces @ counts, rtol=1e-12)
    assert report['maximum_face_multiplicity'] == counts.max()
    assert report['complete_connected_candidate_regions_preserved']
    assert report['overlap_allowed'] and not report['fixture_aware_optimization']
    config = load_config(Path(output_root)/object_name/'step1/step1.1/config.json')
    workspace = sphere_workspace(config, source['workspace']['inset_fraction'])
    assert workspace == report['workspace']
    neighbors = _adjacency(mesh)
    axis = np.asarray(report['fixed_tool_direction_world'])
    length = float(config['feasibility']['tcp_transform']['translation_flange_m'][2])
    gap = float(config['task_constraint']['tcp_to_surface_distance_m']['nominal'])
    reach = workspace['base_reach_estimate']
    cache, summaries = {}, []
    for i, (row, index) in enumerate(zip(report['working_areas'], selected)):
        for key, value in replay[i].items():
            assert row[key] == value, (i, key)
        np.testing.assert_array_equal(np.flatnonzero(masks[i]), row['face_ids'])
        assert row['id'] == arrays['working_area_ids'][i] == f'W{i+1:02d}'
        transform = arrays['screening_world_from_object'][i]
        np.testing.assert_array_equal(transform, row['screening_world_from_object'])
        np.testing.assert_array_equal(transform[:3, :3], pool['candidate_rotation_world_object'][index])
        np.testing.assert_array_equal(transform[:3, 3], pool['candidate_translation_world_object_m'][index])
        np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(transform[:3, :3])-1) < 1e-12
        np.testing.assert_allclose(transform[:3, :3] @ arrays['outward_axis_object'][i], -axis, atol=1e-12)
        assert len(face_components(row['face_ids'], neighbors)) == 1
        world = mesh.triangles[masks[i]] @ transform[:3, :3].T+transform[:3, 3]
        assert spatial_workspace_mask(world, workspace).all(), (i, 'finite work envelope')
        flange = world-(length+gap)*axis
        radius = np.linalg.norm(flange-reach['center_world_m'], axis=2).max()
        assert radius <= reach['usable_radius_m']+1e-12, (i, 'flange reach')
        visible = first_layer_faces(mesh, arrays['outward_axis_object'][i], cache)
        assert visible[masks[i]].all(), (i, 'whole-triangle first-hit visibility')
        summaries.append(dict(id=row['id'], face_count=int(masks[i].sum()),
            edge_connected=True, whole_triangles_inside_first_solid_ring=True,
            all_object_occluders_checked=True, complete_region_first_visible=True,
            geometric_flange_reach_passed=True))
    with Image.open(folder/'working_areas_overview.png') as im:
        im.load()
        assert list(im.size) == report['visualization']['overview_size']
    result = dict(status='passed', working_area_count=len(selected), source_face_count=len(mesh.faces),
        surface_coverage_fraction=1., uncovered_face_count=0,
        automatic_pool_only=True, greedy_selection_replayed=True,
        all_regions_complete_and_edge_connected=True,
        screening_alignments_preserved=True, selected_region_geometry_independently_rechecked=True,
        global_minimum_region_count_certified=False, fixture_or_native_robot_certified=False,
        working_areas=summaries, report_sha256=sha256(path), validator_sha256=sha256(__file__))
    write_json(folder/'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate(args.output_root, args.object)
    print(json.dumps({k: v for k, v in result.items() if k != 'working_areas'}, indent=2))
