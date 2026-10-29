"""Replay the serialized full fixture with freshly rebuilt continuous poses."""
import argparse
import json
from pathlib import Path
import numpy as np

from subdivide_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.step1.first_layer import first_layer_faces
from .angular_model import AngularTaskModel, AngularLayout
from .validate_mobile import validate_artifacts


def validate(folder):
    folder = Path(folder)
    path = folder/'report.json'
    report = json.loads(path.read_text())
    for name, digest in report['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Changed continuous-run input: '+name)
    for name, digest in report['artifact_sha256'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Changed continuous-run artifact: '+name)
    model = AngularTaskModel(name=report['object'], output_root=report['input_root'])
    model.require_connected_core = False
    with np.load(folder/'step4.2/layout.npz') as z:
        layout = AngularLayout.from_arrays(z)
    model.activate(layout)
    assert len(layout.active) == len(model.poses) == report['pose_count'] == 13
    assert tuple(layout.active) == tuple(range(13))
    np.testing.assert_array_equal(layout.angles_deg, report['final_axial_angles_deg'])
    family = json.loads(model.family_source.read_text())
    union = np.zeros(len(model.mesh.faces), bool)
    paired_rows = 0
    for k in layout.active:
        task = model.tasks[k]; row = family['pose_families'][k]
        np.testing.assert_array_equal(task.domain.work_ids, row['working_face_ids'])
        union[task.domain.work_ids] = True
        reference = np.asarray(row['reference_world_from_object'])
        axis = model.rotation_axis; angle = np.radians(layout.angles_deg[k])
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        a = np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*skew
        np.testing.assert_allclose(model.native[k, :3, :3], a @ reference[:3, :3], atol=1e-12)
        outward = -axis @ model.native[k, :3, :3]
        assert first_layer_faces(model.mesh, outward)[task.domain.work_ids].all()
        u, v, theta, phi, magnitude = task.parameters.T
        values = task.domain.evaluate(task.work_face_index, u, v, magnitude,
                                      check_visibility=False, theta=theta, phi=phi)
        force = values['force_push_mg']
        x, y, z = (values['pt_m']-task.domain.com).T
        fx, fy, fz = force.T
        raw = np.c_[-force-task.domain.gravity,
                    -(y*fz-z*fy), -(z*fx-x*fz), -(x*fy-y*fx)]
        np.testing.assert_allclose(raw*task.scale, task.targets, atol=1e-13, rtol=0)
        cosine = np.einsum('ij,ij->i', values['force_direction_world'], task.domain.inward[task.work_face_index])
        assert cosine.min() >= np.cos(task.domain.half_angle)-1e-12
        assert np.linalg.norm(force, axis=1).max() <= task.domain.k+1e-12
        paired_rows += len(raw)
    coverage = float(model.mesh.area_faces[union].sum()/model.mesh.area)
    np.testing.assert_allclose(coverage, report['selected_surface_union_fraction'], atol=1e-12)
    result = validate_artifacts(model, folder, layout)
    with np.load(folder/'step4.2/force_certificate.npz') as z:
        for k in layout.active:
            assert z[f'sample_mask_{k}'].shape == (32768,) and z[f'sample_mask_{k}'].all()
            assert len(z[f'endpoint_mask_{k}']) == report['loads'][model.poses[k]]['endpoint_count']
            assert z[f'endpoint_mask_{k}'].all()
    assert paired_rows == report['total_original_sample_count'] == 425984
    result.update(status='passed', working_area_count=13,
        continuous_pose_family_members_verified=True, selected_working_face_ids_preserved=True,
        selected_surface_union_fraction=coverage,
        all_selected_triangles_first_visible=True,
        all_new_pose_paired_loads_independently_recomputed=True, original_paired_load_count=paired_rows,
        contact_patches_may_be_disconnected_but_complete_fixture_is_connected=True,
        native_robot_IK_certified=False, report_sha256=sha256(path), validator_sha256=sha256(__file__))
    write_json(folder/'validation.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', nargs='?', type=Path, default=DEFAULT_OUTPUT/'B/step4/continuous13')
    a = p.parse_args()
    result = validate(a.folder)
    print(json.dumps({k: v for k, v in result.items() if k not in ['execution', 'complete_geometry', 'complete_path_overlap_m3']}, indent=2))
