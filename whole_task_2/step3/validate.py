"""Check current input bindings, serialized support, contacts and full loads."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh

from whole_task_2.step1.common import DEFAULT_OUTPUT, sha256, write_json
from .initialize import register, supply
from .work_access import RegisteredWorkVolumes, SurfaceSegments, read_source_cone
from .legacy import kernel as K
from whole_task_2.common.wrenches import endpoint_demands
from whole_task_2.common.force_cone import certificate_description


def validate_pose_access(pose, task, transform, support, access, tolerance):
    """Check each installed support against its own finite world-space volume."""
    family = access.for_pose(pose)
    world = SurfaceSegments(task.domain.mesh, task.domain.work_ids, access.head_world)
    np.testing.assert_allclose(family.head @ transform[:3, :3].T+transform[:3, 3],
                               world.head, atol=1e-14, rtol=0)
    np.testing.assert_allclose(family.cells @ transform[:3, :3].T+transform[:3, 3],
                               world.cells, atol=1e-14, rtol=0)
    # Every original triangle, including filled faces, is an exact finite end.
    for t in (0., .37, 1.):
        q = (1-t)*world.head+t*world.triangles.reshape(-1, 3)
        assert world.contains_points(q).all()
    installed_mesh = support.copy()
    installed_mesh.apply_transform(transform)
    overlap = K.material_volume(K.S.solid(installed_mesh) ^ world.solid())
    assert overlap <= tolerance, (pose, "own finite work volume overlap", overlap)
    return dict(pose=pose, own_installation_checked=True,
                head_world_m=world.head.tolist(), work_face_count=len(world.face_ids),
                own_world_access_overlap_m3=overlap, own_world_access_clear=True,
                segment_parameter_range=[0., 1.], full_working_triangles_included=True,
                extends_beyond_surface=False, extends_behind_head=False)


def check_binding(folder, report):
    for filename, digest in report["input_sha256"].items():
        if sha256(filename) != digest:
            raise ValueError(f"Step3 input changed; regenerate initialization: {filename}")
    for filename, digest in report["artifact_sha256"].items():
        if sha256(folder/filename) != digest:
            raise ValueError(f"Step3 artifact changed: {filename}")


def validate_directory(folder, recheck_force=False):
    folder = Path(folder)
    report = json.loads((folder/"report.json").read_text())
    check_binding(folder, report)
    mesh, states, registration, _ = register(report["output_root"], report["object"], report["poses"])
    assert registration == report["registration"]
    policy = report["policy"]
    saved_object = trimesh.load(folder/"step3.1/registered_object.obj", force="mesh", process=False)
    np.testing.assert_array_equal(saved_object.vertices, mesh.vertices)
    np.testing.assert_array_equal(saved_object.faces, mesh.faces)
    support = trimesh.load(folder/"step3.1/wrapped_support.obj", force="mesh", process=False)
    assert len(support.faces) == 0 or (support.is_watertight and support.is_winding_consistent)
    source_cone = read_source_cone(report["output_root"], report["object"])
    access = RegisteredWorkVolumes(mesh, states, source_cone)
    assert report["work_access"] == access.definition()
    assert json.loads((folder/"step3.1/work_access.json").read_text()) == access.definition()
    with np.load(folder/"step3.1/work_access.npz") as saved:
        np.testing.assert_array_equal(saved["pose_ids"], list(states))
        np.testing.assert_array_equal(saved["head_world_m"], access.head_world)
        np.testing.assert_array_equal(saved["segment_parameter_range"], [0., 1.])
        np.testing.assert_array_equal(saved["T_fixture_to_world"], [t for _, t, _ in states.values()])
        np.testing.assert_array_equal(saved["heads_material_m"], [f.head for f in access.families])
        np.testing.assert_array_equal(saved["face_offsets"], np.r_[0, np.cumsum([len(f.face_ids) for f in access.families])])
        np.testing.assert_array_equal(saved["working_face_ids"], np.concatenate([f.face_ids for f in access.families]))
        np.testing.assert_array_equal(saved["cells_material_m"], np.concatenate([f.cells for f in access.families]))
    forbidden = access.combined_solid()
    solid = K.S.solid(support)
    overlap = K.material_volume(solid ^ forbidden)
    object_overlap = K.material_volume(solid ^ K.S.solid(mesh))
    assert max(overlap, object_overlap) <= policy["boolean_volume_tolerance_m3"]
    with np.load(folder/"step3.1/contacts.npz") as saved:
        triangles, sources = saved["triangles_material_m"], saved["source_face_ids"]
        allowed = np.setdiff1d(np.arange(len(mesh.faces)), access.ids)
        np.testing.assert_array_equal(saved["allowed_face_ids"], allowed)
        np.testing.assert_array_equal(saved["excluded_work_face_ids"], access.ids)
    rebuilt_triangles, rebuilt_sources = K.contact_boundary(mesh, support, allowed)
    np.testing.assert_array_equal(sources, rebuilt_sources)
    np.testing.assert_allclose(triangles, rebuilt_triangles, atol=1e-13, rtol=0)
    assert not np.isin(sources, access.ids).any()
    rows = []
    for pose, (task, transform, _) in states.items():
        own_access = validate_pose_access(pose, task, transform, support, access,
                                         policy["boolean_volume_tolerance_m3"])
        full, floor = supply(task, transform, triangles, sources, policy)
        record = json.loads((folder/f"step3.1/{pose}/force.json").read_text())
        assert record["floor"] == floor
        assert record['continuous_cover'] == certificate_description(task.domain)
        with np.load(folder/f"step3.1/{pose}/coverage.npz") as saved:
            expected_targets = K.U.target(endpoint_demands(task.domain)['endpoint_need_wrench']*task.scale)
            np.testing.assert_allclose(saved['endpoint_targets_7d'], expected_targets, atol=1e-13, rtol=0)
            np.testing.assert_array_equal(full, saved["supply_7d"])
            assert len(saved["mask"]) == len(task.targets)
            assert int(saved["mask"].sum()) == record["covered_samples"]
            assert int(saved["endpoint_mask"].sum()) == record["covered_endpoints"]
            if recheck_force and record["status"] != "unresolved":
                targets = np.vstack((task.targets_7d, saved["endpoint_targets_7d"]))
                mask, _ = K.J.classify(full, targets)
                np.testing.assert_array_equal(mask, np.r_[saved["mask"], saved["endpoint_mask"]])
        rows.append(dict(pose=pose, all_original_load_count=len(task.targets),
                         force_status=record["status"], force_replayed=recheck_force and record["status"] != "unresolved",
                         work_access=own_access,
                         serialized_contact_generators_verified=True))
    visualization = json.loads((folder/"step3.1/visualization.json").read_text())
    assert visualization["pose_order"] == list(states)
    assert visualization["separate_work_exclusion_stage"] is False
    assert report["step3_2_removed"] is True
    assert not (folder/"step3.2").exists()
    result = dict(status="passed", report_sha256=sha256(folder/"report.json"),
                  continuous_work_overlap_m3=overlap, object_overlap_m3=object_overlap,
                  contacts_rebuilt_from_serialized_boundary=True, all_current_inputs_bound=True,
                  execution_access_checked_per_pose=True,
                  segment_parameter_range=[0., 1.], poses=rows,
                  initialization_verified=True, full_fixture_accepted=False)
    write_json(folder/"validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    parser.add_argument("--recheck-force", action="store_true")
    args = parser.parse_args()
    from whole_task_2.step1.common import ROOT
    membranes=ROOT/'whole_task_2/data'/args.object/'step3_membranes'
    if (membranes/'report.json').exists():
        from .run_membranes import validate
        result=validate(membranes)
        print('STEP3 MEMBRANE VALIDATION',result['status'],result['independent_fixture_count'])
        return result
    result = validate_directory(args.output_root/args.object/"step3", args.recheck_force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
