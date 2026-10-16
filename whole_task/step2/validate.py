"""Check saved object poses against their source surfaces and fixed station."""
import argparse
import json
from pathlib import Path

import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT, ROOT, sha256, write_json
from .poses import load_inputs, audit_region
from .transforms import rigid_transform


def validate(output_root=DEFAULT_OUTPUT, object_name="B"):
    output = Path(output_root)/object_name/"step2/step2.1"
    path = output/"object_poses.json"
    report = json.loads(path.read_text())
    for name, digest in report["input_sha256"].items():
        assert sha256(name) == digest, name
    for name, digest in report["computation_source_sha256"].items():
        assert sha256(Path(__file__).with_name(name)) == digest, name
    for name, digest in report["artifact_sha256"].items():
        assert sha256(output/name) == digest, name
    mesh, source, upstream, config, _ = load_inputs(output_root, object_name)
    with np.load(output/report["arrays"]) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.testing.assert_array_equal(arrays["vertices_object"], mesh.vertices)
    np.testing.assert_array_equal(arrays["faces"], mesh.faces)
    np.testing.assert_array_equal(arrays["working_area_face_membership"], upstream["working_area_face_membership"])
    np.testing.assert_array_equal(arrays["filled_face_membership"], upstream["working_area_filled_face_membership"])
    station = np.asarray(report["fixed_station"]["world_from_station"])
    np.testing.assert_allclose(station[:3, 3], source["workspace"]["center_world_m"], atol=1e-12)
    np.testing.assert_allclose(station[:3, :3], source["fixed_tcp_rotation_world"], atol=1e-12)
    np.testing.assert_array_equal(arrays["world_from_station"], station)
    assert report["pose_count"] == len(report["poses"]) == len(source["working_areas"])
    summaries = []
    for i, (pose, area) in enumerate(zip(report["poses"], source["working_areas"])):
        assert pose["id"] == area["id"]
        transform = np.asarray(pose["world_from_object"])
        rotation, translation = transform[:3, :3], transform[:3, 3]
        np.testing.assert_array_equal(transform, rigid_transform(rotation, translation))
        np.testing.assert_array_equal(transform, arrays["world_from_object"][i])
        np.testing.assert_array_equal(rotation, area["screening_alignment"]["rotation_world_object"])
        np.testing.assert_array_equal(translation, area["screening_alignment"]["translation_world_object_m"])
        np.testing.assert_allclose(transform @ np.asarray(pose["object_from_world"]), np.eye(4), atol=1e-12)
        np.testing.assert_allclose(station @ np.asarray(pose["station_from_object"]), transform, atol=1e-12)
        expected_points = np.einsum("ij,nj->ni", rotation, mesh.vertices)+translation
        np.testing.assert_allclose(arrays["vertices_world"][i], expected_points, atol=1e-12)
        np.testing.assert_allclose(arrays["face_normals_world"][i], np.einsum("ij,nj->ni", rotation, mesh.face_normals), atol=1e-12)
        np.testing.assert_array_equal(pose["working_face_ids"], area["face_ids"])
        np.testing.assert_array_equal(pose["filled_face_ids"], area["filled_face_ids"])
        np.testing.assert_allclose(rotation @ np.asarray(pose["gravity_object_m_s2"]), [0, 0, -9.81], atol=1e-12)
        audit, passed = audit_region(mesh, transform, area["face_ids"], source, config)
        assert audit == pose["geometry_audit"]
        ids = np.asarray(area["face_ids"])
        np.testing.assert_array_equal(np.flatnonzero(arrays["geometric_pass_face_membership"][i]), ids[passed])
        assert np.isin(area["raw_face_ids"], ids[passed]).all()
        assert np.isin(audit["unresolved_face_ids"], area["filled_face_ids"]).all()
        assert not pose["executable_working_pose_verified"]
        summaries.append(dict(id=pose["id"], source_alignment_preserved=True,
                              selected_faces=len(ids), unresolved_faces=int((~passed).sum())))
    assert (ROOT/"whole_task/step1/step1_2.py").stat().st_size == 0
    result = dict(status="passed", pose_count=len(summaries), poses=summaries,
                  rigid_transforms_and_frames_verified=True,
                  fixed_station_preserved=True, work_regions_preserved=True,
                  geometry_rechecked_including_fill_exceptions=True,
                  native_robot_motion_verified=False, fixture_verified=False,
                  report_sha256=sha256(path), arrays_sha256=sha256(output/report["arrays"]),
                  validator_sha256=sha256(__file__))
    write_json(output/"validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    print(json.dumps(validate(args.output_root, args.object), indent=2))


if __name__ == "__main__":
    main()
