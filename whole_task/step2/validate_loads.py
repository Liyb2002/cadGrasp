"""Independent force/moment substitution and all-sample geometry audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from .wrenches import WrenchDomain


def check_binding(report, folder):
    for path, digest in report["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Current-pose load input changed: {path}")
    for name, digest in report["artifact_sha256"].items():
        if sha256(folder/name) != digest:
            raise ValueError(f"Load artifact changed: {name}")


def check_case(folder, expected_count):
    data = json.loads((folder/"needs.json").read_text())
    domain = WrenchDomain(data)
    with np.load(folder/"samples.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    with np.load(folder/"forward.npz") as saved:
        forward = {name: saved[name] for name in saved.files}
    params = arrays["parameters"]
    assert params.shape == (expected_count, 5)
    ids = arrays["work_face_index"]
    assert np.issubdtype(ids.dtype, np.integer)
    u, v, theta, phi, magnitude = params.T
    assert np.all(theta == 0) and np.all(phi == 0)
    evaluated = domain.evaluate(ids, u, v, magnitude)
    for key in ("pt_m", "force_push_mg", "need_wrench"):
        np.testing.assert_allclose(arrays[key], evaluated[key], atol=1e-13, rtol=0)
    for key in ("source_face_id", "tool_visible", "face_passes_step2_1_geometry", "geometrically_screened"):
        np.testing.assert_array_equal(arrays[key], evaluated[key])
    assert arrays["need_wrench"].shape == (expected_count, 6)
    force = arrays["force_push_mg"]
    x, y, z = (arrays["pt_m"]-domain.com).T
    fx, fy, fz = force.T
    # Independent scalar component expansion, not the generator's cross product.
    process_moment = np.c_[y*fz-z*fy, z*fx-x*fz, x*fy-y*fx]
    residual = np.c_[arrays["need_wrench"][:, :3]+force+domain.gravity,
                     arrays["need_wrench"][:, 3:]+process_moment]
    np.testing.assert_allclose(residual, 0, atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.einsum("ni,ni->n", force, arrays["need_wrench"][:, 3:]), 0, atol=1e-13, rtol=0)
    weight = domain.weight_N
    np.testing.assert_allclose(arrays["need_wrench_SI"], arrays["need_wrench"]*weight, atol=1e-13, rtol=0)
    world_moment = -np.cross(arrays["pt_m"], force)-np.cross(domain.com, domain.gravity)
    np.testing.assert_allclose(arrays["need_wrench_world_origin"][:, 3:], world_moment, atol=1e-13, rtol=0)
    np.testing.assert_allclose(arrays["need_wrench_world_origin_SI"], arrays["need_wrench_world_origin"]*weight, atol=1e-13, rtol=0)
    transform = np.asarray(data["frame"]["T_world_mesh"])
    np.testing.assert_allclose(arrays["pt_object_m"] @ transform[:3, :3].T+transform[:3, 3], arrays["pt_m"], atol=1e-13, rtol=0)
    for start in (0, 3):
        np.testing.assert_allclose(arrays["need_wrench_object"][:, start:start+3] @ transform[:3, :3].T,
                                   arrays["need_wrench"][:, start:start+3], atol=1e-13, rtol=0)
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    np.testing.assert_array_equal(arrays["wrench_scale"], scale)
    np.testing.assert_allclose(arrays["targets_scaled"], arrays["need_wrench"]*scale, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(arrays["targets_7d"][:, :6], arrays["targets_scaled"])
    assert np.all(arrays["targets_7d"][:, 6] == 0)
    assert float(arrays["weight_per_sample"]) == 1./expected_count
    np.testing.assert_array_equal(forward["work_face_ids"], domain.work_ids)
    features = np.c_[np.ones(expected_count), force, u[:, None]*force, v[:, None]*force]
    compiled = np.einsum("nij,nj->ni", forward["B"][ids], features)
    np.testing.assert_allclose(compiled, arrays["need_wrench"], atol=1e-13, rtol=0)
    fixed_features = np.c_[np.ones(expected_count), magnitude, u*magnitude, v*magnitude]
    np.testing.assert_allclose(np.einsum("nij,nj->ni", forward["A"][ids], fixed_features), compiled, atol=1e-13, rtol=0)
    # Every sample is a convex combination of gravity and its three K-vertex loads.
    vertex_ids = forward["source_vertex_ids"]
    np.testing.assert_array_equal(vertex_ids, np.r_[-1, np.unique(domain.mesh.faces[domain.work_ids])])
    end = forward["endpoint_need_wrench"]
    vertex_force = domain.k*domain.axis
    expected_end = np.c_[np.tile(-domain.gravity-vertex_force, (len(vertex_ids)-1, 1)),
                         -np.cross(domain.mesh.vertices[vertex_ids[1:]]-domain.com, vertex_force)]
    np.testing.assert_allclose(end, np.vstack((np.r_[-domain.gravity, np.zeros(3)], expected_end)), atol=1e-13, rtol=0)
    np.testing.assert_allclose(forward["endpoint_need_wrench_SI"], end*weight, atol=1e-13, rtol=0)
    lookup = np.full(len(domain.mesh.vertices), -1, dtype=int)
    lookup[vertex_ids[1:]] = np.arange(1, len(vertex_ids))
    corner_rows = lookup[domain.mesh.faces[domain.work_ids[ids]]]
    bary = np.c_[1-u-v, u, v]
    maximum = np.einsum("ni,nij->nj", bary, end[corner_rows])
    reconstructed = (1-magnitude[:, None]/domain.k)*end[0]+magnitude[:, None]/domain.k*maximum
    np.testing.assert_allclose(reconstructed, arrays["need_wrench"], atol=1e-13, rtol=0)
    return dict(pose_id=data["pose_id"], sample_count=expected_count,
                max_balance_residual=float(np.abs(residual).max()),
                max_compiled_forward_residual=float(np.abs(compiled-arrays["need_wrench"]).max()),
                geometry_review_samples=int((~arrays["geometrically_screened"]).sum()),
                endpoint_count=len(vertex_ids), all_paired_rows_verified=True,
                SI_and_world_origin_conversion_verified=True, all_visibility_flags_replayed=True,
                fixed_axis_endpoint_identity_verified=True)


def validate_directory(folder):
    folder = Path(folder)
    report = json.loads((folder/"loads.json").read_text())
    check_binding(report, folder)
    source = json.loads(Path(report["pose_source"]).read_text())
    physical = json.loads(Path(report["physical_properties_source"]).read_text())
    with np.load(Path(report["pose_source"]).with_name("object_poses.npz")) as saved:
        vertices_world = saved["vertices_world"].copy()
        faces = saved["faces"].copy()
    assert len(source["poses"]) == len(report["poses"])
    for i, (pose, row) in enumerate(zip(source["poses"], report["poses"])):
        domain = json.loads((folder/row["folder"]/"needs.json").read_text())
        assert pose["id"] == row["id"] == domain["pose_id"]
        np.testing.assert_array_equal(domain["frame"]["T_world_mesh"], pose["world_from_object"])
        np.testing.assert_array_equal(domain["geometry"]["work_face_ids"], pose["working_face_ids"])
        np.testing.assert_array_equal(domain["geometry"]["filled_face_ids"], pose["filled_face_ids"])
        np.testing.assert_array_equal(domain["geometry"]["vertices_m"], vertices_world[i])
        np.testing.assert_array_equal(domain["geometry"]["faces"], faces)
        transform = np.asarray(pose["world_from_object"])
        expected_com = transform[:3, :3] @ physical["com_mesh_frame"]+transform[:3, 3]
        np.testing.assert_allclose(domain["frame"]["moment_origin_m"], expected_com, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(domain["load"]["fixed_direction_world"], source["fixed_station"]["tool_to_surface_axis_world"])
        assert domain["physical_properties"]["mass_kg"] == physical["mass_kg"]
        assert domain["load"]["K"] == report["policy"]["maximum_processing_force_mg"]
        assert row["sample_count"] == report["policy"]["sample_count_per_pose"]
    cases = [check_case(folder/row["folder"], row["sample_count"]) for row in report["poses"]]
    assert sum(case["sample_count"] for case in cases) == report["total_sample_count"]
    result = dict(status="passed", total_sample_count=report["total_sample_count"], cases=cases,
        physical_equilibrium_identity_verified=True, selected_surfaces_not_filtered=True,
        fixture_force_balance_solved=False, robot_path_verified=False,
        report_sha256=sha256(folder/"loads.json"), validator_sha256=sha256(__file__))
    write_json(folder/"validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    result = validate_directory(args.output_root/args.object/"step2/step2.2")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
