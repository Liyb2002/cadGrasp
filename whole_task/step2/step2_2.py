"""Step2.2: generate paired force/moment requirements from computed poses."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT, ROOT, load_shape, sha256, write_json
from .validate import validate as validate_poses
from .wrenches import build_domain, sample_domain, endpoint_demands

POLICY = Path(__file__).with_name("load_policy.json")
SOURCE_FILES = ("wrenches.py", "step2_2.py", "transforms.py")


def run(output_root=DEFAULT_OUTPUT, object_name="B"):
    validate_poses(output_root, object_name)
    base = Path(output_root)/object_name/"step2"
    pose_path = base/"step2.1/object_poses.json"
    poses = json.loads(pose_path.read_text())
    policy = json.loads(POLICY.read_text())
    mesh_path = ROOT/"objects"/object_name/"mesh.stl"
    physical_path = mesh_path.with_name("meta.json")
    mesh = load_shape(mesh_path)
    physical = json.loads(physical_path.read_text())
    input_hashes = dict(poses["input_sha256"])
    for path in (pose_path, base/"step2.1/object_poses.npz", physical_path, POLICY):
        input_hashes[str(path.resolve())] = sha256(path)
    for name, digest in poses["computation_source_sha256"].items():
        input_hashes[str(Path(__file__).with_name(name))] = digest
    for name in SOURCE_FILES:
        path = Path(__file__).with_name(name)
        input_hashes[str(path)] = sha256(path)
    output = base/"step2.2"
    with tempfile.TemporaryDirectory(prefix=".step2_2_", dir=base) as temporary:
        temporary = Path(temporary)
        rows, artifacts = [], {}
        for pose in poses["poses"]:
            folder = temporary/pose["id"]
            folder.mkdir()
            domain = build_domain(mesh, pose, poses["fixed_station"], physical, policy)
            domain.data["object"] = object_name
            samples = sample_domain(domain, policy["sample_count_per_pose"], policy["sample_seed"])
            endpoints = endpoint_demands(domain)
            write_json(folder/"needs.json", domain.data)
            np.savez_compressed(folder/"samples.npz", **samples)
            np.savez_compressed(folder/"forward.npz", B=domain.B, A=domain.A,
                                work_face_ids=domain.work_ids, **endpoints)
            metadata = dict(pose_id=pose["id"], count=policy["sample_count_per_pose"], seed=policy["sample_seed"],
                pair_order=policy["reaction_target_order"], parameter_order=["u", "v", "theta_rad", "phi_rad", "magnitude_mg"],
                direction_parameters="theta=phi=0 relative to fixed world needle axis; not relative to each face normal",
                moment_origin_m=domain.com.tolist(), weight_per_sample=1./policy["sample_count_per_pose"],
                magnitude_range_mg=[0., domain.k], distribution="surface area x uniform force magnitude; fixed direction",
                geometry_rejection_sampling=False, all_selected_faces_retained=True,
                geometry_review_sample_count=int((~samples["geometrically_screened"]).sum()),
                continuous_forward=dict(general="b=B_f z, z=(1,P,uP,vP)", fixed_axis="b=A_f y, y=(1,m,um,vm)",
                    compiled_from="current pose geometry and physical constants only", fitted_from_samples=False,
                    parameter_derivative_order=["u", "v", "magnitude_mg"]),
                endpoint_certificate=dict(count=len(endpoints["source_vertex_ids"]),
                    row_zero="pure gravity", other_rows="maximum processing force at each selected source vertex",
                    identity="b(q,m)=(1-m/K)*b_gravity+(m/K)*sum_i(bary_i*b(vertex_i,K))",
                    scope="sufficient continuous equilibrium targets for a fixed convex reaction set; tool access remains separate"),
                downstream=dict(scale=samples["wrench_scale"].tolist(), scaled_targets="samples.npz:targets_scaled",
                    no_uplift_targets="samples.npz:targets_7d", seventh_coordinate=0,
                    whole_fixture_verified=False, legacy_native_pose_reader_compatible=False),
                artifacts={name: sha256(folder/name) for name in ("needs.json", "samples.npz", "forward.npz")})
            write_json(folder/"sample_metadata.json", metadata)
            for name in ("needs.json", "samples.npz", "forward.npz", "sample_metadata.json"):
                artifacts[f"{pose['id']}/{name}"] = sha256(folder/name)
            rows.append(dict(id=pose["id"], folder=pose["id"], sample_count=policy["sample_count_per_pose"],
                work_face_count=len(domain.work_ids), filled_face_count=len(pose["filled_face_ids"]),
                unresolved_source_face_count=len(pose["geometry_audit"]["unresolved_face_ids"]),
                sample_geometry_review_count=metadata["geometry_review_sample_count"],
                status="provisional_geometric_exceptions" if metadata["geometry_review_sample_count"] else "geometrically_screened_loads",
                com_world_m=domain.com.tolist(), mass_kg=physical["mass_kg"], weight_N=domain.weight_N,
                maximum_processing_force_N=domain.k*domain.weight_N,
                maximum_required_moment_norm_Nm=float(np.linalg.norm(endpoints["endpoint_need_wrench_SI"][:, 3:], axis=1).max()),
                example=dict(sample_index=0, point_world_m=samples["pt_m"][0].tolist(),
                    processing_force_N=(samples["force_push_mg"][0]*domain.weight_N).tolist(),
                    required_force_N=samples["need_wrench_SI"][0, :3].tolist(),
                    required_moment_Nm=samples["need_wrench_SI"][0, 3:].tolist(),
                    geometry_review_required=not bool(samples["geometrically_screened"][0])),
                continuous_endpoint_count=len(endpoints["source_vertex_ids"])))
            print(f"Step2.2 {pose['id']}: {len(samples['need_wrench'])} paired demands; "
                  f"{metadata['geometry_review_sample_count']} samples retain geometry-review flags", flush=True)
        report = dict(schema_version=1, object=object_name, stage="step2.2", computed_from="current_step2.1_object_poses",
            policy=policy, pose_count=len(rows), total_sample_count=sum(r["sample_count"] for r in rows), poses=rows,
            old_fixed_pose_load_arrays_read=False, original_objects_and_slides_outputs_modified=False,
            force_and_moment_sampled_independently=False, gravity_retained=True,
            moment_origin="per-pose world COM", input_sha256=input_hashes, artifact_sha256=artifacts,
            pose_source=str(pose_path.resolve()), physical_properties_source=str(physical_path.resolve()),
            reference_implementations=["codes/precompute_objects/loads.py:demand",
                "objects/r6_quadratic.py:QuadraticR6", "slides/Co-optimize/helper_func/current_task.py",
                "slides/Co-optimize/helper_func/legacy_kernel/step3_scheculer/passive_support.py:target"],
            scope="mechanical demand generation and algebraic verification; not fixture equilibrium or robot execution certification")
        write_json(temporary/"loads.json", report)
        from .validate_loads import validate_directory
        validate_directory(temporary)
        output.mkdir(exist_ok=True)
        for source in sorted(temporary.rglob("*")):
            if source.is_file() and source.name != "loads.json":
                destination = output/source.relative_to(temporary)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
        (temporary/"loads.json").replace(output/"loads.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    report = run(args.output_root, args.object)
    print(json.dumps(dict(stage=report["stage"], total_sample_count=report["total_sample_count"], poses=report["poses"]), indent=2))


if __name__ == "__main__":
    main()
