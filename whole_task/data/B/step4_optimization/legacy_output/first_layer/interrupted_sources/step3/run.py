"""Initialize shared support outside finite working-surface-to-head segments."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import time
import numpy as np
import trimesh

from whole_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from .legacy import kernel as K, loaded_sources, COOPT
from .initialize import register, build_seed, force_diagnostics
from .render import render
from .validate import validate_directory
from .work_access import read_source_cone
from .export_geometry import condition_wrap

POLICY = Path(__file__).with_name("policy.json")


def run(output_root=DEFAULT_OUTPUT, object_name="B", pose_ids=None):
    began = time.monotonic()
    output_root = Path(output_root).resolve()
    mesh, states, registration, loads = register(output_root, object_name, pose_ids)
    policy = json.loads(POLICY.read_text())
    source_cone = read_source_cone(output_root, object_name)
    wrapped, allowed, access, geometry = build_seed(mesh, states, policy, source_cone)
    base = output_root/object_name
    destination = base/"step3"
    with tempfile.TemporaryDirectory(prefix=".step3_", dir=base) as temporary:
        temporary = Path(temporary)
        step31 = temporary/"step3.1"
        step31.mkdir()
        K.D.export_exact_obj(mesh, step31/"registered_object.obj")
        wrapped, conditioning = condition_wrap(mesh, wrapped, states, access,
            step31/"wrapped_support.obj", policy['boolean_volume_tolerance_m3'])
        actual_solid = K.S.solid(wrapped)
        geometry.update(export_conditioning=conditioning,
            material_volume_cm3=K.material_volume(actual_solid)*1e6,
            work_cut_material_volume_cm3=geometry['uncut_material_volume_cm3']-K.material_volume(actual_solid)*1e6,
            continuous_work_overlap_m3=conditioning['continuous_work_overlap_m3'],
            object_overlap_m3=conditioning['object_overlap_m3'],
            boundary_component_count=len(actual_solid.decompose()),
            positive_boundary_component_count=sum(part.volume()>0 for part in actual_solid.decompose()))
        # Rebuild contacts on the actual serialized boundary after any recut.
        triangles, sources = K.contact_boundary(mesh, wrapped, allowed)
        np.savez_compressed(step31/"contacts.npz", triangles_material_m=triangles, source_face_ids=sources,
                            allowed_face_ids=allowed, excluded_work_face_ids=access.ids)
        rows = force_diagnostics(states, triangles, sources, policy, step31)
        print("Step3: rendering the updated support; no separate Step3.2", flush=True)
        visualization = render(mesh, wrapped, states, temporary)
        write_json(step31/"visualization.json", visualization)
        write_json(step31/"work_access.json", access.definition())
        np.savez_compressed(step31/"work_access.npz", pose_ids=np.asarray(list(states)),
            head_world_m=access.head_world, segment_parameter_range=np.array([0., 1.]),
            T_fixture_to_world=np.asarray([transform for _, transform, _ in states.values()]),
            heads_material_m=np.asarray([family.head for family in access.families]),
            face_offsets=np.r_[0, np.cumsum([len(f.face_ids) for f in access.families])],
            working_face_ids=np.concatenate([f.face_ids for f in access.families]),
            cells_material_m=np.concatenate([f.cells for f in access.families]))
        inputs = dict(loads["input_sha256"])
        load_folder = base/"step2/step2.2"
        inputs[str(load_folder/"loads.json")] = sha256(load_folder/"loads.json")
        inputs.update({str(load_folder/name): digest for name, digest in loads["artifact_sha256"].items()})
        for path in [POLICY, COOPT/"step3.1/step31.py", *loaded_sources()]:
            inputs[str(path)] = sha256(path)
        report = dict(schema="whole_task_finite_work_access_wrap_v5", stage="step3", status="initialized",
            object=object_name, output_root=str(output_root), poses=list(states), registration=registration,
            policy=policy, geometry=geometry, work_access=access.definition(), state_results=rows,
            working_union_face_count=len(access.ids), contact_triangle_count=len(triangles),
            actual_contact_area_m2=float(trimesh.triangles.area(triangles).sum()) if len(triangles) else 0.,
            all_original_load_count=sum(len(task.targets) for task, _, _ in states.values()),
            all_original_loads_retained=True, legacy_fixed_pose_inputs_read=False,
            execution_access_policy="each installed support avoids only its own working-surface-to-head segments",
            aggregated_access_scope="common-material initialization after each installation's own access-volume pullback",
            work_access_artifact="step3.1/work_access.json", step3_2_removed=True,
            initial_force_status="unresolved" if any(row["status"].startswith("unresolved") for row in rows)
                else "pass" if all(row["status"] == "pass" for row in rows) else "fail",
            source_geometry_review_required=any(task.requires_geometry_review for task, _, _ in states.values()),
            force_diagnostics_gate_initialization=False, step4_ready=True,
            full_fixture_accepted=False, exit_paths_checked=False, base_constructed=False,
            connected_material_selected=False, finite_tool_body_checked=False,
            algorithm_reference="slides/Co-optimize/step3.1/step31.py; user-directed finite working-surface-to-head segments",
            adaptations=["computed Step2 pose/load inputs", "only finite segments between each selected surface and the existing head-side endpoint",
                         "actual airborne workpieces have no floor reactions", "continuous local-normal force cones checked using conservative outer-cover targets"],
            legacy_step33_not_run="historical perimeter seeds; current Co-optimize pipeline constructs the base later",
            seconds=time.monotonic()-began, input_sha256=inputs,
            artifact_sha256={str(path.relative_to(temporary)):sha256(path)
                for path in sorted(temporary.rglob("*")) if path.is_file()})
        write_json(temporary/"report.json", report)
        print("Step3: checking serialized geometry against finite access segments and contact generators", flush=True)
        validate_directory(temporary)
        destination.mkdir(exist_ok=True)
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "report.json":
                target = destination/path.relative_to(temporary)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
        (temporary/"report.json").replace(destination/"report.json")
        # The user removed this stage. Delete only its superseded generated
        # outputs after the replacement has passed validation.
        obsolete = destination/"step3.2"
        if obsolete.exists():
            shutil.rmtree(obsolete)
        for filename in ("station_cone.json", "station_cone.npz"):
            (destination/"step3.1"/filename).unlink(missing_ok=True)
    print(f"Step3 complete: {report['initial_force_status']} initial contact diagnostics; "
          f"{report['all_original_load_count']} original demands", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    parser.add_argument("--poses", nargs="+")
    args = parser.parse_args()
    run(args.output_root, args.object, args.poses)


if __name__ == "__main__":
    main()
