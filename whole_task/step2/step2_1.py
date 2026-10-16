"""Step2.1: export object poses in the fixed station, with geometry audits."""
import argparse
import json
from pathlib import Path

import numpy as np

from whole_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from whole_task.step1.validate_pose_coverage import validate as validate_step1
from .poses import load_inputs, formalize


def run(output_root=DEFAULT_OUTPUT, object_name="B"):
    validate_step1(output_root, object_name)
    mesh, source, arrays, config, input_hashes = load_inputs(output_root, object_name)
    report, data = formalize(mesh, source, arrays, config)
    output = Path(output_root)/object_name/"step2/step2.1"
    output.mkdir(parents=True, exist_ok=True)
    report["input_sha256"] = input_hashes
    report["computation_source_sha256"] = {name: sha256(Path(__file__).with_name(name))
        for name in ("poses.py", "transforms.py", "step2_1.py")}
    np.savez_compressed(output/report["arrays"], **data)
    report["artifact_sha256"] = {report["arrays"]: sha256(output/report["arrays"])}
    write_json(output/"object_poses.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    report = run(args.output_root, args.object)
    from .validate import validate
    result = validate(args.output_root, args.object)
    print(json.dumps(dict(pose_count=report["pose_count"],
                          validation=result["status"],
                          poses=[dict(id=p["id"], status=p["status"],
                              translation_m=p["translation_world_object_m"],
                              unresolved_faces=len(p["geometry_audit"]["unresolved_face_ids"]))
                                 for p in report["poses"]]), indent=2))


if __name__ == "__main__":
    main()
