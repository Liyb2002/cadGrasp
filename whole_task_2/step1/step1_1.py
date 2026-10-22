"""Step 1.1: use a fixed oblique 25-degree config as the example user input."""
import argparse
import json
from pathlib import Path
from robots_config.constraints import load_config
from .common import DEFAULT_OUTPUT, ROOT, sha256, stage_directory, write_json
from .render import render_configuration


DEFAULT_CONFIG = ROOT / "robots_config/04_kuka_lbr_med14_r820.json"


def select_robot(output_root=DEFAULT_OUTPUT, object_name="B", config_path=None):
    source = DEFAULT_CONFIG if config_path is None else Path(config_path).resolve()
    mode = "fixed_example_of_user_input" if config_path is None else "explicit_user_input"
    config = load_config(source)
    output = stage_directory(output_root, object_name, "step1.1")
    # Preserve the supplied config, including all robot and task parameters.
    snapshot = output / "config.json"
    snapshot.write_bytes(source.read_bytes())
    render_configuration(config, source, output / "result.png")
    selection = dict(
        schema_version=1, stage="step1.1", object=object_name,
        mode=mode, selected_config=config["id"],
        source=str(source), source_sha256=sha256(source),
        config_file="config.json", config_sha256=sha256(snapshot),
        figure="result.png", figure_sha256=sha256(output / "result.png"),
        planning_performed=False,
    )
    write_json(output / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="B")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = select_robot(args.output_root, args.object, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
