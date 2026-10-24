"""Render one text-free Step1.3 composite and remove superseded stage images."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageColor, ImageOps

from robots_config.constraints import load_config
from .common import DEFAULT_OUTPUT, ROOT, sha256, write_json
from .render import COLORS

FIGURE = "working_areas_overview.png"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}


def remove_old_figures(output, keep=()):
    keep = set(keep)
    removed = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path not in keep:
            removed.append(str(path.relative_to(output)))
            path.unlink()
    return removed


def compose_working_areas(config, report, output):
    output = Path(output)
    if len(report["working_areas"]) != 4:
        raise ValueError("The requested composite needs four working areas")
    removed = remove_old_figures(output, keep=[output/FIGURE])
    with tempfile.TemporaryDirectory(prefix="cadgrasp_working_areas_") as directory:
        temporary = Path(directory)
        context = temporary/"context.json"
        colors = [tuple(value/255 for value in ImageColor.getrgb(color)) for color in COLORS]
        write_json(context, dict(config=config, report=report, colors=colors))
        script = Path(__file__).with_name("render_working_areas_blender.py")
        log = temporary/"render.log"
        with log.open("w") as stream:
            result = subprocess.run(["blender", "-b", "-t", "8", "--factory-startup", "--python-exit-code", "1",
                                     "--python", str(script), "--", "--context", str(context),
                                     "--arrays", str((output/"pose_coverage.npz").resolve()),
                                     "--output", str(temporary), "--python-site-packages", str(Path(np.__file__).parent.parent)],
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(log.read_text()[-5000:])
        canvas = Image.new("RGB", (4096, 2560), "white")
        for ordinal in range(4):
            row, column = divmod(ordinal, 2)
            x, y = column*2048, row*1280
            for kind, bounds in [("area", (x+28, y+232, 816, 816)),
                                  ("robot", (x+844, y+24, 1204, 1232))]:
                with Image.open(temporary/f"{ordinal:02d}_{kind}.png") as raw:
                    image = ImageOps.contain(raw.convert("RGB"), bounds[2:], Image.Resampling.LANCZOS)
                position = (bounds[0]+(bounds[2]-image.width)//2,
                            bounds[1]+(bounds[3]-image.height)//2)
                canvas.paste(image, position)
        canvas.save(temporary/FIGURE)
        (temporary/FIGURE).replace(output/FIGURE)
        metadata = json.loads((temporary/"render_metadata.json").read_text())
    report["figures"] = [FIGURE]
    report["visualization"] = dict(layout="2x2_groups_each_with_surface_and_fixed_cone_closeups",
                                   no_text=True, pixel_size=[4096, 2560], scenes=metadata,
                                   one_color_per_region=True,
                                   color_includes_selected_area_and_initialization_hole_fills=True,
                                   needle_direction_fixed_across_all_faces_and_examples=True,
                                   work_ball_drawn=False,
                                   context_view="fixed_cone_tool_end_and_enlarged_tabletop",
                                   gold_cone_is_fixed_spatial_selection_constraint=True,
                                   gold_cone_matches_selection_nominal_boundary=True,
                                   gold_cone_is_spray_fan=False,
                                   schematic_robot_posture_not_ik_solved=True,
                                   removed_previous_figures=removed)
    report["visualization_source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in [
        Path(__file__), Path(__file__).with_name("render_working_areas_blender.py"), ROOT/"robots_config/arm_visuals.py"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    stage = args.output_root/args.object/"step1"
    output = stage/"step1.3"
    report_path = output/"pose_coverage.json"
    report = json.loads(report_path.read_text())
    config_path = stage/"step1.1/config.json"
    if sha256(config_path) != report["config_sha256"] or sha256(output/report["arrays"]) != report["artifact_sha256"][report["arrays"]]:
        raise ValueError("Saved Step1.3 inputs changed; refresh Step1.3 first")
    report = compose_working_areas(load_config(config_path), report, output)
    report["artifact_sha256"] = {name: sha256(output/name) for name in [report["arrays"], *report["figures"]]}
    write_json(report_path, report)
    print(str(output/FIGURE), flush=True)


if __name__ == "__main__":
    main()
