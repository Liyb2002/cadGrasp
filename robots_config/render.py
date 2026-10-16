"""Draw mechanical arms and their allowed tool direction angles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
try:
    from .constraints import HERE, load_configs
except ImportError:
    from constraints import HERE, load_configs


def font(size, bold=False):
    suffix = '-Bold' if bold else ''
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf', size)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE)
    args = parser.parse_args()
    configs = load_configs(args.directory)
    if len(configs) != 10:
        parser.error('Expected ten configurations')
    blender = shutil.which('blender')
    if not blender:
        parser.error('Blender is required for the robot illustrations')
    output = args.directory / 'figures'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='robot_direction_figures_') as temp:
        work = Path(temp)
        subprocess.run([blender, '-b', '-t', '8', '--factory-startup', '--python-exit-code', '1',
                        '--python', str(HERE / 'render_arms_blender.py'), '--',
                        '--directory', str(args.directory.resolve()), '--output', str(work)], check=True)
        entries, posters = [], []
        for config in configs:
            raw = Image.open(work / (config['id'] + '.png')).convert('RGB')
            poster = Image.new('RGB', (1500, 1120), 'white')
            poster.paste(raw, (0, 120))
            draw = ImageDraw.Draw(poster)
            draw.text((70, 37), config['name'], font=font(43, True), fill='#213348')
            coordinates = json.loads((work / (config['id'] + '.json')).read_text())
            point = coordinates['angle_label_px']
            label = f'±{config["task_constraint"]["cone_half_angle_deg"]:g}°'
            draw.text((point[0], point[1] + 120), label, font=font(39, True), fill='#d85d26',
                      anchor='mm', stroke_width=5, stroke_fill='white')
            point = coordinates['distance_label_px']
            distance = config['task_constraint']['world_position_constraint']['base_to_center_distance_m']
            draw.text((point[0], point[1] + 120), f'{distance:.2f} m', font=font(33, True), fill='#346d9d',
                      anchor='mm', stroke_width=4, stroke_fill='white')
            path = output / (config['id'] + '.png')
            poster.save(path)
            posters.append(poster)
            entries.append(dict(config=config['id']+'.json', config_sha256=sha(args.directory/(config['id']+'.json')),
                                figure='figures/'+path.name, figure_sha256=sha(path)))
        overview = Image.new('RGB', (1800, 3360), 'white')
        for i, poster in enumerate(posters):
            overview.paste(poster.resize((900, 672), Image.Resampling.LANCZOS), ((i % 2)*900, (i // 2)*672))
        overview.save(output/'overview.png')
    manifest = dict(entries=entries, overview='figures/overview.png', renderer_sha256=sha(__file__),
                    visual_style='robot, assumed surface disk, direction cone, base-to-surface dimension; title, half-angle and one distance label',
                    geometry_units='metres; station, standoff, tool length and work disk follow configuration values; arm links remain illustrative',
                    robot_geometry='Parametric 6/7-axis illustrative geometry, not manufacturer CAD or an IK solution.',
                    blender_renderer_sha256=sha(HERE/'render_arms_blender.py'),
                    graphics_helpers_sha256=sha(HERE/'arm_visuals.py'))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Rendered ten robot illustrations and overview.png', flush=True)


if __name__ == '__main__': main()
