"""Explain the green working area with multiple incoming process-force arrows."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'tools'))
import slide_scene as S


def main():
    domain = S.load()
    cam = S.camera(domain, size=1400)
    examples = S.working_forces(domain, cam, count=12)
    parts = S.force_parts(domain, cam, examples)
    picture, _, _ = S.render(domain, cam=cam, parts=parts)
    page = Image.new('RGB', (1700, 1650), S.PAPER)
    draw = ImageDraw.Draw(page)
    S.text(draw, (850, 68), 'The working area', 52)
    S.text(draw, (850, 130), 'Tools apply forces at different positions and directions on the green surface.', 27)
    page.paste(picture, (150, 165))
    S.text(draw, (850, 1535), 'Green: working area     Red arrows: possible process forces', 28)
    S.text(draw, (850, 1600), 'B / pose 2  |  Separate possible loads; arrow lengths are illustrative.', 25, S.MUTED)
    page.save(HERE/'working_area.png')
    S.record(HERE/'working_area.json', sample_indices=examples['sample_indices'],
             arrow_points_m=examples['points'].tolist(),
             arrow_directions=examples['directions'].tolist(),
             sampled_forces_mg=examples['force_push_mg'].tolist(),
             arrow_count=len(examples['points']), simultaneously_applied=False,
             arrow_tips_on_actual_work_surface=True, depth_tested=True)
    print(HERE/'working_area.png', flush=True)


if __name__ == '__main__':
    main()
