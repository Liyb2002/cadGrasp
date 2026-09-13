"""Render the current setup slide from B/pose_2, without rebuilding any pose."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
import slide_scene as S


def main():
    domain = S.load()
    picture, cam, _ = S.render(domain, size=1200)
    S.applied_force(picture, cam, domain)
    page = Image.new('RGB', (1500, 1450), S.PAPER)
    draw = ImageDraw.Draw(page)
    S.text(draw, (750, 65), 'The target pose', 48)
    S.text(draw, (750, 120), 'B / pose 2', 30, S.MUTED)
    page.paste(picture, (150, 160))
    S.text(draw, (750, 1330), 'Green: working surface     Red: applied force on the workpiece', 26)
    S.text(draw, (750, 1387), 'The workpiece is held at this pose while the support is inserted.', 26, S.MUTED)
    for filename in ('target_pose.png', 'tip_B.png'):
        page.save(HERE / filename)
    S.record(HERE / 'target_pose.json', load_face=S.LOAD_FACE,
             setup_data_rebuilt=False, placement_trajectory_claimed=False)
    print(HERE / 'target_pose.png', flush=True)


if __name__ == '__main__':
    main()
