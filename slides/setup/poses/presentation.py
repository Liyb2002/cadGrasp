"""Render five current object/pose examples, without rebuilding any pose."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
import slide_scene as S


def render_case(name, pose):
    domain = S.load(name, pose)
    picture, cam, _ = S.render(domain, size=1200)
    page = Image.new('RGB', (1500, 1450), S.PAPER)
    draw = ImageDraw.Draw(page)
    S.text(draw, (750, 65), 'The target pose', 48)
    S.text(draw, (750, 120), S.case_label(domain), 30, S.MUTED)
    page.paste(picture, (150, 160))
    S.text(draw, (750, 1330), 'Green: the working surface', 28)
    S.text(draw, (750, 1387), 'The workpiece is held at this pose while the support is inserted.', 26, S.MUTED)
    out = HERE/name/pose/'pose.png'
    page.save(out)
    if (name, pose) == ('B', 'pose_2'):
        for filename in ('target_pose.png', 'tip_B.png'):
            page.save(HERE/filename)
    print(out, flush=True)
    return page


def main():
    pages = [render_case(name, pose) for name, pose in S.PRESENTATION_CASES]
    S.gallery(HERE/'target_poses.png', pages, 'Working surfaces at five target poses')
    S.record(HERE/'target_pose.json', cases=[list(case) for case in S.PRESENTATION_CASES],
             setup_data_rebuilt=False, placement_trajectory_claimed=False)
    print(HERE/'target_poses.png', flush=True)


if __name__ == '__main__':
    main()
