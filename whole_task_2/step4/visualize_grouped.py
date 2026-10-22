"""Arrange the actual execution renders into columns sharing one installation."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from whole_task_2.step1.common import sha256, write_json


def render(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    groups = report['support_installations']['groups']
    source = folder/'execution_overview.png'
    sheet = Image.open(source).convert('RGB')
    width, height, header = 720, 666, 56
    canvas = Image.new('RGB', (width*len(groups), header+height*max(len(g['working_areas']) for g in groups)), 'white')
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    for column, group in enumerate(groups):
        pen.text((column*width+18, 15), f"Support {column+1}: "+' / '.join(group['working_areas']), font=font, fill='#434b53')
        for row, pose in enumerate(group['working_areas']):
            tile = report['poses'].index(pose)+1
            x, y = (tile % 4)*width, (tile//4)*height
            canvas.paste(sheet.crop((x, y, x+width, y+height)), (column*width, header+row*height))
    canvas.save(folder/'grouped_overview.png')
    write_json(folder/'grouped_visualization.json', dict(
        column_definition='identical full centered world_from_support transform',
        groups=groups, source_sha256=sha256(source), figure_sha256=sha256(folder/'grouped_overview.png'),
        source_code_sha256=sha256(__file__), actual_complete_fixture=True, working_area_color='pink'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    render(parser.parse_args().folder)
