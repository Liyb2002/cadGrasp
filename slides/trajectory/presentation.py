"""Render the saved, verified B/pose_2 insertion; never rebuild the support."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'tools'))
import slide_scene as S
from step5_connect_support import solids, rigid_path, belt_geometry


def scene():
    domain = S.load()
    folder = S.CASE/'step5_connect_support'
    with np.load(folder/'geometry.npz') as data:
        parts = solids.unpack_parts(data)
        labels = data['part_labels'].tolist()
    path = json.loads((folder/'trajectory.json').read_text())
    assert path['passed'] and path['continuous_sweep_verified']
    rigid_path.replay(belt_geometry.Scene(domain.mesh), parts, path)
    colors = [S.ORANGE if label.startswith('contact_head_') else S.FRAME for label in labels]
    start = min(path['final_withdrawal_amount'], .06/path['length_scale_m'])
    moved = []
    for part in parts:
        copy = part.copy()
        copy.vertices = rigid_path.transform(copy.vertices, path['origin_m'],
            path['length_scale_m'], path['motion'], start)
        moved.append(copy)
    cloud = np.vstack([p.vertices for p in parts+moved])
    ground = cloud[np.abs(cloud[:, 2]) < 1e-8]
    cam = S.camera(domain, size=1100, extra=cloud, ground_points=ground)
    images = [S.render(domain, parts=list(zip(group, colors)), cam=cam, ground_points=ground)[0]
              for group in (moved, parts)]
    return domain, path, start, images


def main():
    domain, path, start, images = scene()
    canvas = Image.new('RGB', (2400, 1440), S.PAPER)
    draw = ImageDraw.Draw(canvas)
    S.text(draw, (1200, 65), 'One connected support slides into place', 46)
    S.text(draw, (1200, 124), 'B / pose 2  |  The workpiece stays fixed', 29, S.MUTED)
    for i, (image, title) in enumerate(zip(images, ('(a) During insertion', '(b) Installed'))):
        canvas.paste(image, (50+1200*i, 205))
        S.text(draw, (600+1200*i, 197), title, 32)
    S.text(draw, (1200, 1330), 'The whole support shares one straight motion; its swept volume stays clear of the workpiece.', 27)
    S.text(draw, (1200, 1390), 'Saved continuous trajectory replayed before rendering.', 25, S.MUTED)
    canvas.save(HERE/'sweep_demo.png')
    fig = plt.figure(figsize=(18, 9), dpi=160, facecolor='white')
    fig.text(.5, .94, 'A common insertion trajectory', ha='center', fontsize=30)
    fig.text(.5, .88, 'B / pose 2', ha='center', fontsize=19, color=S.MUTED)
    fig.text(.05, .69, r'$\mathrm{Sweep}(S,a)=\{x-ta:\ x\in S,\ t\geq0\}$', fontsize=25)
    fig.text(.05, .53, r'$\mathrm{Sweep}(S,a)\cap\mathrm{int}(W)=\varnothing$', fontsize=25)
    fig.text(.05, .38, r'$\mathrm{Sweep}(S,a)\cap\{z<0\}=\varnothing$', fontsize=25)
    fig.text(.05, .23, 'S: the complete rigid support     W: the workpiece', fontsize=18)
    fig.text(.05, .16, 'a: insertion direction; withdrawal moves along -a.', fontsize=17, color=S.MUTED)
    ax = fig.add_axes([.57, .10, .41, .74], facecolor='white'); ax.imshow(images[1]); ax.axis('off')
    fig.savefig(HERE/'sweep_eq.png', facecolor='white', edgecolor='white', transparent=False); plt.close(fig)
    S.record(HERE/'sweep_demo.json', trajectory_replayed=True,
             withdrawal_motion=path['motion'], depicted_withdrawal_amount=start,
             full_withdrawal_amount=path['final_withdrawal_amount'],
             source_geometry=str((S.CASE/'step5_connect_support/geometry.npz').relative_to(S.SLIDES)),
             geometry_rebuilt=False, comparison='During the saved insertion / installed')
    print(HERE/'sweep_demo.png', flush=True)
    print(HERE/'sweep_eq.png', flush=True)


if __name__ == '__main__':
    main()
