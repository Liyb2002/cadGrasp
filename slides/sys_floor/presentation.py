"""Current B/pose_2 floor diagrams, with independently computed Y-up landings."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
import slide_scene as S


def landings(q, push, com):
    """Balance vertical force and both horizontal moments about the origin.

    This necessary aggregate footprint test does not certify friction or yaw.
    Forces are in object body weights; positions are metres, world Y-up.
    """
    weight = np.array([0., -1., 0.])
    total_force = np.asarray(push) + weight
    moment = np.cross(com, weight) + np.cross(q, push)
    normal = -total_force[..., 1]
    if np.any(normal <= 0):
        raise ValueError('A positive floor normal is required')
    p = np.zeros_like(moment)
    p[..., 0] = -moment[..., 2]/normal
    p[..., 2] = moment[..., 0]/normal
    residual = moment + np.cross(p, -total_force)
    assert np.max(np.abs(residual[..., [0, 2]])) < 1e-12
    return p, normal


def floor_cloud():
    domain = S.load()
    sample = S.samples(domain)
    points, normal = landings(sample['q'], sample['push'], domain.com)
    saved = np.load(S.CASE / 'step4_floor_contact/floor_contact.npz')
    assert np.allclose(points[:, [0, 2]], saved['floor_demands_xz_m'][1:1+len(points)], atol=1e-10, rtol=0)
    picture, cam, ids = S.render(domain, size=1200, ground_points=points)
    draw = ImageDraw.Draw(picture)
    projected = cam.project(points)
    shown = 0
    for x, y, _ in projected:
        ix, iy = int(round(x)), int(round(y))
        # The workpiece occludes points on the far side of its floor contact.
        if 0 <= ix < cam.size and 0 <= iy < cam.size and ids[iy, ix] in (0, 1):
            draw.ellipse((x-2, y-2, x+2, y+2), fill=S.ORANGE)
            shown += 1
    page = Image.new('RGB', (1500, 1460), S.PAPER)
    draw = ImageDraw.Draw(page)
    S.text(draw, (750, 64), 'Where the load reaches the floor', 44)
    S.text(draw, (750, 120), 'B / pose 2', 30, S.MUTED)
    page.paste(picture, (150, 160))
    S.text(draw, (750, 1340), 'Orange: required floor-resultant locations', 27)
    S.text(draw, (750, 1398), '32,768 sampled loads  |  Gravity + process force from 0 to 0.5 mg', 25, S.MUTED)
    page.save(HERE / 'on_the_floor_B.png')
    S.record(HERE / 'on_the_floor_B.json', sample_count=len(points),
             visible_points=shown, minimum_normal_mg=float(normal.min()),
             baseline_floor_points_reproduced=True,
             claim='Sampled aggregate floor locations; not full bearing or a continuous boundary certificate')
    print(HERE / 'on_the_floor_B.png', flush=True)


def resultant():
    domain = S.load()
    q = domain.mesh.triangles_center[S.LOAD_FACE]
    push = np.array([0., -.5, 0.])
    p, normal = landings(q, push, domain.com)
    # For these parallel vertical forces, their weighted application point is
    # a point on the resultant line. No 3-D intersection of skew lines is assumed.
    x = (domain.com+.5*q)/1.5
    assert np.allclose(x[[0, 2]], p[[0, 2]])
    picture, cam, _ = S.render(domain, size=1200)
    S.applied_force(picture, cam, domain)
    draw = ImageDraw.Draw(picture)
    c2 = cam.project(domain.com)[:2]
    g2 = cam.project(domain.com+[0., -.035, 0.])[:2]
    S.arrow(draw, c2, g2, S.BLUE)
    S.text(draw, g2+[20, -8], 'mg', 25, S.BLUE, 'lm')
    a, b = cam.project(x)[:2], cam.project(p)[:2]
    for t in np.arange(0., 1., .07):
        u, v = a+t*(b-a), a+min(t+.035, 1.)*(b-a)
        draw.line([tuple(u), tuple(v)], fill=S.ORANGE, width=4)
    S.arrow(draw, a, a+.43*(b-a), S.ORANGE)
    draw.ellipse((b[0]-7, b[1]-7, b[0]+7, b[1]+7), fill=S.ORANGE)
    S.text(draw, b+[20, 14], 'p', 29, S.ORANGE, 'lm')
    fig = plt.figure(figsize=(18, 10), dpi=160, facecolor='white')
    fig.text(.5, .94, 'The load resultant and the floor', ha='center', fontsize=30)
    fig.text(.5, .89, 'B / pose 2', ha='center', fontsize=19, color=S.MUTED)
    for y, label, equation in (
        (.72, '1   External forces', r'$W=-mg\,\hat y+F_{\rm push}$'),
        (.53, '2   Their moment about the origin', r'$M=c\times(-mg\,\hat y)+q\times F_{\rm push}$'),
        (.34, '3   Required floor location', r'$N=-W_y,\qquad p=(-M_z/N,\;0,\;M_x/N)$')):
        fig.text(.055, y, label, fontsize=18, color=S.MUTED)
        fig.text(.055, y-.075, equation, fontsize=22)
    fig.text(.055, .13, 'The floor contact hull must contain p for every load.', fontsize=17)
    fig.text(.055, .085, 'Necessary for tipping resistance; friction and yaw need joint checks.',
             fontsize=14, color=S.MUTED)
    ax = fig.add_axes([.50, .10, .48, .74])
    ax.imshow(picture); ax.axis('off')
    fig.savefig(HERE / 'row3.png', facecolor='white')
    plt.close(fig)
    S.record(HERE / 'row3.json', q_m=q.tolist(), force_push_mg=push.tolist(),
             floor_point_m=p.tolist(), normal_mg=float(normal),
             horizontal_moment_balance_verified=True)
    print(HERE / 'row3.png', flush=True)


if __name__ == '__main__':
    floor_cloud()
    resultant()
