"""Show this round's choice and its joint-coverage ranking."""
import argparse
import io
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
M = load_stage('select', 'choose')
C = load_stage('score', 'contribution')
from step2_local_support import render as R
from step3_scheculer import contacts as I
BLUE = np.array([49., 127., 195.])
ORANGE = R.ORANGE

def scene(domain, data, basis, focus, width, size):
    base = domain.mesh.triangles
    colors = np.tile(R.GREY, (len(base), 1))
    colors[domain.work_ids] = R.GREEN
    split = data.offsets[-2]
    triangles = np.concatenate([base, data.triangles])
    colors = np.concatenate([colors, np.tile(BLUE, (split, 1)),
                            np.tile(ORANGE, (len(data.triangles)-split, 1))])
    picture, ids = R.raster(triangles, colors, focus, basis, width, size,
                           overlay=np.arange(len(base), len(triangles)))
    counts = [int(np.count_nonzero((ids >= len(base)+data.offsets[i]) &
                                 (ids < len(base)+data.offsets[i+1]))) for i in range(len(data.valid))]
    return picture, counts


def contact_camera(domain, data, index):
    """Search above and below the object; underside contacts need lower views."""
    a, b = data.offsets[index:index+2]
    points = data.triangles[a:b].mean(axis=1)
    normals = domain.mesh.face_normals[data.source_faces[a:b]]
    weights = data.triangle_areas[a:b]
    directions = [domain.mesh.face_normals[data.center_faces[index]]]
    directions.extend([np.cos(azimuth), elevation, np.sin(azimuth)]
        for azimuth in np.linspace(0, 2*np.pi, 16, endpoint=False)
        for elevation in [-1.5, -.7, -.25, 0., .25, .7, 1.5])
    best = (-1., None)
    for direction in directions:
        view = np.asarray(direction)/np.linalg.norm(direction)
        clear = ~domain.mesh.ray.intersects_any(points+1e-7*normals, np.tile(view, (len(points), 1)))
        score = float((weights*clear*np.maximum(normals@view, 0.)).sum())
        if score > best[0]:
            best = (score, view)
    return R.axes(best[1])


def ranking_plot(selection, score):
    paper = np.array(R.PAPER)/255
    fig, (top, all_scores) = plt.subplots(2, 1, figsize=(7.5, 10.2), dpi=130)
    fig.patch.set_facecolor(paper)
    base = 100*score['base_covered_count']/score['sample_count']
    rows = [score['contributions'][i] for i in selection['ranking']]
    best = rows[:10]
    top.barh(np.arange(len(best)), base, color=BLUE/255)
    top.barh(np.arange(len(best)), [r['covered_percent']-base for r in best], left=base, color=ORANGE/255)
    top.set_yticks(np.arange(len(best)), [r['id'] for r in best])
    for i, row in enumerate(best):
        top.text(row['covered_percent']+.6, i, f'{row["covered_percent"]:.2f}%', va='center', fontsize=10)
    top.invert_yaxis()
    top.set_xlim(0, 112)
    top.set_xticks([0, 25, 50, 75, 100])
    top.set_xlabel('Joint covered samples (%)')
    top.set_title('Top 10 / current fixed contacts + candidate', loc='left', fontsize=14, pad=15)
    all_scores.plot(np.arange(1, len(rows)+1), [r['covered_percent'] for r in rows], color=ORANGE/255, linewidth=2)
    all_scores.axhline(base, color=BLUE/255, label='Previously fixed contacts + original floor')
    all_scores.set(xlabel='Candidate rank by joint coverage', ylabel='Covered samples (%)', ylim=(-2, 104))
    all_scores.set_title(f'All {len(rows)} eligible candidates', loc='left', fontsize=14)
    all_scores.legend(fontsize=9, frameon=False)
    for ax in [top, all_scores]:
        ax.set_facecolor(paper)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.12, right=.97, top=.94, bottom=.10, hspace=.36)
    stream = io.BytesIO()
    fig.savefig(stream, dpi=130, facecolor=paper)
    plt.close(fig)
    stream.seek(0)
    return Image.open(stream).convert('RGB')


def run(name, round_number=1):
    selected = M.read(name, round_number)
    score = C.read(name, round_number)
    if selected['winner'] is None:
        return
    out = I.folder(name, M.OUTPUT_NAME, round_number)
    domain, _, _ = C.P.read(name)
    contacts = I.read_contacts(out/'contacts_before_optimization.npz')
    data = I.as_data(contacts)
    winner = selected['winner']
    picture = Image.new('RGB', (2220, 1510), R.PAPER)
    ink = ImageDraw.Draw(picture)
    ink.text((35, 24), f'{name} / Step 3.2 / Round {round_number} / Select {winner["id"]}', font=R.font(43), fill=R.INK)
    ink.text((35, 88), f'Joint coverage {winner["covered_percent"]:.2f}% before size optimization; {len(contacts)} contacts.',
             font=R.font(29), fill=R.INK)
    views = []
    for j, index in enumerate([0, len(contacts)-1]):
        x = 25+700*j
        basis = contact_camera(domain, data, index)
        focus, width = R.overall_camera(domain, basis)
        image, visible = scene(domain, data, basis, focus, width, 690)
        picture.paste(image, (x, 200))
        ink.text((x+15, 165), f'View toward {contacts[index]["candidate_id"]}', font=R.font(27), fill=R.INK)
        width_zoom = 2.8*max(p['radius_m'] for p in contacts)
        detail, pixels = scene(domain, data, basis, data.centers_m[index], width_zoom, 460)
        picture.paste(detail, (x+110, 940))
        ink.text((x+15, 1420), f'Contact detail / radius {1000*contacts[index]["radius_m"]:.2f} mm', font=R.font(22), fill=R.INK)
        assert visible[index] > 0 and pixels[index] > 0
        views.append(dict(index=index, candidate_id=contacts[index]['candidate_id'], basis=basis.tolist(),
            focus_m=focus.tolist(), width_m=float(width), visible_pixels=visible,
            detail_focus_m=data.centers_m[index].tolist(), detail_width_m=width_zoom))
    chart = ranking_plot(selected, score)
    chart.save(out/'ranking.png')
    picture.paste(chart.resize((765, 1040), Image.Resampling.LANCZOS), (1440, 175))
    ink.text((1485, 1270), f'{len(selected["tied_best_ids"])} tied best; ordered by ID.', font=R.font(23), fill=R.INK)
    ink.text((1485, 1320), 'Next: Step 3.3 optimizes the orange contact.', font=R.font(23), fill=R.INK)
    ink.text((35, 1470), 'Blue: previously fixed contacts. Orange: current selection. Green: working surface. Original floor reaction included; floor omitted from views.', font=R.font(20), fill='#65706c')
    picture.save(out/'selection.png')
    I.save(out/'selection_views.json', dict(object=name, round=round_number, views=views,
        selection_sha256=C.sha256(out/'selection.json'), code=I.hashes([Path(__file__), Path(R.__file__)]),
        artifacts={f:C.sha256(out/f) for f in ['selection.png', 'ranking.png']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name, args.round)
