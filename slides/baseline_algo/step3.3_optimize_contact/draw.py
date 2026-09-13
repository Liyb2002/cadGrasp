"""Draw the same combination before and after optimizing the current contact."""
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
A = load_stage('optimize', 'adjust')
D = load_stage('select', 'draw')
from step2_local_support import render as R
from step3_scheculer import contacts as I


def curve(result):
    paper = np.array(R.PAPER)/255
    fig, axes = plt.subplots(2, 1, figsize=(7.7, 8), dpi=130, sharex=True)
    fig.patch.set_facecolor(paper)
    rows = result['curve']
    for ax, field, ylabel in zip(axes, ['efficiency', 'covered_percent'],
                               ['Joint coverage % / total area %', 'Joint covered samples (%)']):
        ax.set_facecolor(paper)
        ax.plot([r['area_m2']*1e6 for r in rows], [r[field] for r in rows], '.--', color='#99aaa6', markersize=4)
        ax.axvline(result['area_constraint']['minimum_area_m2']*1e6,
                   color='#ad5634', linestyle=':', label='0.5% area limit')
        for key, color in [('initial', '#317fc3'), ('adjusted', '#ef8b25')]:
            row = result[key]
            ax.scatter(1e6*row['area_m2'], row[field], s=100, color=color, label=key)
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend()
    axes[0].set_title('Optimize coverage / total contact area', loc='left', fontsize=15)
    axes[1].set_xlabel('Current contact area (mm²); previous contact areas fixed')
    axes[1].set_ylim(0, 102)
    fig.subplots_adjust(left=.15, right=.98, top=.93, bottom=.12, hspace=.17)
    stream = io.BytesIO()
    fig.savefig(stream, dpi=130, facecolor=paper)
    plt.close(fig)
    stream.seek(0)
    return Image.open(stream).convert('RGB')


def run(name, round_number=1):
    result = A.read(name, round_number)
    out = I.folder(name, A.OUTPUT_NAME, round_number)
    source = I.folder(name, A.M.OUTPUT_NAME, round_number)
    sets = [I.read_contacts(source/'contacts_before_optimization.npz'), I.read_contacts(out/'contacts.npz')]
    domain, _, _ = A.P.read(name)
    data = [I.as_data(contacts) for contacts in sets]
    index = len(sets[0])-1
    basis = D.contact_camera(domain, data[0] if result['initial']['radius_m'] >= result['adjusted']['radius_m'] else data[1], index)
    focus, width = R.overall_camera(domain, basis)
    detail_width = 2.8*max(result[k]['radius_m'] for k in ['initial', 'adjusted'])
    paper = Image.new('RGB', (2220, 1510), R.PAPER)
    ink = ImageDraw.Draw(paper)
    ink.text((35, 25), f'{name} / Step 3.3 / Round {round_number} / Optimize {result["candidate_id"]}', font=R.font(43), fill=R.INK)
    ink.text((35, 88), 'Only the current contact changes. All previous contact geometry stays fixed.', font=R.font(28), fill=R.INK)
    views = []
    for i, (state, label) in enumerate([('initial', 'Before'), ('adjusted', 'After')]):
        x = 25+700*i
        row = result[state]
        ink.text((x+15, 165), f'{label}: covered {row["covered_percent"]:.2f}% / efficiency {row["efficiency"]:.2f}', font=R.font(25), fill=R.INK)
        picture, shown = D.scene(domain, data[i], basis, focus, width, 690)
        paper.paste(picture, (x, 210))
        detail, shown_detail = D.scene(domain, data[i], basis, data[i].centers_m[index], detail_width, 460)
        paper.paste(detail, (x+110, 940))
        ink.text((x+15, 1420), f'Current area {1e6*row["area_m2"]:.4g} mm2 / radius {1000*row["radius_m"]:.4g} mm', font=R.font(22), fill=R.INK)
        if not shown_detail[index]:
            ink.text((x+15, 905), 'Current contact is below image resolution.', font=R.font(21), fill='#ad5634')
        views.append(dict(state=state, visible_pixels=shown, detail_visible_pixels=shown_detail))
    chart = curve(result)
    chart.save(out/'coverage_curve.png')
    paper.paste(chart.resize((770, 800), Image.Resampling.LANCZOS), (1440, 190))
    for j, line in enumerate([result['direction'].capitalize(),
            f'Previous area: {1e6*result["fixed_area_m2"]:.2f} mm2',
            f'Total area: {1e6*result["adjusted"]["total_area_m2"]:.2f} mm2',
            f'Coverage: {result["adjusted"]["covered_percent"]:.4f}%',
            f'Each area > {1e6*result["area_constraint"]["minimum_area_m2"]:.2f} mm2 (0.5%)',
            'Scheduler checks completion AFTER this step.']):
        ink.text((1490, 1040+66*j), line, font=R.font(26), fill=R.INK)
    ink.text((35, 1470), 'Blue: previously fixed contacts. Orange: current contact. Same camera and scale before/after. Coverage is measured on the stored samples.', font=R.font(21), fill='#65706c')
    paper.save(out/'adjustment.png')
    I.save(out/'adjustment_views.json', dict(object=name, round=round_number, views=views,
        adjustment_sha256=A.C.sha256(out/'adjustment.json'), same_camera_and_scale=True,
        basis=basis.tolist(), focus_m=focus.tolist(), width_m=float(width), detail_width_m=detail_width,
        code=I.hashes([Path(__file__), Path(D.__file__), Path(R.__file__)]),
        artifacts={f:A.C.sha256(out/f) for f in ['adjustment.png', 'coverage_curve.png']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()
    for name in args.objects or A.C.OBJECTS:
        run(name, args.round)
