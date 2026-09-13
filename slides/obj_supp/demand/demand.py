r"""Demand in the pipeline's coverage-sphere and relief-sphere visual language.

    python slides/obj_supp/demand/demand.py
        -> demand_pairs.png

Actual rendering is shared with pipeline/step2: cover.globe/relief_shell.
The fields are DEMAND, not step2's supply LPs. Force paint is at -F_D/|F_D|;
moment relief is the maximum sampled |tau_D| in each +tau_D direction bin.
Highlighted pairs are complete original rows that win their moment bin.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "slides/tools"), str(ROOT / "slides/setup/poses")]
import cover as C
import slide_scene as SC

K = 0.5
SUBDIVISIONS = 4
SHEET_ROUNDS = 3
VIEW_ELEVATION = -20.0             # see the southern -F cap; shared by both balls
RELIEF_TOP = 0.55                   # pipeline/step2's linear height budget
RING_STEP = 5.0                     # mg mm; converted by exactly the same map
BARE = "#cbc6ba"
COLORS = ("#2867b2", "#f0a122", "#459f80")
OUT = HERE / "demand_pairs.png"


def sample_current_pose():
    """Read paired loads for the exact B/pose_2 shown throughout the deck."""
    domain = SC.load()
    samples = SC.samples(domain)
    magnitude = np.linalg.norm(samples['push'], axis=1)
    assert np.all(magnitude > 0) and np.all(magnitude <= K+1e-12)
    direction = samples['push']/magnitude[:, None]
    return dict(pt=samples['q'], d=direction, push=samples['push'],
                magnitude=magnitude, force=samples['wrench'][:, :3],
                moment=1000*samples['wrench'][:, 3:], com=domain.com,
                sample_seed=samples['seed'])


def direction_fields(data):
    """Keep per-bin winner provenance; a bin max is not an independent partner."""
    ico, tiles = C.tiling(SUBDIVISIONS)
    tree = cKDTree(tiles)
    force_norm = np.linalg.norm(data["force"], axis=1)
    moment_norm = np.linalg.norm(data["moment"], axis=1)
    assert np.all(force_norm > 0) and np.all(moment_norm > 0)
    force_dir = data["force"] / force_norm[:, None]
    moment_dir = data["moment"] / moment_norm[:, None]
    force_bin = tree.query(-force_dir)[1]       # original pipeline -F convention
    moment_bin = tree.query(moment_dir)[1]     # moment axes have no reversal
    force_raw = np.zeros(len(tiles), bool)
    force_raw[force_bin] = True
    force_sheet = C.sheet(C.neighbours(ico), force_bin, rounds=SHEET_ROUNDS)
    order = np.lexsort((np.arange(len(moment_norm)), -moment_norm, moment_bin))
    winners = order[np.r_[True, np.diff(moment_bin[order]) != 0]]
    winner_id = np.full(len(tiles), -1, dtype=int)
    winner_id[moment_bin[winners]] = winners
    hi = np.full(len(tiles), np.nan)
    hi[moment_bin[winners]] = moment_norm[winners]
    top = float(np.nanmax(hi))
    height = RELIEF_TOP * hi / top             # step2's linear map, no clipping
    rings = np.arange(RING_STEP, top, RING_STEP)
    assert np.allclose(hi[moment_bin[winners]], moment_norm[winners])
    assert np.isclose(np.nanmax(height), RELIEF_TOP)
    data.update(force_norm=force_norm, moment_norm=moment_norm, force_dir=force_dir,
                moment_dir=moment_dir, force_bin=force_bin, moment_bin=moment_bin,
                winner_id=winner_id, hi=hi, top=top, height=height,
                force_raw=force_raw, force_sheet=force_sheet, ring_values=rings)
    return ico, tiles


def figure_points(ax, fig, xyz):
    x, y, _ = proj3d.proj_transform(*np.asarray(xyz).T, ax.get_proj())
    return fig.transFigure.inverted().transform(ax.transData.transform(np.c_[x, y]))


def select_highlights(data, left, right, fig):
    """Actual moment-bin winners whose two marks are legible in the common view."""
    eye = C.screen_axes()[0]
    winners = data["winner_id"][data["winner_id"] >= 0]
    front = ((-data["force_dir"][winners] @ eye > 0.015)
             & (data["moment_dir"][winners] @ eye > 0.2))
    candidates = winners[front]
    if len(candidates) < len(COLORS):
        raise RuntimeError("Too few visible bin-winning pairs in the shared view")
    F = figure_points(left, fig, -1.018 * data["force_dir"][candidates])
    h = RELIEF_TOP * data["moment_norm"][candidates] / data["top"]
    M = figure_points(right, fig, (1 + h[:, None]) * data["moment_dir"][candidates])
    rng = np.random.default_rng(17)
    best = None
    pairs = [(0, 1), (0, 2), (1, 2)]
    for _ in range(4000):
        ids = rng.choice(len(candidates), len(COLORS), replace=False)
        fl, ml = F[ids], M[ids]
        # Figure-coordinate distances use one common screen metric.
        fl = fl * [fig.get_figwidth(), fig.get_figheight()]
        ml = ml * [fig.get_figwidth(), fig.get_figheight()]
        score = min(np.linalg.norm(fl[i] - fl[j]) for i, j in pairs)
        score += 0.36 * min(np.linalg.norm(ml[i] - ml[j]) for i, j in pairs)
        if best is None or score > best[0]:
            best = score, ids
    chosen = best[1]
    chosen = chosen[np.argsort(F[chosen, 0])]
    return candidates[chosen], F[chosen], M[chosen]


@contextlib.contextmanager
def shared_globe_view():
    """Reuse the renderer with one lower camera that reveals the -F cap."""
    old_elev, old_light = C.ELEV, C.LIGHT.copy()
    try:
        C.ELEV = VIEW_ELEVATION
        C.LIGHT = C._light()
        yield
    finally:
        C.ELEV, C.LIGHT = old_elev, old_light


def draw(data, ico):
    """The same solid paint and stepped radial shell as pipeline/step2."""
    fig = plt.figure(figsize=(16, 8.8), dpi=180, facecolor=C.PAPER)
    left = fig.add_axes([0.015, 0.245, 0.46, 0.58], projection="3d", facecolor=C.PAPER)
    right = fig.add_axes([0.525, 0.245, 0.46, 0.58], projection="3d", facecolor=C.PAPER)
    C.globe(left, ico, [(np.flatnonzero(data["force_sheet"]), C.NEED, .95)],
            [], [], [], "", triad=True, reach=1.60, weight=True)
    C.globe(right, ico, [], [], [], [], "", reach=1.60, weight=False,
            relief=(data["height"], C.NEED, BARE,
                    tuple(RELIEF_TOP * data["ring_values"] / data["top"])),
            rings_front=True)
    fig.text(.5, .955, "Force–moment demand", ha="center", fontsize=28, color=C.INK)
    fig.text(.5, .907,
             rf"B / pose 2 · {len(data['force']):,} sampled pushes · $0\leq|F_{{\rm push}}|\leq0.5\,mg$",
             ha="center", fontsize=15, color=C.MUTED)
    fig.text(.245, .853, "Force coverage", ha="center", fontsize=22, color=C.INK)
    fig.text(.245, .815, r"[FORCE dirs] · displayed at $-F_D$", ha="center", fontsize=14, color=C.MUTED)
    fig.text(.755, .853, "Moment relief", ha="center", fontsize=22, color=C.INK)
    fig.text(.755, .815, r"[TURNING AXES] · $+\tau_D$ about $c$", ha="center", fontsize=14, color=C.MUTED)
    fig.canvas.draw()
    ids, F, M = select_highlights(data, left, right, fig)
    for number, (idx, lf, rt, color) in enumerate(zip(ids, F, M, COLORS), 1):
        assert data["winner_id"][data["moment_bin"][idx]] == idx
        assert data["hi"][data["moment_bin"][idx]] == data["moment_norm"][idx]
        for point, offset in [(lf, (-.013, -.022)), (rt, (.014, .015))]:
            fig.add_artist(Line2D([point[0]], [point[1]], marker="o", markersize=7.5,
                                  markerfacecolor=color, markeredgecolor=C.PAPER,
                                  markeredgewidth=1.7, linestyle="none",
                                  transform=fig.transFigure, zorder=230))
            label = fig.text(*(point + offset), str(number), color=color,
                             fontsize=15, weight="bold", ha="center", va="center", zorder=240)
            label.set_path_effects([pe.withStroke(linewidth=3.5, foreground=C.PAPER)])
    fig.text(.245, .246, "red = sampled demand directions", ha="center", fontsize=13, color=C.MUTED)
    fig.text(.755, .246, r"height = bin max $|\tau_D|$ · rings every $5\,mg\,{\cdot}\,\mathrm{mm}$",
             ha="center", fontsize=13, color=C.MUTED)
    fig.text(.5, .193, r"Same colour + number = one push · values are $(|F_D|,\;|\tau_D|)$",
             ha="center", fontsize=14, color=C.INK)
    for number, (idx, color, x) in enumerate(zip(ids, COLORS, [.2, .5, .8]), 1):
        value = (rf"{number}   $({data['force_norm'][idx]:.3f}\,mg,\;"
                 rf"{data['moment_norm'][idx]:.2f}\,mg\,{{\cdot}}\,\mathrm{{mm}})$")
        fig.text(x, .135, value, ha="center", fontsize=17, color=color)
    fig.text(.5, .061, "Each highlighted push is the actual maximum sample in its moment bin.",
             ha="center", fontsize=12.5, color=C.MUTED)
    fig.text(.5, .025, "Finite sampled demand · separate displays do not test joint feasibility",
             ha="center", fontsize=11.5, color=C.MUTED)
    fig.savefig(OUT, facecolor=C.PAPER, edgecolor=C.PAPER, transparent=False)
    plt.close(fig)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, help="optional audit NPZ, normally under /tmp")
    args = parser.parse_args()
    data = sample_current_pose()
    ico, _ = direction_fields(data)
    with shared_globe_view():
        ids = draw(data, ico)
    SC.record(HERE/'demand_pairs.json', sample_count=len(data['force']),
              sample_seed=data['sample_seed'], highlighted_ids=ids.tolist(),
              moment_units='mg mm', moment_relief_max=float(data['top']),
              scope='Finite paired demand samples, not joint coverage or a continuous boundary')
    print(f"Wrote {OUT}; {len(data['force'])} paired samples")
    print("Shared c / m:", data["com"].tolist())
    print("Force bins: raw", int(data["force_raw"].sum()), "closed", int(data["force_sheet"].sum()))
    print("Moment bins:", int(np.isfinite(data["hi"]).sum()), "/", len(data["hi"]))
    print("Linear relief top / (mg mm):", data["top"], "; h = 0.55 M / top; no clipping")
    print("Rings / (mg mm):", data["ring_values"].tolist())
    for number, idx in enumerate(ids, 1):
        print(f"pair {number}: original zero-based sample {idx}, moment bin {data['moment_bin'][idx]}, "
              f"|F|={data['force_norm'][idx]:.9f}, |tau|={data['moment_norm'][idx]:.9f}")
    if args.audit:
        np.savez_compressed(args.audit, **data, highlighted_ids=ids,
                            K=K, seed=data['sample_seed'])
        print("Audit:", args.audit)


if __name__ == "__main__":
    main()
