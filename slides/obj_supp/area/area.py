"""Find and verify several small, separate contact regions at the existing pose.

    python slides/obj_supp/area/area.py A1-f B C5

Score small patches across the touchable surface, refine valuable locations,
then compare growing the current patches with adding another small patch.
All candidate values use the complete paired demand table. Actual original-mesh
shared edges determine connectivity; area_search.py records every comparison.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, nnls
from scipy.sparse.csgraph import connected_components
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch as P
import drawing as D
import views as V

OPTIONS = {"primal_feasibility_tolerance": 1e-9,
           "dual_feasibility_tolerance": 1e-9}
# A single patch's normals stay within one open hemisphere. The former 40-degree
# display restriction unnecessarily truncated the useful variation at edges.
NORMAL_ANGLE = 89.0


def load(name):
    """Reconstruct the current pose and contact mesh from the local demand table."""
    path = HERE/f'demand_{name}_tip1.npz'
    if not path.exists():
        with P.I.current_pose(name) as source:
            P.I.demand_table(name, source.mesh, source.record)
    with np.load(path) as z:
        demand = {key: z[key] for key in z.files}
    for filename, field in (('mesh.stl', 'mesh_sha256'), ('poses.json', 'poses_sha256')):
        assert P.I.sha256(P.ROOT/'objects'/name/filename) == str(demand[field])
    assert float(demand['K']) == .5
    mesh, _ = P.G.refine(trimesh.load(P.ROOT/'objects'/name/'mesh.stl', force='mesh'))
    T = demand['T_world_mesh']
    R, t = T[:3, :3], T[:3, 3]
    extent = float(mesh.extents.max())
    touch = ~demand['work_faces'] & ((mesh.triangles_center@R.T+t)[:, 1] > P.CONTACT_EPS)
    skin = P.Skin(mesh, touch, R, t, P.CUT*extent, print)
    import hashlib
    fingerprint = hashlib.sha256()
    for data in (skin.cs, skin.ns, skin.src, skin.area):
        fingerprint.update(data.tobytes())
    return SimpleNamespace(name=name, demand=demand, mesh=mesh, skin=skin, touch=touch,
                           T=T, R=R, t=t, adj=mesh.face_adjacency,
                           edges=mesh.vertices[mesh.face_adjacency_edges]@R.T+t,
                           com=demand['com_m'], normal_angle=NORMAL_ANGLE,
                           geometry_sha256=fingerprint.hexdigest(),
                           floor6=np.r_[P.UP, np.cross(demand['floor_contact_m']-demand['com_m'], P.UP)],
                           scale=np.r_[np.ones(3), np.ones(3)/extent],
                           targets=np.c_[demand['force_w'], demand['moment_wmm']/1000],
                           radius=.03*extent)


def region(S, seed, radius=None):
    """Small ball, normal cap, and connectivity across real shared edges.

    Both adjacent source faces must be touchable and within the normal cap of the
    seed normal; their shared edge must intersect the ball. Thus a patch cannot
    quietly cross the solid to its backside via a centroid proximity link.
    """
    radius = S.radius if radius is None else radius
    centre, normal = S.skin.cs[seed], S.skin.ns[seed]
    a, b = S.edges[:, 0], S.edges[:, 1]
    ab = b - a
    s = np.clip(np.einsum("ij,ij->i", centre-a, ab) /
                np.einsum("ij,ij->i", ab, ab), 0, 1)
    distance = np.linalg.norm(a + s[:, None] * ab - centre, axis=1)
    allowed = S.touch & ((S.mesh.face_normals @ S.R.T) @ normal >=
                         np.cos(np.deg2rad(NORMAL_ANGLE)))
    adj = S.adj[(distance <= radius + 1e-12) & allowed[S.adj].all(axis=1)]
    graph = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
                          shape=(len(S.mesh.faces), len(S.mesh.faces))).tocsr()
    labels = connected_components(graph, directed=False)[1]
    ids = np.array(S.skin.tree.query_ball_point(centre, radius), dtype=int)
    ids = ids[allowed[S.skin.src[ids]] &
              (labels[S.skin.src[ids]] == labels[S.skin.src[seed]])]
    mask = np.zeros(len(S.skin.cs), bool)
    mask[ids] = True
    assert mask[seed]
    return mask


def equilibrium(targets, columns):
    """Return checked primal solutions and separating duals for failed demands.

    NNLS first provides inexpensive candidate solutions; direct substitution
    accepts only small residuals. A phase-I LP resolves every remaining sample.
    Its positive optimum has a checked dual certificate of infeasibility.
    """
    A = np.asarray(columns).T.copy(order="F")
    dim, n = A.shape
    phase = np.c_[A, np.eye(dim), -np.eye(dim)]
    objective = np.r_[np.zeros(n), np.ones(2*dim)]
    ok = np.zeros(len(targets), bool)
    coefficients = np.zeros((len(targets), n))
    duals = np.zeros((len(targets), dim))
    for i, target in enumerate(targets):
        try:
            x = nnls(A, target, maxiter=max(1000, 3*n))[0]
        except RuntimeError:
            x = np.zeros(n)
        if np.max(np.abs(A @ x - target)) > 1e-9:
            result = linprog(objective, A_eq=phase, b_eq=target,
                             bounds=(0, None), method="highs", options=OPTIONS)
            if not result.success:
                raise RuntimeError(result.message)
            x = result.x[:n]
            if result.fun > 1e-8:
                y = result.eqlin.marginals
                assert np.max(A.T @ y) < 2e-8 and target @ y > 1e-8
                duals[i] = y
                continue
        assert x.min() >= -1e-10 and np.max(np.abs(A @ x-target)) < 2e-8
        ok[i], coefficients[i] = True, x
    return ok, coefficients, duals


def grow_to_area(S, seed, ratio=3.):
    """Keep the seed and match actual mesh area, rather than a radius factor."""
    base = region(S, seed)
    target = ratio*S.skin.area[base].sum()
    low, high = S.radius, 2*S.radius
    for _ in range(12):
        if S.skin.area[region(S, seed, high)].sum() >= target:
            break
        high *= 1.25
    else:
        raise RuntimeError('The connected region cannot reach the requested area')
    for _ in range(32):
        mid = (low+high)/2
        if S.skin.area[region(S, seed, mid)].sum() < target:
            low = mid
        else:
            high = mid
    radius = min((low, high), key=lambda r: abs(S.skin.area[region(S, seed, r)].sum()-target))
    actual = S.skin.area[region(S, seed, radius)].sum()/S.skin.area[base].sum()
    assert abs(actual/ratio-1) < .005, actual
    return radius, actual


def designs(S, search_report):
    """One small region, triple its area, then exactly two and three supports."""
    from area_search import Search, RADIUS_LEVELS
    trace = search_report['steps']
    seed = trace[0]['state'][0][0]
    radius, actual = grow_to_area(S, seed)
    pair = next(step for step in trace if len(step['state']) == 2)
    triple = next((step for step in trace if len(step['state']) == 3), None)
    selection = 'first three-region state of the value-first search'
    if triple is None:
        # The search already solved every sampled demand with two regions.
        # For the requested three-support demo, add the highest standalone-value
        # eligible disjoint candidate; do not claim any additional solved demand.
        assert pair['joint_count'] == len(S.targets)
        searcher = Search(S, region, P.fps)
        state = tuple(tuple(p) for p in pair['state'])
        ranked = sorted((c for c in search_report['candidates'] if c['eligible']),
                        key=lambda c: (-c['joint_count'], c['area_cm2'], c['seed']))
        third = next(c for c in ranked if searcher.compatible(state, c['seed']))
        triple = dict(state=pair['state']+[[third['seed'], 0]], joint_count=pair['joint_count'])
        selection = 'pair already covers all samples; add highest standalone-value compatible small region'
    def specs(state):
        return [(int(seed), float(S.radius*RADIUS_LEVELS[level]), int(level)) for seed, level in state]
    plan = [specs(trace[0]['state']), [(seed, radius, None)], specs(pair['state']), specs(triple['state'])]
    expected = [trace[0]['joint_count'], None, pair['joint_count'], triple['joint_count']]
    return plan, expected, dict(row2_target_area_ratio=3., row2_actual_area_ratio=actual,
                                third_support_selection=selection,
                                pair_state=pair['state'], triple_state=triple['state'])


def verify(S, search_report):
    """Recheck each chosen search state with every actual contact generator."""
    from area_search import RADIUS_LEVELS
    trace = search_report['steps']
    states, expected_counts, selection = designs(S, search_report)
    labels = ['One small support', 'Contact area ×3', 'Two supports', 'Three supports']
    row_masks = [[region(S, seed, radius) for seed, radius, level in state] for state in states]
    arrays = dict(patch_masks=np.array(row_masks[-1]),
                  seeds=np.array([p[0] for p in states[-1]]),
                  skin_centres_m=S.skin.cs, skin_outward_normals=S.skin.ns,
                  skin_face_area_m2=S.skin.area, source_face=S.skin.src,
                  targets=S.targets, scale=S.scale, com_m=S.com,
                  T_world_mesh=S.T, floor6=S.floor6)
    rows = []
    for k, (label, state, masks) in enumerate(zip(labels, states, row_masks), start=1):
        ids = np.flatnonzero(np.logical_or.reduce(masks))
        columns = np.vstack([S.skin.gens(ids, S.com), S.floor6[None]])
        ok, x, dual = equilibrium(S.targets*S.scale, columns*S.scale)
        okf, _, _ = equilibrium(S.targets[:, :3], columns[:, :3])
        okm, _, _ = equilibrium(S.targets[:, 3:]*S.scale[3:], columns[:, 3:]*S.scale[3:])
        assert np.all(~ok | (okf & okm))
        if expected_counts[k-1] is not None:
            assert int(ok.sum()) == expected_counts[k-1], 'Full-contact verification disagrees with selection'
        if k > 1:
            parent = 3 if k == 4 else 1
            assert np.all(~arrays[f'ok_joint_{parent}'] | ok)
            assert np.all(~arrays[f'row_masks_{parent}'].any(axis=0) | np.logical_or.reduce(masks))
        residual = x[ok] @ columns-S.targets[ok]
        row = dict(row=k, n_regions=len(masks), n_samples=len(ok), joint_count=int(ok.sum()),
                   force_count=int(okf.sum()), moment_count=int(okm.sum()),
                   label=label,
                   area_cm2=float(1e4*S.skin.area[ids].sum()),
                   patches=[dict(number=j+1, seed=int(seed), level=level,
                                 radius_mm=1000*radius,
                                 centre_mm=(1000*S.skin.cs[seed]).tolist(),
                                 area_cm2=float(1e4*S.skin.area[m].sum()))
                            for j, ((seed, radius, level), m) in enumerate(zip(state, masks))],
                   max_force_residual_mg=float(np.abs(residual[:, :3]).max()) if ok.any() else None,
                   max_moment_residual_mgmm=float(1000*np.abs(residual[:, 3:]).max()) if ok.any() else None)
        rows.append(row)
        arrays.update({f'row_masks_{k}': np.array(masks),
                       f'columns_{k}': columns, f'face_ids_{k}': ids,
                       f'ok_joint_{k}': ok, f'multipliers_{k}': x, f'dual_{k}': dual,
                       f'ok_force_{k}': okf, f'ok_moment_{k}': okm})
        print(S.name, 'VERIFIED ROW', k, row['label'], len(masks), int(ok.sum()),
              '/', len(ok), 'area', row['area_cm2'], flush=True)
    assert [len(m) for m in row_masks] == [1, 1, 2, 3]
    final_masks = row_masks[-1]
    without = []
    for j in range(len(final_masks)):
        remaining = [m for i, m in enumerate(final_masks) if i != j]
        ids = np.flatnonzero(np.logical_or.reduce(remaining)) if remaining else np.array([], dtype=int)
        cols = np.vstack([S.skin.gens(ids, S.com), S.floor6[None]])
        ok, _, dual = equilibrium(S.targets*S.scale, cols*S.scale)
        without.append(int(ok.sum()))
        arrays[f'without_{j+1}_ok'] = ok
        arrays[f'without_{j+1}_dual'] = dual
    report = dict(object=S.name, K=.5, tip=1, n_samples=len(S.targets),
                  source_demand_sha256=P.I.sha256(HERE/f'demand_{S.name}_tip1.npz'),
                  geometry_sha256=S.geometry_sha256,
                  search_file=f'area_{S.name}_search.json',
                  singleton_file=f'area_{S.name}_candidates.npz',
                  base_radius_mm=1000*S.radius, radius_levels=RADIUS_LEVELS.tolist(),
                  normal_angle_from_seed_max_deg=NORMAL_ANGLE,
                  candidate_count=len(search_report['candidates']),
                  eligible_candidate_count=sum(r['eligible'] for r in search_report['candidates']),
                  positive_singleton_count=sum(r['eligible'] and r['joint_count']>0
                                               for r in search_report['candidates']),
                  best_singleton_count=trace[0]['joint_count'],
                  minimum_candidate_area_cm2=search_report['minimum_candidate_area_cm2'],
                  singleton_objective=search_report['singleton_objective'],
                  move_objective=search_report['move_objective'],
                  model='Nonnegative unbounded normal reactions; original floor ray; shared force/moment solution; full finite paired demand table',
                  display_selection=selection, patches=rows[-1]['patches'], rows=rows,
                  joint_count_without_each_patch=without)
    np.savez_compressed(HERE/f'area_{S.name}_results.npz', **arrays)
    (HERE/f'area_{S.name}_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(S.name, 'remove each final patch:', without, flush=True)
    return report, arrays




def render(S, report, arrays):
    from disturbances import _font
    assert [row['n_regions'] for row in report['rows']] == [1, 1, 2, 3]
    # Frame the largest version of each selected region across all four rows.
    # Row 2 grows the first region; rows 3/4 add the second/third support.
    grouped = {}
    for k, row in enumerate(report['rows'], start=1):
        for patch, mask in zip(row['patches'], arrays[f'row_masks_{k}']):
            seed = patch['seed']
            grouped[seed] = grouped.get(seed, np.zeros(len(S.skin.cs), bool)) | mask
    masks = list(grouped.values())
    ids, views = V.overview_cameras(S, masks)
    details = {seed: V.detail_camera(S, mask, seed) for seed, mask in grouped.items()}
    detail_width, tile = 300, 220
    ico, tiles = P.tiling(4)
    F, M = S.targets[:, :3], S.targets[:, 3:]
    tree = cKDTree(tiles)
    um = M/np.linalg.norm(M, axis=1)[:, None]
    maximum = np.full(len(tiles), -np.inf)
    np.maximum.at(maximum, tree.query(um)[1], 1000*np.linalg.norm(M, axis=1))
    maximum[~np.isfinite(maximum)] = np.nan
    top = float(np.nanmax(maximum))
    h = .55*maximum/top
    ring_step = 5 if top >= 15 else 2
    drawing = SimpleNamespace(ico=ico, tiles=tiles, tree=cKDTree(tiles),
                              A_t=P.neighbours(ico), h=h,
                              uf=F/np.linalg.norm(F, axis=1)[:, None],
                              um=M/np.linalg.norm(M, axis=1)[:, None],
                              ring_r=.55*np.arange(ring_step, top, ring_step)/top)
    few = np.random.default_rng(11).choice(len(F), min(26, len(F)), replace=False)
    strips = []
    with P.I.current_pose(S.name, print) as source:
        assert np.allclose(source.record["T_star"], S.T, atol=1e-12, rtol=0)
        assert np.array_equal(source.record["take"], S.demand["work_faces"])
        tmp = source.directory/P.G.TMP
        for k in range(1, len(report['rows'])+1):
            current_masks = list(arrays[f'row_masks_{k}'])
            tmp.mkdir()
            try:
                parts = P.G.paint_parts(S.mesh, S.demand["work_faces"], tmp, f"area_{k}")
                parts.append(P.sticker(S.skin, current_masks, tmp, f"area_{k}_patches"))
                panels = []
                for cam, seen, visibility, projected in views:
                    panel = P.shot(S.name, S.T, parts, cam, P.ORANGE)
                    draw = ImageDraw.Draw(panel)
                    labels = []
                    for j, mask in enumerate(current_masks):
                        face_ids = ids[mask[ids] & seen]
                        if not len(face_ids) or S.skin.area[face_ids].sum()/S.skin.area[mask].sum() < .25:
                            continue
                        centre = np.average(S.skin.cs[face_ids], axis=0,
                                            weights=S.skin.area[face_ids])
                        xy = P.TS.screen(centre, cam)[0]
                        for dx, dy in ((38, -38), (-38, -38), (38, 38), (-38, 38), (0, -70)):
                            label = xy+np.array([dx, dy])
                            if all(np.linalg.norm(label-other) > 45 for other in labels):
                                break
                        labels.append(label)
                        draw.line([tuple(xy), tuple(label)], fill="#55554f", width=2)
                        x, y = label
                        draw.ellipse((x-17, y-17, x+17, y+17), fill=P.PAPER,
                                     outline="#55554f", width=2)
                        draw.text((x, y), str(j+1), font=_font(22), fill="#1b1b1a", anchor="mm")
                    panels.append(panel)
                closeups = Image.new('RGB', (detail_width, P.PX), P.PAPER)
                draw = ImageDraw.Draw(closeups)
                top = (P.PX-len(current_masks)*tile)/2
                row = report['rows'][k-1]
                row['detail_visible_area_fraction'] = []
                for j, (patch, mask) in enumerate(zip(row['patches'], current_masks)):
                    cam, crop = details[patch['seed']]
                    patch_ids = np.flatnonzero(mask)
                    seen, _ = V.visible(S, patch_ids, cam)
                    fraction = float(S.skin.area[patch_ids][seen].sum()/S.skin.area[patch_ids].sum())
                    assert fraction >= .95, (S.name, k, j+1, fraction)
                    row['detail_visible_area_fraction'].append(fraction)
                    detail = P.shot(S.name, S.T, parts, cam, P.ORANGE).crop(crop)
                    closeups.paste(detail.resize((tile-8, tile-8), Image.Resampling.LANCZOS), (4, int(top+4)))
                    draw.rectangle((4, top+4, tile-5, top+tile-5), outline='#c9c9c3', width=1)
                    label = (tile+35, top+tile/2)
                    draw.ellipse((label[0]-20, label[1]-20, label[0]+20, label[1]+20),
                                 fill=P.PAPER, outline='#55554f', width=2)
                    draw.text(label, str(j+1), font=_font(27), fill='#1b1b1a', anchor='mm')
                    top += tile
            finally:
                shutil.rmtree(tmp)
            balls, _ = D.balls(drawing, np.isfinite(h), arrays[f"ok_force_{k}"],
                               arrays[f"ok_moment_{k}"], few)
            strip = Image.new("RGB", (len(views)*P.PX+detail_width+balls.width, P.PX), P.PAPER)
            for j, panel in enumerate(panels):
                strip.paste(panel, (j*P.PX, 0))
            strip.paste(closeups, (len(views)*P.PX, 0))
            strip.paste(balls, (len(views)*P.PX+detail_width, 0))
            strips.append(strip)
    path = HERE/f"area_{S.name}.png"
    D.page(path, strips, report["rows"])
    body = Image.open(path).convert("RGB")
    page = Image.new("RGB", (body.width, body.height+80), P.PAPER)
    page.paste(body, (0, 80))
    draw = ImageDraw.Draw(page)
    headers = [(P.GAP+(j+.5)*P.PX, f'{S.name} contact view {j+1}') for j in range(len(views))]
    headers += [(P.GAP+len(views)*P.PX+detail_width/2, 'Contact close-ups'),
                (P.GAP+len(views)*P.PX+detail_width+P.BALL/2, 'Force equilibrium'),
                (P.GAP+len(views)*P.PX+detail_width+1.5*P.BALL, 'Moment equilibrium')]
    for x, label in headers:
        draw.text((x, 42), label, font=_font(32), fill="#6b6b66", anchor="mm")
    page.save(path)
    report["detail_views"] = [dict(elev=cam.elev, azim=cam.azim,
                                    lookat_m=cam.lookat.tolist(), distance_m=cam.dist,
                                    visible_area_fraction=visibility.tolist())
                               for cam, _, visibility, _ in views]
    report['contact_detail_cameras'] = [dict(seed=seed, elev=cam.elev, azim=cam.azim,
                                              lookat_m=cam.lookat.tolist(), distance_m=cam.dist, crop_pixels=crop)
                                         for seed, (cam, crop) in details.items()]
    seen_union = np.logical_or.reduce([view[1] for view in views])
    for k, row in enumerate(report['rows'], start=1):
        row['combined_visible_area_fraction'] = [
            float(S.skin.area[ids][seen_union & mask[ids]].sum()/S.skin.area[mask].sum())
            for mask in arrays[f'row_masks_{k}']]
    report['combined_visible_area_fraction'] = [float(S.skin.area[ids][seen_union & m[ids]].sum()/S.skin.area[m].sum())
                                                for m in masks]
    (HERE/f"area_{S.name}_results.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects", nargs="*", metavar="{A1-f,B,C5}")
    parser.add_argument("--render-only", action="store_true", help="Reuse verified four-row results")
    parser.add_argument("--from-search", action="store_true", help="Verify and draw a completed saved search")
    parser.add_argument("--no-render", action="store_true", help="Only search and verify")
    args = parser.parse_args()
    if any(name not in P.OBJECTS for name in args.objects):
        parser.error("objects must be A1-f, B or C5")
    for name in args.objects or P.OBJECTS:
        S = load(name)
        if args.render_only:
            report = json.loads((HERE/f"area_{name}_results.json").read_text())
            with np.load(HERE/f"area_{name}_results.npz") as z:
                arrays = {k: z[k] for k in z.files}
            assert report["source_demand_sha256"] == P.I.sha256(HERE/f"demand_{name}_tip1.npz")
            assert report["geometry_sha256"] == S.geometry_sha256
        else:
            from area_search import Search
            if args.from_search:
                search_report = json.loads((HERE/f"area_{name}_search.json").read_text())
                assert search_report['source_demand_sha256'] == P.I.sha256(HERE/f'demand_{name}_tip1.npz')
                assert search_report['geometry_sha256'] == S.geometry_sha256
                assert abs(search_report['base_radius_mm']-1000*S.radius) < 1e-10
                assert search_report['normal_angle_from_seed_max_deg'] == NORMAL_ANGLE
            else:
                searcher = Search(S, region, P.fps)
                try:
                    searcher.run()
                finally:
                    searcher.save_catalog(HERE)
                search_report = json.loads((HERE/f"area_{name}_search.json").read_text())
            report, arrays = verify(S, search_report)
        # Search samples choose useful patches. The figure's coverage verdict
        # requires a separate continuous-domain check, including render-only.
        from continuous import audit
        audit(S, report, arrays)
        from coverage import ensure
        ensure(S, report, arrays)
        if not args.no_render:
            render(S, report, arrays)


if __name__ == "__main__":
    main()
