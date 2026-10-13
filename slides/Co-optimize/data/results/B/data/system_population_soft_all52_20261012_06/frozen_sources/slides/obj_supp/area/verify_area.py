"""Check saved area demos by direct physical substitution and dual separation.

    python slides/obj_supp/area/verify_area.py

This reads the artifacts only; no search, equilibrium solver or rendering runs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def check_duals(y, matrix, target):
    if not len(y):
        return
    violation = float(np.max(y @ matrix.T))
    gap = float(np.einsum("ij,ij->i", y, target).min())
    assert violation < 1e-8 and gap > 1e-8, (violation, gap)


def verify(name):
    report = json.loads((HERE/f"area_{name}_results.json").read_text())
    for field, path in (("source_demand_sha256", HERE/f"demand_{name}_tip1.npz"),):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report[field]
    with np.load(HERE/f"demand_{name}_tip1.npz") as demand, \
         np.load(HERE/f"area_{name}_results.npz") as z:
        assert float(demand['K']) == .5 and int(demand['tip']) == 1
        for filename, field in (('mesh.stl', 'mesh_sha256'), ('poses.json', 'poses_sha256')):
            path = HERE.parents[2]/'objects'/name/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == str(demand[field])
        assert np.allclose(demand['force_w'], [0, 0, 1]-.5*demand['d'], atol=1e-12, rtol=0)
        assert np.allclose(demand['moment_wmm'],
                           -500*np.cross(demand['q_m']-demand['com_m'], demand['d']), atol=1e-10, rtol=0)
        assert np.array_equal(z['T_world_mesh'], demand['T_world_mesh'])
        final_masks = z["patch_masks"]
        assert np.max(final_masks.sum(axis=0)) == 1
        target = np.c_[demand["force_w"], demand["moment_wmm"]/1000]
        assert np.array_equal(target, z["targets"])
        fingerprint = hashlib.sha256()
        for field in ('skin_centres_m', 'skin_outward_normals', 'source_face', 'skin_face_area_m2'):
            fingerprint.update(z[field].tobytes())
        assert fingerprint.hexdigest() == report['geometry_sha256']
        assert len(report['rows']) == 4
        com, floor = demand["com_m"], demand["floor_contact_m"]

        def geometry(ids):
            p = np.vstack([z["skin_centres_m"][ids], floor[None]])
            u = np.vstack([-z["skin_outward_normals"][ids], [[0, 0, 1]]])
            return p, u, np.c_[u, np.cross(p-com, u)]

        for k, row in enumerate(report["rows"], start=1):
            masks = z[f"row_masks_{k}"]
            assert len(masks) == row['n_regions'] and masks.sum(axis=0).max() == 1
            for patch, mask in zip(row['patches'], masks):
                seed = patch['seed']
                distance = np.linalg.norm(z["skin_centres_m"][mask]-z["skin_centres_m"][seed], axis=1)
                assert distance.max() <= patch['radius_mm']/1000 + 1e-12
                cosine = z["skin_outward_normals"][mask] @ z["skin_outward_normals"][seed]
                assert cosine.min() >= np.cos(np.deg2rad(report["normal_angle_from_seed_max_deg"]))-1e-12
            ids = np.flatnonzero(masks.any(axis=0))
            if k > 1:
                parent = 3 if k == 4 else 1
                assert np.isin(z[f'face_ids_{parent}'], ids).all()
            assert np.array_equal(ids, z[f"face_ids_{k}"])
            p, u, matrix = geometry(ids)
            assert np.allclose(matrix, z[f"columns_{k}"], atol=1e-12, rtol=0)
            ok, x = z[f"ok_joint_{k}"], z[f"multipliers_{k}"]
            assert int(ok.sum()) == row["joint_count"]
            if k > 1:
                assert np.all(~z[f'ok_joint_{parent}'] | ok)
            assert x.min() >= -1e-10
            if ok.any():
                forces = x[ok, :, None] * u[None]
                ferr = float(np.abs(forces.sum(axis=1)-target[ok, :3]).max())
                merr = float(1000*np.abs(np.cross(p[None]-com, forces).sum(axis=1)-target[ok, 3:]).max())
                assert ferr < 1e-8 and merr < 1e-6
                print(name, "row", k, "joint", int(ok.sum()), "force/moment error", ferr, merr)
            else:
                print(name, "row", k, "joint 0; checking all separating certificates")
            check_duals(z[f"dual_{k}"][~ok], matrix*z["scale"], target[~ok]*z["scale"])
            assert abs(1e4*z["skin_face_area_m2"][ids].sum()-row["area_cm2"]) < 1e-10
        assert [row['n_regions'] for row in report['rows']] == [1, 1, 2, 3]
        ratio = report['rows'][1]['area_cm2']/report['rows'][0]['area_cm2']
        assert abs(ratio/3-1) < .005
        assert abs(ratio-report['display_selection']['row2_actual_area_ratio']) < 1e-10
        for row in report['rows']:
            if 'detail_visible_area_fraction' in row:
                assert min(row['detail_visible_area_fraction']) >= .95
        for j, count in enumerate(report["joint_count_without_each_patch"], start=1):
            selected = np.delete(final_masks, j-1, axis=0).any(axis=0)
            _, _, matrix = geometry(np.flatnonzero(selected))
            ok = z[f"without_{j}_ok"]
            assert int(ok.sum()) == count
            check_duals(z[f"without_{j}_dual"][~ok], matrix*z["scale"], target[~ok]*z["scale"])
        search = json.loads((HERE/report['search_file']).read_text())
        assert search['source_demand_sha256'] == report['source_demand_sha256']
        assert search['geometry_sha256'] == report['geometry_sha256']
        shown_final = [[p['seed'], p['level']] for p in report['rows'][-1]['patches']]
        assert shown_final == report['display_selection']['triple_state']
        pair = next(step for step in search['steps'] if len(step['state']) == 2)
        assert pair['state'] == report['display_selection']['pair_state']
        assert pair['joint_count'] == report['rows'][2]['joint_count']
        triple = next((step for step in search['steps'] if len(step['state']) == 3), None)
        if triple:
            assert shown_final == triple['state']
            assert triple['joint_count'] == report['rows'][3]['joint_count']
        else:
            assert report['rows'][2]['joint_count'] == report['rows'][3]['joint_count'] == len(target)
        eligible = [c for c in search['candidates'] if c['eligible']]
        best = max(eligible, key=lambda c: (c['joint_count'], -c['area_cm2'], -c['seed']))
        assert best['seed'] == report['rows'][0]['patches'][0]['seed']
        assert best['joint_count'] == report['rows'][0]['joint_count'] > 0
        with np.load(HERE/report['singleton_file']) as single:
            assert np.array_equal(single['ok_joint'].sum(axis=1),
                                  [c['joint_count'] for c in search['candidates']])
            if triple is None:
                # Independently replay the requested third-support tie-break.
                row_ids = {int(seed): i for i, seed in enumerate(single['seeds'])}
                ranked = sorted(eligible, key=lambda c: (-c['joint_count'], c['area_cm2'], c['seed']))
                for candidate in ranked:
                    i = row_ids[candidate['seed']]
                    start, stop = single['face_offsets'][i:i+2]
                    faces = single['face_ids'][start:stop]
                    if np.intersect1d(faces, z['face_ids_3']).size:
                        continue
                    centre = z['skin_centres_m'][candidate['seed']]
                    normal = z['skin_outward_normals'][candidate['seed']]
                    compatible = True
                    for patch in report['rows'][2]['patches']:
                        distance = np.linalg.norm(centre-z['skin_centres_m'][patch['seed']])
                        if distance < (report['base_radius_mm']+patch['radius_mm'])/1000 and \
                                normal@z['skin_outward_normals'][patch['seed']] >= -.5:
                            compatible = False
                    if compatible:
                        assert candidate['seed'] == shown_final[-1][0]
                        break
                else:
                    raise AssertionError('No compatible third support')
        # Replay the declared decision rule over every recorded alternative.
        def key(move):
            return (move['gain_per_cm2'], move['gain'], -len(move['state']),
                    -move['total_area_cm2'], tuple(tuple(-v for v in x) for x in move['state']))
        for step in search['steps'][1:]:
            assert step['gain'] > 0
            for move in step['compared']:
                assert abs(move['gain']/move['added_area_cm2']-move['gain_per_cm2']) < 1e-7
            winner = max(step['compared'], key=key)
            assert winner['state'] == step['state']
        print(name, "primal/dual evidence, singleton ranking and grow/add decisions passed")


if __name__ == "__main__":
    for name in ("A1-f", "B", "C5"):
        verify(name)
