"""Current target-pose inputs for the contact-area demos.

Setup's public pose builder runs against temporary copies of mesh.stl and
poses.json. Every generated setup image and rendering scratch file stays in
that temporary directory. Local demand tables retain the complete paired
(q, d, F, tau) sample, at the slides' fixed full load K = 0.5.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'slides/tools'), str(ROOT / 'slides/setup/poses')]
import big_tip as G
import tip_sequence as TS
from cone_model import CONE_HALF_DEG, cone_pushes

K, TIP = 0.5, 1
POINTS, DIRS, SEED = 90, 24, 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def current_pose(name, log=print):
    """Yield the current tip and keep rendering isolated until the page is done."""
    t0 = time.time()
    original = ROOT / 'objects' / name
    with tempfile.TemporaryDirectory(prefix=f'cadgrasp-area-{name}-') as tmp:
        tmp = Path(tmp)
        directory = tmp / name
        directory.mkdir()
        for filename in ('mesh.stl', 'poses.json'):
            shutil.copy2(original / filename, directory / filename)
        old = G.HERE, G.obj_path, TS.obj_path
        try:
            G.HERE = tmp
            G.obj_path = TS.obj_path = lambda requested: directory
            with contextlib.redirect_stdout(io.StringIO()):
                records = G.sweep(name)
            mesh, _ = G.refine(trimesh.load(directory / 'mesh.stl', force='mesh'))
            log(f'\n=== {name} tip {TIP} / {len(records)}; '
                f'current setup rebuilt in isolated scratch ({time.time() - t0:.1f} s)')
            yield SimpleNamespace(directory=directory, mesh=mesh,
                                  record=records[TIP - 1], n_tips=len(records))
        finally:
            G.HERE, G.obj_path, TS.obj_path = old


def demand_table(name, mesh, record, log=print):
    """Regenerate the declared sample and verify any existing local table."""
    T, take = record['T_star'], record['take']
    com = T[:3, :3] @ mesh.center_mass + T[:3, 3]
    q, d = cone_pushes(mesh, T, take, POINTS, DIRS, SEED, CONE_HALF_DEG)
    F = np.array([0., 1., 0.]) - K * d
    M = -1000.0 * K * np.cross(q - com, d)
    assert np.max(np.abs(np.linalg.norm(d, axis=1) - 1)) < 1e-12
    assert np.max(np.abs(np.einsum('ij,ij->i', M, d))) < 1e-10
    path = HERE / f'demand_{name}_tip{TIP}.npz'
    data = dict(q_m=q, d=d, force_w=F, moment_wmm=M, K=K,
                cone_half_deg=CONE_HALF_DEG, points=POINTS, directions=DIRS,
                seed=SEED, tip=TIP, T_world_mesh=T, com_m=com,
                floor_contact_m=np.asarray(record['contact']), work_faces=take,
                mesh_sha256=sha256(ROOT / 'objects' / name / 'mesh.stl'),
                poses_sha256=sha256(ROOT / 'objects' / name / 'poses.json'))
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            for field in ('q_m', 'd', 'force_w', 'moment_wmm', 'K', 'cone_half_deg'):
                if old[field].shape != np.asarray(data[field]).shape or not np.allclose(
                        old[field], data[field], rtol=0, atol=1e-9):
                    raise ValueError(f'{path.name}: {field} disagrees with current setup; '
                                     'remove this derived table to rebuild it')
    np.savez_compressed(path, **data)
    log(f'  demand: {len(F)} paired samples, {POINTS} points x {DIRS} directions, '
        f'seed {SEED}, cone {CONE_HALF_DEG:g} deg, K={K}; current pose verified')
    return data
