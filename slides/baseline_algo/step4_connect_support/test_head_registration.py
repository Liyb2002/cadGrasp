"""Regressions for the actual shared-head split and floor certificate."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import manifold3d as md

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import head_registration as H, greedy_geometry as GG
from step4_connect_support.run_greedy import clear_previous_outputs
from step4_connect_support import build_local_bodies as L, run_fast_local_bodies as F


def fixture():
    rotations = np.array([np.eye(3), [[0., -1, 0], [1, 0, 0], [0, 0, 1]]])
    translations = np.array([[0., 0, 0], [.3, -.2, 0]])
    offsets = np.array([-t@b for t, b in zip(translations, rotations)])
    tetra = np.array([[0., 0, 1], [.1, 0, 1], [0, .1, 1], [0, 0, 1.1]])
    local = {name:tetra+[i*.2, 0, 0] for i, name in enumerate('ABCDE')}
    groups, cells, tasks = [], [], []
    for k, names in enumerate(('ABC', 'ADE')):
        row, solids = [], []
        for name in names:
            world = local[name]@rotations[k].T+translations[k]
            row.append(dict(candidate_id=name, triangles_m=world[:3][None]))
            solids.append([world])
        groups.append(row); cells.append(solids)
        transform = np.eye(4); transform[:3, :3] = rotations[k]; transform[:3, 3] = translations[k]
        tasks.append(SimpleNamespace(pose=f'pose_{k}', floor=np.zeros(3),
            domain=SimpleNamespace(data={'frame':{'T_world_mesh':transform}})))
    return groups, cells, rotations, offsets, tasks


class RegistrationTests(unittest.TestCase):
    def test_nontrivial_transforms_keep_one_shared_solid(self):
        groups, cells, bases, offsets, tasks = fixture()
        actual_bases, actual_offsets = H.fixed_placements(tasks)
        np.testing.assert_allclose(actual_bases, bases, atol=1e-14)
        np.testing.assert_allclose(actual_offsets, offsets, atol=1e-14)
        heads, check = H.register(groups, cells, bases, offsets)
        self.assertEqual([h.ident for h in heads], list('ABCDE'))
        self.assertEqual(heads[0].active_poses, (0, 1))
        self.assertLess(check['shared_heads'][0]['contact_surface_error_m'], 1e-14)
        constructor = GG.Constructor(groups, cells, np.array([[1., 0, 0], [0, 1., 0]]),
                                     bases, offsets, md.Manifold())
        self.assertEqual(len(constructor.heads), 5)
        self.assertEqual(len(constructor.bodies), 5)

    def test_separated_duplicate_is_rejected_not_silently_merged(self):
        groups, cells, bases, offsets, _ = fixture()
        offsets[1, 0] += .2
        with self.assertRaisesRegex(ValueError, 'Shared head A is split'):
            GG.Constructor(groups, cells, np.array([[1., 0, 0]]*2), bases, offsets, md.Manifold())

    def test_fast_default_rejects_split_before_construction(self):
        import tempfile
        groups, cells, bases, offsets, tasks = fixture()
        offsets[1, 0] += .2
        case = SimpleNamespace(groups=groups, heads=cells, tasks=tasks,
                               demands=[np.array([[0., 0.]])]*2)
        placement = dict(bases=bases, offsets=offsets)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(F, 'recipe', return_value=(case, placement, {}, {})), \
                patch.object(L, 'build') as build:
            with self.assertRaises(H.SharedHeadRegistrationError) as caught:
                F.run_pair(Path(tmp)/'pose1+3', verify=False)
            build.assert_not_called()
        check = caught.exception.registration_checks
        self.assertFalse(check['saved_layout']['passed'])
        self.assertEqual(check['exact_registration']['registration']['physical_head_count'], 5)
        self.assertFalse(check['exact_registration']['constructed'])

    def test_direct_constructor_cannot_bypass_identity_with_audits_disabled(self):
        import tempfile
        groups, cells, bases, offsets, tasks = fixture()
        offsets[1, 0] += .2
        case = SimpleNamespace(groups=groups, heads=cells, tasks=tasks,
                               source=None, schedule={}, paths=[])
        placement = dict(bases=bases, offsets=offsets, directions=np.array([[1.,0,0]]*2))
        with tempfile.TemporaryDirectory() as tmp:
            work, cache = Path(tmp)/'work', Path(tmp)/'cache'
            with self.assertRaisesRegex(ValueError, 'Shared head A is split'):
                L.build(work, [], case=case, placement=placement, verify=False,
                        allow_failed=True, cache_dir=cache)
            self.assertFalse(work.exists())
            self.assertFalse(cache.exists())

    def test_registration_failure_retires_stale_public_shape(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pair = Path(tmp)/'pose1+3'
            out = pair/'step4'
            (out/'data').mkdir(parents=True)
            (out/'shape.obj').write_text('invalid six-patch geometry')
            (out/'data/report.json').write_text('{"passed": true}')
            error = H.SharedHeadRegistrationError('Shared head is split', {'saved_layout': {'passed': False}})
            F.publish_failure(pair, F.failure_record(pair, error))
            report = json.loads((out/'data/report.json').read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['complete'])
            self.assertEqual(report['status'], 'shared_head_registration_failed')
            self.assertFalse((out/'shape.obj').exists())
            old = out/'data/history/before_fast_construction/shape.obj'
            self.assertEqual(old.read_text(), 'invalid six-patch geometry')

    def test_same_surface_but_different_head_solid_is_rejected(self):
        groups, cells, bases, offsets, _ = fixture()
        cells[1][0][0][-1, 2] += .01
        with self.assertRaisesRegex(ValueError, 'head_solid_vertex_error|solid error'):
            H.register(groups, cells, bases, offsets)

    def test_six_distinct_ids_do_not_satisfy_three_plus_two(self):
        groups, cells, bases, offsets, _ = fixture()
        groups[1][0]['candidate_id'] = 'F'
        with self.assertRaisesRegex(ValueError, 'five IDs'):
            H.register(groups, cells, bases, offsets)

    def test_floor_halfplane_detects_known_impossible_cop(self):
        bases = np.array([np.eye(3), [[0., 1, 0], [0, 0, 1], [1, 0, 0]]])
        tasks = [SimpleNamespace(pose=str(k), floor=np.zeros(3)) for k in range(2)]
        demands = [np.array([[-.02, .1], [.03, .1]]), np.array([[.1, .02]])]
        check = H.floor_compatibility(tasks, demands, bases, np.zeros((2, 3)))
        self.assertFalse(check['passed'])
        self.assertEqual(check['per_pose'][0]['violating_sample_count'], 1)
        self.assertAlmostEqual(check['per_pose'][0]['minimum_other_floor_height_m'], -.02)
        demands[0][0, 0] = .02
        self.assertTrue(H.floor_compatibility(tasks, demands, bases, np.zeros((2, 3)))['passed'])

    def test_failed_rerun_removes_stale_pass_meshes_but_preserves_other_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ('fixture_mm.stl', 'equilibrium.npz', 'overview.png', 'body5.obj', 'notes.md'):
                (out/name).write_text('previous')
            (out/'historical_shared_head_audit.json').write_text('original six-patch evidence')
            (out/'shared_head_audit.json').write_text('latest five-head check')
            clear_previous_outputs(out)
            self.assertEqual({p.name for p in out.iterdir()}, {'notes.md', 'historical_shared_head_audit.json'})
            self.assertEqual((out/'historical_shared_head_audit.json').read_text(), 'original six-patch evidence')


if __name__ == '__main__':
    unittest.main()
