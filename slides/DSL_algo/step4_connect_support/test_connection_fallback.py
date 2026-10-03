"""Real-solid fallback tests: detours, maximal free domain and blocked space."""
from pathlib import Path
import sys
import unittest

import manifold3d as md
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import connection_fallback as F
from step4_connect_support import build_coupled_saddle as S


def block(size, center):
    mesh = trimesh.creation.box(size)
    mesh.apply_translation(center)
    return S.solid(mesh)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.full = block([.012]*3, [-.04,0,.02])+block([.012]*3, [.04,0,.02])

    def check_connector(self, obstacle):
        carve = lambda value:value.trim_by_plane([0,0,1],0)-obstacle
        joined, bridge, record = F.connect(self.full, carve, S.SCALE, S.unpack(obstacle).bounds)
        self.assertEqual(len(joined.decompose()),1)
        self.assertTrue(S.unpack(joined).is_watertight)
        self.assertLess(abs((self.full-joined).volume())*S.SCALE**3,8e-14)
        self.assertLess(abs((bridge^obstacle).volume())*S.SCALE**3,8e-14)
        self.assertGreaterEqual(S.unpack(bridge).vertices[:,2].min(),-1e-10)
        replay, _ = F.replay(self.full, record, carve, S.SCALE)
        self.assertLess(abs((replay-joined).volume())*S.SCALE**3,8e-14)
        self.assertLess(abs((joined-replay).volume())*S.SCALE**3,8e-14)
        return record

    def test_detours_around_wall_instead_of_leaving_a_cut_bridge(self):
        obstacle = block([.018,.030,.15],[0,0,.03])
        record = self.check_connector(obstacle)
        self.assertEqual(record['phase'],'local_expansion')
        self.assertGreater(record['padding_m'],.004)
        self.assertTrue(any(not r['connected'] for r in record['attempts']))

    def test_wide_wall_uses_full_legal_envelope(self):
        obstacle = block([.018,.28,.28],[0,0,.03])
        record = self.check_connector(obstacle)
        self.assertEqual(record['phase'],'full_free_envelope')

    def test_genuinely_split_domain_is_not_reported_as_connected(self):
        # An infinite forbidden slab, represented with two clipping planes,
        # separates the available domain regardless of envelope size.
        def carve(value):
            left = value.trim_by_plane([-1,0,0],.01/S.SCALE)
            right = value.trim_by_plane([1,0,0],.01/S.SCALE)
            return left+right
        with self.assertRaises(RuntimeError) as context:
            F.connect(self.full,carve,S.SCALE,np.array([[-.01,-.1,-.1],[.01,.1,.1]]))
        self.assertIn('bounded envelope',context.exception.connection_diagnostic['scope'])


if __name__ == '__main__':
    unittest.main()
