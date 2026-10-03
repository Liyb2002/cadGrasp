"""Regular reconstruction must retain the network without filling huge voids."""
from pathlib import Path
import sys
import unittest

import manifold3d as md
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import build_coupled_saddle as S
from step4_connect_support import regular_body_geometry as B, material_graph as M


class RegularBodyTests(unittest.TestCase):
    def setUp(self):
        # Two nearby heads lead to a long right-angle material branch.
        xy = [(x, 0) for x in range(11)]+[(10, y) for y in range(1, 11)]
        solids = [md.Manifold.cube([.01/S.SCALE]*3).translate([x*.01/S.SCALE, y*.01/S.SCALE, 0]) for x, y in xy]
        meshes = [S.unpack(s) for s in solids]
        self.graph = dict(vertices=[m.vertices for m in meshes], faces=[m.faces for m in meshes],
            edges=np.array([[i, i+1] for i in range(len(xy)-1)]), cost=np.r_[0., 0., np.ones(len(xy)-2)], heads=[0, 1])
        self.context = dict(heads=solids[:2], forbidden=md.Manifold(), bases=np.array([np.eye(3)]*2), offsets=np.zeros((2, 3)))
        self.selected = np.ones(len(xy), bool)

    def test_split_limits_void_filling_while_preserving_every_original_cell(self):
        large, loose = B.reconstruct(self.graph, self.selected, self.context, 100.)
        full, report = B.reconstruct(self.graph, self.selected, self.context, 2.)
        self.assertGreater(len(report['local_hulls']), len(loose['local_hulls']))
        self.assertLess(full.volume(), large.volume())
        self.assertTrue(all(r['fill_ratio'] <= 2.+1e-9 for r in report['local_hulls']))
        seed = M.material_solid(self.graph, self.selected)
        self.assertLess((seed-full).volume()*S.SCALE**3, 8e-14)
        for head in self.context['heads']:
            self.assertLess((head-full).volume()*S.SCALE**3, 8e-14)
        self.assertEqual(len(full.decompose()), 1)
        self.assertTrue(S.unpack(full).is_watertight)

    def test_reconstruction_does_not_hide_a_disconnected_input(self):
        self.selected[8] = False
        with self.assertRaisesRegex(RuntimeError, 'disconnected or incomplete'):
            B.reconstruct(self.graph, self.selected, self.context)


if __name__ == '__main__': unittest.main()
