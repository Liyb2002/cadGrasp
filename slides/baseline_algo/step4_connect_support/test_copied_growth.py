"""Direct thick rods must shortcut free space and still avoid real obstacles."""
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step4_connect_support.run_copied_reference import StagedGrow
from step4_connect_support import build_coupled_saddle as S, growing_support as L
from step2_local_support import geometry as G


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    mesh = trimesh.creation.box(hi-lo)
    mesh.apply_translation((lo+hi)/2)
    return S.solid(mesh)


class CopiedGrowthTests(unittest.TestCase):
    def growth(self, obstacle=False):
        g = StagedGrow.__new__(StagedGrow)
        g.group = SimpleNamespace(name='synthetic-thick')
        g.mask = box([0,0,0],[.07,.07,.07])
        wall = box([.032,0,0],[.038,.05,.05])
        if obstacle:
            g.mask = g.mask-wall
        g.reference = S.unpack(g.mask)
        sphere = trimesh.creation.icosphere(subdivisions=2,radius=.003)
        g.bead = sphere.vertices
        g.guaranteed_radius = float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        g.paths = []
        g.save_stage = lambda *args: None
        g.graph(.004)
        _, ids = g.tree.query([[.012,.026,.026],[.058,.034,.026]])
        g.terminals = [dict(node=int(i),solid=S.solid(G.hull_mesh(g.nodes[i]+g.bead))) for i in ids]
        return g, wall, sphere

    def test_free_space_uses_one_diagonal_rod_without_grid_bends(self):
        g, _, sphere = self.growth()
        solid, report = g.connect()
        self.assertEqual(report['grown_segment_count'],1)
        self.assertEqual(report['direct_connection_count'],1)
        self.assertEqual(len(g.paths[0]),2)
        radius = np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0])))
        self.assertGreater(2*radius,.005)
        self.assertLess(abs((solid-g.mask).volume())*S.SCALE**3,8e-14)

    def test_obstacle_requires_bends_and_preserves_complete_thick_rods(self):
        g, wall, _ = self.growth(obstacle=True)
        solid, report = g.connect()
        self.assertGreater(report['grown_segment_count'],1)
        self.assertGreater(report['removed_grid_bends'],0)
        self.assertLess(abs((solid^wall).volume())*S.SCALE**3,8e-14)
        for rod in g.beams:
            self.assertLess(abs((rod-solid).volume())*S.SCALE**3,8e-14)
        self.assertEqual(len(L.components(solid)),1)


if __name__ == '__main__':
    unittest.main()
