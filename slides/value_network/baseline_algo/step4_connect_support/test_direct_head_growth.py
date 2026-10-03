"""Actual head starts grow directly; graph routing is only an obstacle fallback."""
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step4_connect_support.direct_head_growth import DirectGrow
from step4_connect_support import build_coupled_saddle as S, growing_support as L


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    mesh = trimesh.creation.box(hi-lo)
    mesh.apply_translation((lo+hi)/2)
    return S.solid(mesh)


class DirectHeadTests(unittest.TestCase):
    def growth(self, positions, wall=None):
        g = DirectGrow.__new__(DirectGrow)
        g.group = SimpleNamespace(name='synthetic-head-starts')
        g.mask = box([0,0,0],[.08,.08,.08])
        if wall is not None:g.mask = g.mask-wall
        g.reference = S.unpack(g.mask)
        sphere = trimesh.creation.icosphere(subdivisions=2,radius=.003)
        g.bead = sphere.vertices
        g.guaranteed_radius = float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        g.seed_positions = [np.asarray(p) for p in positions]
        g.terminals = [dict(name=f'head{i}',node=i,kind='head_start',solid=g.sphere(p))
                       for i,p in enumerate(g.seed_positions)]
        g.paths=[];g.segments=[];g.journal=[];g.grid_ready=False;g.partial_steps=0;g.pitch=.004
        g.saved={};g.save_stage=lambda name,solid:g.saved.update({name:solid})
        return g

    def test_off_grid_head_starts_connect_without_constructing_grid(self):
        starts=[[.0131,.0257,.0209],[.0563,.0319,.0237]]
        g=self.growth(starts);solid,report=g.connect()
        self.assertFalse(g.grid_ready)
        self.assertFalse(hasattr(g,'nodes'))
        self.assertFalse(report['preset_grid_anchor'])
        self.assertEqual(report['grown_segment_count'],1)
        np.testing.assert_array_equal(g.journal[0]['path_m'],starts)
        self.assertEqual(len(L.components(solid)),1)

    def test_growth_branches_from_existing_rod_projection(self):
        g=self.growth([[.01,.03,.03],[.05,.03,.03],[.03,.07,.03]])
        solid,report=g.connect()
        self.assertEqual(report['projection_shared_joint_count'],1)
        np.testing.assert_allclose(g.journal[1]['path_m'][0],[.03,.03,.03])
        self.assertFalse(g.grid_ready)
        self.assertEqual(len(L.components(solid)),1)
        # The recorded partial snapshot precedes reaching the third head.
        self.assertEqual(g.partial_steps,1)
        self.assertLess(abs((g.saved['partial_growth']^g.terminals[2]['solid']).volume())*S.SCALE**3,8e-14)

    def test_volume_increment_precedes_shorter_connection(self):
        g=self.growth([[.01,.02,.03],[.01,.02,.045],[.06,.02,.03]])
        g.bases=np.array([np.eye(3)]);g.offsets=np.zeros((1,3))
        g.occupied_lo=np.array([0.,0.,0.]);g.occupied_hi=np.array([.07,.04,.04])
        candidates=g.candidates({1,2},{0})
        # Longer horizontal rod fits the existing box; short vertical expands it.
        self.assertEqual(candidates[0][2],2)
        self.assertGreater(candidates[-1][0],0.)
        self.assertEqual(candidates[0][0],0.)

    def test_wall_triggers_lazy_route_with_complete_unclipped_rods(self):
        wall=box([.032,0,0],[.038,.05,.05])
        g=self.growth([[.012,.026,.026],[.066,.034,.026]],wall)
        solid,report=g.connect()
        self.assertTrue(g.grid_ready)
        self.assertEqual(report['fallback_route_count'],1)
        self.assertGreater(report['grown_segment_count'],1)
        self.assertLess(abs((solid^wall).volume())*S.SCALE**3,8e-14)
        for rod in g.beams:self.assertLess(abs((rod-solid).volume())*S.SCALE**3,8e-14)
        np.testing.assert_allclose(g.journal[0]['path_m'][0],g.seed_positions[0])
        np.testing.assert_allclose(g.journal[0]['path_m'][-1],g.seed_positions[1])


if __name__=='__main__':unittest.main()
