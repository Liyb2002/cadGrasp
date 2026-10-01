"""Exact shared-material accounting and physical-connectivity regressions."""
from pathlib import Path
import sys
import unittest

import manifold3d as md
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step4_connect_support import material_graph as M, material_network as N, build_coupled_saddle as S
from step4_connect_support import loft_growth as G, loft_acceptance as A
from step4_connect_support.loft_candidates import volume


def box(low, size):
    # Test dimensions are centimetres; costs are cubic centimetres.
    return md.Manifold.cube((np.array(size)*.01/S.SCALE).tolist()).translate((np.array(low)*.01/S.SCALE).tolist())


class LoftGrowthTests(unittest.TestCase):
    def test_overlap_in_a_multi_loft_action_is_paid_once(self):
        current=box([0,0,0],[1,1,1]); a=box([.5,0,0],[1,1,1]); b=box([1,0,0],[1,1,1])
        addition=a+b
        delta=G.marginal_volume(addition,current,current)
        self.assertAlmostEqual(delta,1.)
        self.assertAlmostEqual(volume(current+addition)-volume(current),delta)
        self.assertLess(delta,G.marginal_volume(a,current,current)+G.marginal_volume(b,current,current))

    def test_one_complete_body_can_connect_heads_and_cover_both_floors(self):
        solids=[box([0,0,0],[1,1,1]),box([4,0,0],[1,1,1]),
            box([0,0,0],[5,1,1]),box([0,0,0],[2,2,1]),box([3,0,0],[2,2,1])]
        fixed=solids[0]+solids[1]
        edges=np.array([(i,j) for i in range(len(solids)) for j in range(i) if volume(solids[i]^solids[j])>1e-8])
        graph=dict(solids=solids,heads=[0,1],edges=edges,cost=np.array([volume(s-fixed) for s in solids]))
        groups=[N.Group(np.array([0]),'head'),N.Group(np.array([1]),'head'),
            N.Group(np.array([2,3]),'floor',0),N.Group(np.array([2,4]),'floor',1)]
        selected,history=G.greedy(graph,groups,0)
        self.assertTrue(selected[:3].all())
        self.assertFalse(selected[3:].any())
        self.assertTrue(N.connected_selection(selected,edges,len(solids)))
        self.assertAlmostEqual(sum(r['added_union_volume_cm3'] for r in history),3.)
        self.assertEqual(len(history),1)

    def test_numerical_cleanup_never_hides_a_real_disconnected_body(self):
        a=box([0,0,0],[1,1,1]); b=box([4,0,0],[1,1,1])
        with self.assertRaisesRegex(RuntimeError,'genuinely disconnected'):
            A.remove_numerical_debris(a+b)
        tiny=box([4,0,0],[.0001,.0001,.0001])
        full,report=A.remove_numerical_debris(a+tiny)
        self.assertEqual(len(full.decompose()),1)
        self.assertAlmostEqual(volume(full),1.)
        self.assertLess(report['discarded_volume_m3'],8e-14)

    def test_short_joint_and_foot_bodies_are_chosen_in_the_same_search(self):
        solids=[box([0,0,0],[1,1,1]),box([4,0,0],[1,1,1]),
            box([0,0,0],[2,1,1]),box([3,0,0],[2,1,1]),
            box([1.5,0,0],[2,1,1]),box([0,0,0],[5,2,1])]
        fixed=solids[0]+solids[1]
        edges=np.array([(i,j) for i in range(len(solids)) for j in range(i) if volume(solids[i]^solids[j])>1e-8])
        graph=dict(solids=solids,heads=[0,1],edges=edges,cost=np.array([volume(s-fixed) for s in solids]))
        groups=[N.Group(np.array([0]),'head'),N.Group(np.array([1]),'head'),
            N.Group(np.array([2,5]),'floor',0),N.Group(np.array([3,5]),'floor',1)]
        selected,history=G.greedy(graph,groups,0)
        self.assertTrue(selected[:5].all())
        self.assertFalse(selected[5])
        self.assertTrue(N.connected_selection(selected,edges,len(solids)))
        self.assertAlmostEqual(sum(row['added_union_volume_cm3'] for row in history),3.)

    def test_growing_one_body_prices_the_material_between_its_feet(self):
        head=box([0,0,1],[1,1,1]);vertices=S.unpack(head).vertices
        context=dict(heads=[head],records=[dict(head_pose='pose_a')],
            directions=np.array([[0.,0.,1.]]),bases=np.array([np.eye(3)]),
            offsets=np.zeros((1,3)),forbidden=md.Manifold())
        solids=[head];records=[dict(kind='immutable_head',head=0)]
        for x in (-.02,.03):
            xy=np.array([[x,-.004],[x+.004,-.004],[x+.004,.004],[x,.004]])
            bottom=np.c_[xy,np.zeros(4)]
            solids.append(S.solid(S.G.hull_mesh(np.vstack([vertices,vertices+[0,0,.008],
                bottom,bottom+[0,0,.003]]))))
            records.append(dict(kind='head_floor_loft',head=0,floor=0,polygon_xy_m=xy))
        graph=dict(solids=solids,vertices=[vertices],records=records)
        G.enable_whole_body_growth(graph,context)
        first=G.selection_solid(graph,np.array([True,True,False]))
        grown=G.selection_solid(graph,np.ones(3,bool))
        self.assertLess(volume(M.union(solids)),volume(grown))
        self.assertLess(volume(first-grown),1e-9)
        self.assertAlmostEqual(G.marginal_volume(grown,first,head),volume(grown)-volume(first))
        self.assertEqual(len(grown.decompose()),1)


if __name__=='__main__': unittest.main()
