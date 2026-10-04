"""World-direction identity must not depend on pose-local menu IDs."""
import unittest
from types import SimpleNamespace
import numpy as np
from step3_scheculer.shared_direction_joint import direction_menu,local_ray_ids,unit

class SharedDirectionTests(unittest.TestCase):
    def search(self,vectors):return SimpleNamespace(catalogue=dict(vectors=vectors))
    def test_matches_vectors_not_indices_and_is_not_fixed_up(self):
        a=self.search([[-1,0,0],[0,0,-1]])
        b=self.search([[0,-1,0],[-1,0,0]])
        menu=direction_menu([a,b])
        self.assertEqual(len(menu),1)
        np.testing.assert_allclose(menu[0],[1,0,0])
        self.assertEqual(local_ray_ids(a,menu[0]),{0})
        self.assertEqual(local_ray_ids(b,menu[0]),{1})
    def test_empty_intersection_is_not_fabricated_up_solution(self):
        self.assertEqual(direction_menu([self.search([[-1,0,0]]),self.search([[0,-1,0]])]),[])
    def test_full_three_dimensional_oblique_direction(self):
        ray=unit([1,2,3])
        menus=direction_menu([self.search([-ray]),self.search([-2*ray])])
        self.assertEqual(len(menus),1)
        np.testing.assert_allclose(menus[0],ray)
    def test_rejects_zero_direction(self):
        with self.assertRaises(ValueError):unit([0,0,0])

if __name__=='__main__':unittest.main()
