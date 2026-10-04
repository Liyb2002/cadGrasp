import unittest
from types import SimpleNamespace
import numpy as np
import experiment as E
from value import absolute_spread,alignment,terminal_cost,action_target

class ValueTests(unittest.TestCase):
    def test_world_direction_not_object_registration(self):
        # Upright and sideways poses both leave along the workstation +Z.
        self.assertEqual(absolute_spread([[0,0,1],[0,0,1]]),0)
        self.assertEqual(absolute_spread([[0,0,1],[0,0,-1]]),.5)
    def test_subdividing_connected_head_does_not_change_score(self):
        first=absolute_spread([[0,0,1],[1,0,0]],[3,2])
        second=absolute_spread([[0,0,1],[0,0,1],[1,0,0]],[1,2,2])
        self.assertAlmostEqual(first,second)
        self.assertAlmostEqual(alignment([[0,0,1]], [0,0,1],[1]),0)
    def test_final_score_is_measured_volume(self):
        result=dict(object_poses=dict(box_volume_cm3=100),object_and_support_poses=dict(box_volume_cm3=120))
        self.assertAlmostEqual(terminal_cost(result,True),.2)
        self.assertIsNone(action_target([(result,False)]))
        with self.assertRaises(ValueError):terminal_cost(result,False)
    def test_static_support_and_upward_object_exit(self):
        mesh=E.trimesh.creation.box([.1,.1,.1]);mesh.apply_translation([0,0,.06])
        support=E.trimesh.creation.box([.02,.02,.004]);support.apply_translation([0,0,.008])
        d=np.array([0.,0.,-1.])
        old=E.W.Analyzer(mesh,.0004,dict(vectors=[d.tolist()]))
        new=E.W.Analyzer(mesh,.0004,dict(vectors=[d.tolist()],object_withdrawal_from_static_support=True))
        self.assertEqual(old.test([support],d)['reason'],'initial_floor_direction')
        self.assertTrue(new.test([support],d)['clear'])
        self.assertEqual(new.test([support],-d)['reason'],'object_moves_below_floor')
        embedded=E.trimesh.creation.box([.02,.02,.02]);embedded.apply_translation([0,0,.04])
        self.assertFalse(new.test([embedded],d)['clear'])
    def test_internal_cavity_is_one_material_component(self):
        md=E.F.md
        hollow=md.Manifold.cube([1.,1.,1.])-md.Manifold.cube([.2,.2,.2]).translate([.4,.4,.4])
        self.assertEqual(len(hollow.decompose()),2)
        self.assertEqual(len(E.material_components(hollow)),1)
    def test_modified_withdrawal_is_cloned_code(self):
        self.assertTrue(str(E.W.__file__).startswith(str(E.BASE)))

if __name__=='__main__':unittest.main()
