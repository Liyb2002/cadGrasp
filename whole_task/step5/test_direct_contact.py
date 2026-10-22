import unittest

import numpy as np
import trimesh

from whole_task.operations.state import Layout
from .direct_contact import check


class DirectContactTests(unittest.TestCase):
    def scene(self):
        core=trimesh.creation.box([.1,.1,.01]);core.apply_translation([0,0,.205])
        body=trimesh.creation.box([.05,.05,.05]);body.apply_translation([0,0,.25])
        layout=Layout(np.eye(4)[None],np.eye(4)[None],np.array([[0.,0.,1.]]))
        return core,body,layout

    def test_a_long_leg_does_not_excuse_a_floating_contact_membrane(self):
        core,body,q=self.scene()
        leg=trimesh.creation.box([.01,.01,.2]);leg.apply_translation([0,0,.1])
        final=trimesh.util.concatenate([core,leg])
        self.assertAlmostEqual(final.bounds[0,2],0.)
        result=check(core,body,q,['task'])
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['tasks'][0]['shell_gap_m'],.2)

    def test_explicit_raised_plane_reaches_membrane_without_moving_object(self):
        core,body,q=self.scene();original=q.poses.copy()
        self.assertTrue(check(core,body,q,['task'],floor_heights=.2)['passed'])
        np.testing.assert_array_equal(q.poses,original)

    def test_tangent_membrane_is_rejected_if_object_would_cross_the_plane(self):
        core,body,q=self.scene();body.apply_translation([0,0,-.04])
        result=check(core,body,q,['task'],floor_heights=.2)
        self.assertTrue(result['tasks'][0]['shell_touches_ground'])
        self.assertFalse(result['passed'])


if __name__=='__main__':unittest.main()
