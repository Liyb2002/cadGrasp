"""The coverage gate runs before residual assignment or publishing a plan."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from whole_task.step1 import plan_reach as P
from whole_task.step1.areas import VIEWS


class TabletopPoseTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box(extents=[.12,.18,.08]);self.mesh.apply_translation([.04,.09,-.02])
        self.setup=dict(work_point_m=[0.,-.71,.3],table_height_m=.21,
                        working_direction_world=[0.,-2**-.5,-2**-.5])

    def test_six_input_sides_face_the_robot_with_axes_aligned_and_bottom_on_table(self):
        poses=P.placed(self.mesh,VIEWS,self.setup)
        self.assertEqual(len(poses),6)
        for side,pose in zip(VIEWS,poses):
            rotation=np.asarray(pose['rotation_world_object']);translation=np.asarray(pose['translation_world_m'])
            np.testing.assert_array_equal(rotation,np.round(rotation))
            self.assertAlmostEqual(np.linalg.det(rotation),1.)
            np.testing.assert_allclose(rotation@side,[0,1,0],atol=1e-12)
            vertices=self.mesh.vertices@rotation.T+translation
            self.assertAlmostEqual(vertices[:,2].min(),.21)
            np.testing.assert_allclose((vertices.min(0)+vertices.max(0))[:2]/2,[0,-.71],atol=1e-12)

    def test_changing_tool_pitch_does_not_tilt_the_object(self):
        first=P.placed(self.mesh,VIEWS,self.setup)
        changed=dict(self.setup,working_direction_world=[0.,-.5,-np.sqrt(.75)])
        self.assertEqual(first,P.placed(self.mesh,VIEWS,changed))


class CoverageGateTests(unittest.TestCase):
    def test_insufficient_coverage_cannot_lower_the_goal_and_assign_the_rest(self):
        mesh=trimesh.creation.box();F=len(mesh.faces);axes=-VIEWS
        reached=np.zeros((6,F),bool);reached[:,:10]=True
        tilt=np.full((F,6),np.inf);tilt[:10,0]=0.
        setup=dict(needle_length_m=.1,max_tilt_deg=25.,working_direction_world=[0,0,-1])
        cert=dict(pairs_tried=10,rejected_screened_axes=0,arm_pairs=10,arm_no_ik=0,arm_hits_object=0)
        result=(tilt,np.zeros((F,6,3)),np.zeros((F,6,2)),np.zeros((F,6,4,7)),np.zeros((F,6)),cert)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(P,'ROOT',Path(directory)), \
             patch.object(P,'setup',return_value=(mesh,None,np.ones(F,bool),setup)), \
             patch.object(P,'screen',return_value=(axes,reached,.001)), \
             patch.object(P.R,'sphere_axes',return_value=axes), \
             patch('whole_task.step1.reach_video.work_point',return_value=(np.zeros(3),0.)), \
             patch.object(P,'ProcessPoolExecutor'), \
             patch.object(P,'certify_directions',return_value=result), \
             patch.object(P,'areas') as split:
            with self.assertRaisesRegex(ValueError,'below the required 99.5000%'):
                P.plan(render=False,allow_partial=False)
            split.assert_not_called()
            self.assertFalse((Path(directory)/'whole_task/data/B/step1_reach/plan.json').exists())


if __name__=='__main__':
    unittest.main()
