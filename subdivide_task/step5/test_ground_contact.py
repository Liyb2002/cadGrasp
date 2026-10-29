"""Regression of physical wrenches and work regions across ground frames."""
import json
import unittest

import numpy as np

from subdivide_task.step1.common import ROOT
from subdivide_task.step3.run_membranes import load_model
from subdivide_task.step4.membrane_search import read_group
from subdivide_task.step4.ground_oracle import pressure_rows
from subdivide_task.operations.state import Layout
from .ground_contact import GroundFrameModel,FloorObjective,axial_installations,normalize_ground,resting_heights


class FiniteGroundContactTests(unittest.TestCase):
    def test_point_touch_cannot_replace_a_real_ground_patch(self):
        import trimesh
        from subdivide_task.step3 import independent_geometry as G
        from subdivide_task.step4.ground_oracle import ground_enclosure
        from .geometry import enclosure,unpack
        from .grounded_feet import ContactFeet
        tip=trimesh.Trimesh(vertices=[[0,0,0],[.002,0,.004],[0,.002,.004],[.002,.002,.004]],
                            faces=[]).convex_hull
        pads=[G.box([x-.0025,y-.0025,0],[x+.0025,y+.0025,.005])
              for x,y in ((.02,0),(0,.02))]
        mesh=G.mesh(G.union([G.solid(tip),*pads]))
        hull=np.array([[0,0,0],[.018,0,0],[0,.018,0]])
        rows=[dict(pose='test',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),hull=hull)]
        strict=[dict(id='test',world_from_support=np.eye(4),hull=hull[:,:2])]
        self.assertTrue(enclosure(mesh,rows)[0]['passed'])
        self.assertFalse(ground_enclosure(mesh,strict)[0]['passed'])
        builder=ContactFeet(mesh,G.mesh(G.box([-.04,-.04,0],[.04,.04,.04])),rows)
        builder.ground()
        self.assertTrue(ground_enclosure(unpack(builder.body),strict)[0]['passed'])


@unittest.skipUnless((ROOT/'subdivide_task/data/B/step4_compact_best/report.json').exists(),
                     'Saved membrane fixture is unavailable')
class GroundFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base,_,_=load_model(ROOT/'subdivide_task/data/B/step3_membranes')
        group=json.loads((ROOT/'subdivide_task/data/B/step4_compact_best/report.json').read_text())['groups'][0]
        q,cls.core=read_group(group)
        index=group['members'].index(1)
        cls.base=base.subset([1])
        cls.q=Layout(q.poses[index:index+1],q.material[index:index+1],q.exits[index:index+1])

    def test_ground_coordinates_preserve_tool_work_area_and_paired_loads(self):
        original=self.base.prepare(self.q)[0]
        height=np.array([.08]);model=GroundFrameModel(self.base,height)
        moved=normalize_ground(self.q,height);actual=model.prepare(moved)[0]
        np.testing.assert_allclose(model.children[0].head+[0,0,height[0]],self.base.head,atol=1e-15)
        np.testing.assert_allclose(actual['endpoints']['endpoint_need_wrench'],
                                   original['endpoints']['endpoint_need_wrench'],atol=1e-12,rtol=0.)
        np.testing.assert_allclose(actual['samples']['need_wrench'],
                                   original['samples']['need_wrench'],atol=1e-12,rtol=0.)
        np.testing.assert_allclose(moved.material@moved.poses,self.q.material@self.q.poses,atol=1e-14,rtol=0.)

    def test_fast_rotation_pressures_equal_fresh_full_domain_pressures(self):
        objective=FloorObjective(self.base,self.q,self.core);angle=.3
        fast=objective.one(0,angle)['pressure']
        global_q=axial_installations(self.q,[angle],self.base.axis,self.base.head)
        heights=resting_heights(self.core,global_q)
        model=GroundFrameModel(self.base,heights);q=normalize_ground(global_q,heights)
        actual=pressure_rows(model,q)[0]['ground_material']
        directions=np.random.default_rng(81).normal(size=(48,3))
        np.testing.assert_allclose((fast@directions.T).max(0),(actual@directions.T).max(0),
                                   atol=2e-12,rtol=0.)


if __name__=='__main__':unittest.main()
