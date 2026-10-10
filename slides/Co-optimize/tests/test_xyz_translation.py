"""Raised tasks lose physical floor forces; XYZ motion preserves COM demands."""
from collections import OrderedDict
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.common import C
from whole_search.model import Model,Layout
from whole_search.fast_search import FastModel
from juxtapose.reuse_search import JumpScreen
from translation.layout_update import shift,derivative,translation_frame
from whole_search.saved_initialization import check_saved_report


class XYZTranslationTests(unittest.TestCase):
    def setUp(self):
        self.model=FastModel.__new__(FastModel)
        self.model.allow_z_translation=True
        self.model.native=np.array([np.eye(4),np.eye(4)])
        self.model.native[1,:3,:3]=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        self.layout=Layout(np.array([np.eye(4),np.eye(4)]),np.array([[0.,0.,1.],[1.,0.,0.]]),np.array([0,1]),(0,1))
        floor=C.U.floor(C.FLOOR.columns(np.zeros(3),np.zeros(3)),np.ones(6))
        self.model.floors=[floor.copy(),floor.copy()]
        self.model.point_rays=[np.zeros((2,7)),np.zeros((2,7))]

    def test_lifting_removes_four_floor_forces_but_retains_slack_and_stable_head_ids(self):
        flags=np.array([True,False])
        ground,ids0=self.model.supply_at_points(1,flags,self.layout)
        frame=translation_frame(self.model,self.layout,1)
        raised=shift(self.model,self.layout,1,.003*frame[:,2])
        self.assertAlmostEqual(self.model.workpiece_height(raised,1),.003)
        air,ids1=self.model.supply_at_points(1,flags,raised)
        self.assertEqual(ground.shape,(6,7));self.assertEqual(air.shape,(2,7))
        np.testing.assert_array_equal(air[0],[0.,0.,0.,0.,0.,0.,-1.])
        np.testing.assert_array_equal(ids1,[4,5])
        self.assertEqual(ids0[-1],ids1[-1])

    def test_loss_cache_distinguishes_floor_state_when_contact_flags_are_identical(self):
        flags={0:np.zeros(2,bool)}
        self.model.contact_delta=SimpleNamespace(state=lambda q:SimpleNamespace(available=flags))
        screen=JumpScreen.__new__(JumpScreen);screen.model=self.model
        target=C.U.target(np.array([[0.,0.,1.,0.,0.,0.]]))
        screen.demands=[(target,np.ones(1),1.)];screen.cache=OrderedDict();screen.evaluations=0;screen.seconds=0.
        ground=self.layout.copy();ground.active=(0,)
        raised=shift(self.model,ground,0,np.array([0.,0.,.003]))
        self.assertLess(screen.evaluate(ground)['loss'],1e-25)
        self.assertGreater(screen.evaluate(raised)['loss'],.1)
        self.assertLess(screen.evaluate(ground)['loss'],1e-25)
        self.assertEqual(len(screen.cache),2)

    def test_projected_motion_never_crosses_floor_and_orientation_is_unchanged(self):
        frame=translation_frame(self.model,self.layout,1)
        raised=shift(self.model,self.layout,1,.003*frame[:,2])
        lowered=shift(self.model,raised,1,-.01*frame[:,2])
        self.assertAlmostEqual(self.model.workpiece_height(lowered,1),0.)
        self.model.validate_placement(raised,1);self.model.validate_placement(lowered,1)
        invalid=self.layout.copy();invalid.placements[0,2,3]=-.001
        with self.assertRaises(ValueError):self.model.validate_placement(invalid,0)
        self.model.allow_z_translation=False
        with self.assertRaises(ValueError):self.model.validate_placement(raised,1)

    def test_vertical_difference_is_one_sided_on_ground_and_symmetric_in_air(self):
        self.model.proxy=lambda q:dict(loss=sum((self.model.workpiece_height(q,k)-.02)**2 for k in q.active))
        slope=derivative(self.model,self.layout,[0],2,.001)
        self.assertAlmostEqual(slope,-.039)
        raised=shift(self.model,self.layout,0,np.array([0.,0.,.01]))
        self.assertAlmostEqual(derivative(self.model,raised,[0],2,.001),-.02)

    def test_saved_step41_remains_verifiable_with_its_original_source(self):
        path=Path(__file__).resolve().parents[1]/'output/B/pose1+2+3+4+5+6+7+8/step4/step4.1/data/report.json'
        report=check_saved_report(path)
        self.assertEqual(len(report['poses']),8)


if __name__=='__main__':unittest.main()
