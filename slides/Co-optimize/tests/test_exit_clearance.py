
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from exit_clearance import *

class Clearance(unittest.TestCase):
    def setUp(self):
        self.obj=trimesh.creation.box([.01,.012,.008]);self.seed=S.solid(trimesh.creation.box([.03,.03,.03]));self.policy=ExitClearance(self.obj)
    def test_clearance_contains_requested_ball_and_does_not_change_object(self):
        vertices=self.obj.vertices.copy();p=self.policy
        self.assertAlmostEqual(p.radius,.00012)
        np.testing.assert_array_equal(self.obj.vertices,vertices)
        self.assertLess(material_volume(S.solid(self.obj)-S.solid(p.expanded)),1e-14)
        self.assertGreaterEqual(p.metadata['maximum_kernel_radius_m'],p.radius)
    def test_seating_survives_but_margin_is_kept_elsewhere(self):
        p=self.policy;d=np.array([0,0,.04]);nom=p.sweep(d,padded=False);pad=p.sweep(d)
        r=p.construct(self.seed,[nom],[pad],np.arange(len(self.obj.faces)))
        self.assertTrue(r['diagnostics']['geometry_resolved'])
        self.assertTrue(r['diagnostics']['contact_area_preserved'])
        self.assertGreater(r['diagnostics']['remaining_contact_area_m2'],0.)
        self.assertLess(material_volume(r['remaining']^nom),1e-14)
        self.assertLess(material_volume((r['remaining']-r['protected_contact_core'])^pad),1e-14)
        self.assertGreater(material_volume(r['nominal_remaining']-r['remaining']),0.)
    def test_changing_direction_does_not_regrow_inside_current_margin(self):
        p=self.policy;allowed=np.arange(len(self.obj.faces));first=p.construct(self.seed,[p.sweep([0,0,.04],padded=False)],[p.sweep([0,0,.04])],allowed)
        second=p.construct(self.seed,[p.sweep([.04,0,0],padded=False)],[p.sweep([.04,0,0])],allowed)
        self.assertGreater(material_volume(second['remaining']-first['remaining']),0.)
        self.assertLess(material_volume((second['remaining']-second['protected_contact_core'])^second['padded_cut']),1e-14)
        self.assertTrue(second['diagnostics']['contact_area_preserved'])

    def test_other_pose_keeps_blocking_restoration_with_its_clearance(self):
        p=self.policy;allowed=np.arange(len(self.obj.faces))
        up=p.sweep([0,0,.04],padded=False);padded_up=p.sweep([0,0,.04])
        first=p.construct(self.seed,[up,p.sweep([.04,0,0],padded=False)],[padded_up,p.sweep([.04,0,0])],allowed)
        second=p.construct(self.seed,[up,p.sweep([-.04,0,0],padded=False)],[padded_up,p.sweep([-.04,0,0])],allowed)
        restored=second['remaining']-first['remaining']
        self.assertGreater(material_volume(restored),1e-12)
        self.assertLess(material_volume((restored-second['protected_contact_core'])^padded_up),1e-14)
        self.assertLess(material_volume(second['remaining']^up),1e-14)

    def test_collapsed_expansion_prism_uses_equivalent_padding_order(self):
        from unittest.mock import patch
        original=S.swept_solid;p=self.policy
        def collapsed(mesh,*args,**kwargs):
            if mesh is p.expanded:raise RuntimeError('Positive leading prism collapsed numerically')
            return original(mesh,*args,**kwargs)
        with patch.object(S,'swept_solid',side_effect=collapsed):
            padded=p.sweep([0,0,.04])
        nominal=p.sweep([0,0,.04],padded=False)
        self.assertLess(material_volume(nominal-padded),1e-14)
        self.assertGreater(material_volume(padded),material_volume(nominal))

    def test_demo_skips_contact_acceptance_without_changing_support(self):
        p=self.policy;nom=p.sweep([0,0,.04],padded=False);pad=p.sweep([0,0,.04]);allowed=np.arange(len(self.obj.faces))
        calls=[]
        def boundary(*args):
            calls.append(1)
            return contact_boundary(*args)
        formal=p.construct(self.seed,[nom],[pad],allowed,boundary=boundary)
        self.assertEqual(len(calls),2)
        calls.clear()
        demo=p.construct(self.seed,[nom],[pad],allowed,boundary=boundary,check_contacts=False)
        self.assertEqual(len(calls),1)
        self.assertFalse(demo['diagnostics']['contact_check_performed'])
        self.assertIsNone(demo['diagnostics']['contact_area_preserved'])
        self.assertLess(material_volume(formal['remaining']-demo['remaining']),1e-14)
        self.assertLess(material_volume(demo['remaining']-formal['remaining']),1e-14)

if __name__=='__main__':unittest.main()
