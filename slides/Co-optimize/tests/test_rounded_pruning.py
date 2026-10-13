"""Rounded retention preserves contacts, broadens cut links and locks BBX."""
from pathlib import Path
import sys,unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/p)for p in ['helper_func','step5.2','step5.3','tests']]
import _bootstrap
from deterministic_base import box,union,volume,unpack,solid
from rounded_pruning import RoundedPruner
from replay_legs import reconstruct
from test_support_pruning import Certificate


class RoundedPruningTests(unittest.TestCase):
    def test_round_retention_widens_an_artificial_thin_link_without_growing_bbx(self):
        original=box(np.zeros(3),np.array([.06,.04,.012]))
        left=box(np.zeros(3),np.array([.015,.04,.012]))
        right=box(np.array([.045,0.,0.]),np.array([.06,.04,.012]))
        neck=box(np.array([.015,.019,.003]),np.array([.045,.021,.009]))
        anchor=union([left,right,neck]);row=dict(fixture_transform=np.eye(4),object_transform=np.eye(4))
        pruner=RoundedPruner(unpack(original),unpack(original),Certificate([left,right]),reference=unpack(anchor),
            rows=[row],object_vertices=np.array([[0.,0.,0.],[.06,.04,.012]]),require_connected=True)
        mesh,report,_=pruner.run();result=solid(mesh)
        wider=box(np.array([.02,.017,.003]),np.array([.04,.023,.009]))
        self.assertTrue(report['passed']);self.assertTrue(report['locked_bbx_unchanged'])
        self.assertLessEqual(volume(anchor-result),1e-10)
        self.assertLessEqual(volume(result-original),1e-10)
        self.assertGreater(volume(result^wider),volume(anchor^wider)*2)
        self.assertFalse(report['grid_cell_deletion_used'])

    def test_bbx_extension_is_cut_as_whole_planes_before_aesthetic_retention(self):
        original=box(np.zeros(3),np.array([.08,.04,.012]));anchor=box(np.zeros(3),np.array([.06,.04,.012]))
        row=dict(fixture_transform=np.eye(4),object_transform=np.eye(4))
        mesh,report,_=RoundedPruner(unpack(original),unpack(original),Certificate([anchor]),reference=unpack(anchor),
            rows=[row],object_vertices=np.array([[0.,0.,0.],[.06,.04,.012]]),require_connected=True).run()
        self.assertAlmostEqual(mesh.bounds[1,0],.06)
        self.assertLess(report['retained_bbx_l'],report['input_bbx_l'])
        self.assertEqual(report['history'][0]['operation'],'bbx_locked_whole_plane_trim')

    def test_fixed_legs_preserve_floor_and_bbx_on_a_broader_connected_core(self):
        anchor=box(np.array([0.,0.,.01]),np.array([.04,.01,.02]))
        current=box(np.array([0.,0.,.01]),np.array([.04,.02,.02]))
        leg=box(np.array([0.,0.,0.]),np.array([.01,.01,.02]));previous=anchor+leg
        body=unpack(box(np.zeros(3),np.array([.04,.03,.01])))
        witness=unpack(box(np.array([-.01,-.01,0.]),np.array([.05,.04,.03])))
        row=dict(pose='pose_1',fixture_transform=np.eye(4),object_transform=np.eye(4),
            object_ground=np.array([[0.,0.],[.04,0.],[0.,.03]]),hull=np.array([[0.,0.,0.]]))
        mesh,record=reconstruct(unpack(current),unpack(anchor),unpack(previous),witness,[row],body,
            dict(history=[],passed=True,watertight=True))
        self.assertTrue(record['locked_final_bbx_unchanged'])
        self.assertLessEqual(volume(previous-solid(mesh)),1e-10)
        self.assertLessEqual(volume(current-solid(mesh)),1e-10)


if __name__=='__main__':unittest.main()
