"""Coordinate/sign regressions for independently placed contact groups."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from step4_connect_support import compact_layout as C, build_coupled_saddle as S


class CompactLayoutTests(unittest.TestCase):
    def test_rigid_map_matches_homogeneous_transform_and_inverse(self):
        mesh=trimesh.creation.box(extents=[.012,.024,.039])
        mesh.apply_translation([.009,-.017,.041])
        rotation=Rotation.from_euler('xyz',[34,109,71],degrees=True).as_matrix()
        offset=np.array([.028,-.037,.095])
        expected=mesh.copy()
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=offset
        expected.apply_transform(transform)
        actual=C.transform_solid(S.solid(mesh),rotation.T,offset)
        difference=abs(float((actual-S.solid(expected)).volume()))*S.SCALE**3
        self.assertLess(difference,1e-15)
        restored=C.transform_solid(actual,rotation,-rotation.T@offset)
        self.assertLess(abs(float((restored-S.solid(mesh)).volume()))*S.SCALE**3,1e-15)

    def test_compactness_checks_horizontal_extent_in_every_pose(self):
        vertices=trimesh.creation.box(extents=[.02,.03,.20]).vertices
        rotation=Rotation.from_euler('y',90,degrees=True).as_matrix()
        metrics=C.span_metrics(vertices,[np.eye(3),rotation],[np.zeros(3),np.ones(3)])
        # A narrow tower in the first task is a wide fixture in the second.
        self.assertAlmostEqual(metrics['maximum_horizontal_span_m'],.20)
        self.assertAlmostEqual(metrics['horizontal_spans_m'][0][0],.02)

    def test_idle_group_is_checked_in_both_directions(self):
        # A normalized box corridor around y=0. Only the second task is
        # obstructed initially; checking just the first would falsely pass.
        corridor=S.solid(trimesh.creation.box(extents=[.2,.02,.02]))
        head=trimesh.creation.box(extents=[.01,.01,.01])
        far=head.copy();far.apply_translation([0,.1,0])
        case=SimpleNamespace(root_solids=[S.solid(head),S.solid(far)],
            poses=['first','second'],menus=[[0],[0]],padded_sweeps=[{0:corridor},{0:corridor}])
        layout=dict(bases=[np.eye(3),np.eye(3)],offsets=[np.zeros(3),np.zeros(3)])
        legal,checks=C.screen(case,layout)
        self.assertIsNone(legal)
        self.assertLess(checks[0]['foreign_root_sweep_overlap_m3'],C.TOL)
        self.assertGreater(checks[1]['foreign_root_sweep_overlap_m3'],C.TOL)
        layout['offsets'][1]=np.array([0,.3,0])
        legal,_=C.screen(case,layout)
        self.assertEqual(legal,[[0],[0]])


if __name__=='__main__':unittest.main()
