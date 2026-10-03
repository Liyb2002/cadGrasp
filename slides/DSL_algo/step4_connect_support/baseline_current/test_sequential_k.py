from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from step4_connect_support.baseline_current import head_registration as H
from step4_connect_support.baseline_current import build_local_bodies as L
from step4_connect_support.baseline_current.test_head_registration import fixture
from step4_connect_support.baseline_current.run_sequential_k import foot_menu


class MultiPoseRegistrationTests(unittest.TestCase):
    def four(self):
        groups,heads,bases,offsets,tasks=fixture()
        return groups*2,heads*2,np.concatenate([bases,bases]),np.concatenate([offsets,offsets]),tasks*2

    def test_four_pose_identity_deduplicates_physical_ids(self):
        groups,heads,bases,offsets,_=self.four()
        registered,check=H.register(groups,heads,bases,offsets,general_layout=True)
        self.assertEqual(len(registered),5)
        self.assertEqual(check['shared_head_count'],5)
        self.assertEqual(check['shared_instance_checks'],7)
        self.assertEqual(registered[0].active_poses,(0,1,2,3))

    def test_variable_four_heads_allowed_but_old_entry_remains_strict(self):
        groups,heads,bases,offsets,_=self.four()
        groups=[list(x) for x in groups]; heads=[list(x) for x in heads]
        extra=dict(groups[0][1],candidate_id='F',triangles_m=groups[0][1]['triangles_m']+[0,.3,0])
        groups[0].append(extra); heads[0].append([v+[0,.3,0] for v in heads[0][1]])
        registered,_=H.register(groups,heads,bases,offsets,general_layout=True)
        self.assertEqual(len(registered),6)
        with self.assertRaises(ValueError):H.register(groups,heads,bases,offsets)

    def test_split_shared_head_in_fourth_pose_cannot_bypass_constructor(self):
        groups,heads,bases,offsets,tasks=self.four()
        offsets[3,0]+=.02
        case=SimpleNamespace(groups=groups,heads=heads,tasks=tasks,source=None,schedule={},paths=[])
        placement=dict(bases=bases,offsets=offsets,directions=np.array([[1.,0,0]]*4))
        with TemporaryDirectory() as tmp:
            work=Path(tmp)/'work'
            with self.assertRaisesRegex(ValueError,'Shared head'):
                L.build(work,[],case=case,placement=placement,general_layout=True,verify=False)
            self.assertFalse(work.exists())

    def test_all_twelve_floor_pairs_checked_including_fourth(self):
        bases=np.array([np.eye(3)]*3+[[[0.,1,0],[0,0,1],[1,0,0]]])
        tasks=[SimpleNamespace(pose=f'pose_{k}',floor=np.zeros(3)) for k in range(4)]
        demands=[np.array([[-.02,.1],[.03,.1]])]+[np.array([[.1,.1]])]*3
        check=H.floor_compatibility(tasks,demands,bases,np.zeros((4,3)))
        self.assertEqual(len(check['per_pose']),12)
        row=next(r for r in check['per_pose'] if r['pose']=='pose_0' and r['other_pose']=='pose_3')
        self.assertEqual(row['violating_sample_count'],1)
        self.assertFalse(check['passed'])

    def test_terminals_are_clipped_by_every_other_floor(self):
        bases=np.array([np.eye(3)]*3+[[[0.,1,0],[0,0,1],[1,0,0]]])
        offsets=np.zeros((4,3))
        case=SimpleNamespace(poses=range(4),demands=[np.array([[.01,.01],[.02,.01],[.01,.02]])]*4)
        options=list(foot_menu(case,bases,offsets))
        self.assertTrue(options)
        for feet in options:
            for k,pads in enumerate(feet):
                for pad in pads:
                    points=np.c_[pad,np.zeros(len(pad))]@bases[k]+offsets[k]
                    for j in range(4):
                        self.assertTrue(np.all((points-offsets[j])@bases[j][2]>=-1e-12))
