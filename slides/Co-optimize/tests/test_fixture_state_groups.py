"""State capacity is independent of a Juxtapose operation's arity."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from publish_fast_gradient import fixture_state_groups


class FixtureStateTests(unittest.TestCase):
    def test_one_state_can_fit_three_poses(self):
        native=np.repeat(np.eye(4)[None],4,axis=0);native[3,0,3]=1.
        states=fixture_state_groups(['p1','p2','p3','p4'],[0,0,0,3],range(4),native)
        self.assertEqual([s['poses'] for s in states],[['p1','p2','p3'],['p4']])

    def test_equivalent_world_transform_is_one_state_despite_distinct_host_labels(self):
        native=np.repeat(np.eye(4)[None],3,axis=0)
        states=fixture_state_groups(['p1','p2','p3'],[0,1,2],range(3),native)
        self.assertEqual(len(states),1)
        self.assertEqual(len(states[0]['poses']),3)


if __name__=='__main__':unittest.main()
