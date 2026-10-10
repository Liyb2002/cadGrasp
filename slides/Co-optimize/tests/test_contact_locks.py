"""Locks compose across poses; released contacts stay blocked by other poses."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
import numpy as np
import trimesh
from contact_lock_chain import ContactLocks

class LockTests(unittest.TestCase):
    def test_release_requires_every_pose_to_unlock(self):
        model=ContactLocks.__new__(ContactLocks)
        calls=[]
        def locked(d):calls.append(tuple(d));return np.array([d[0]>0,d[1]>0])
        model.locked=locked
        directions=np.array([[1.,0.,0.],[1.,0.,0.]])
        locks,count,active=model.update(directions)
        previous=dict(directions=directions,locks=locks)
        changed=np.array([[-1.,0.,0.],[1.,0.,0.]])
        locks,count,active=model.update(changed,previous)
        self.assertEqual(len(calls),3)
        self.assertFalse(active[0]);self.assertEqual(count[0],1)
        locks,count,active=model.update(np.array([[-1.,0.,0.],[-1.,0.,0.]]),dict(directions=changed,locks=locks))
        self.assertTrue(active[0]);self.assertEqual(count[0],0)
    def test_normal_and_nonlocal_obstruction(self):
        cube=trimesh.creation.box()
        model=ContactLocks(cube,np.array([[.5,0.,0.]]),np.array([[1.,0.,0.]]),5.)
        self.assertTrue(model.locked(np.array([1.,0.,0.]))[0])
        self.assertFalse(model.locked(np.array([-1.,0.,0.]))[0])
        other=cube.copy();other.apply_translation([3.,0.,0.])
        pair=trimesh.util.concatenate([cube,other])
        model=ContactLocks(pair,np.array([[.5,0.,0.]]),np.array([[1.,0.,0.]]),5.)
        self.assertTrue(model.locked(np.array([-1.,0.,0.]))[0])
        model=ContactLocks(pair,np.array([[.5,0.,0.]]),np.array([[1.,0.,0.]]),1.)
        self.assertFalse(model.locked(np.array([-1.,0.,0.]))[0])

if __name__=='__main__':unittest.main()
