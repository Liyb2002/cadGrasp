
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from animate_exit_directions import direction_schedule, remaining_support, pending_red, pending_red_mesh
from co_common import *

class ExitAnimation(unittest.TestCase):
    def test_one_direction_changes_in_native_floor_hemisphere(self):
        transforms=[trimesh.transformations.rotation_matrix(.7,[1,0,0]),trimesh.transformations.rotation_matrix(-.5,[0,1,0])]
        previous=None
        for active,u,directions in direction_schedule(transforms,12):
            np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12)
            for T,d in zip(transforms,directions):self.assertGreaterEqual((T[:3,:3]@d)[2],0.)
            if previous is not None:
                changed=np.flatnonzero(np.linalg.norm(directions-previous,axis=1)>1e-10)
                self.assertEqual(changed.tolist(),[active])
            previous=directions

    def test_fixed_meridian_reaches_equator_without_turning_back(self):
        transforms=[np.eye(4),trimesh.transformations.rotation_matrix(.7,[1,0,0])]
        previous=np.zeros(2)
        last=None
        for active,u,directions in direction_schedule(transforms,16):
            for i,(T,d) in enumerate(zip(transforms,directions)):
                world=T[:3,:3]@d;tilt=np.arctan2(np.linalg.norm(world[:2]),world[2])
                self.assertGreaterEqual(tilt+1e-12,previous[i]);previous[i]=tilt
                if np.linalg.norm(world[:2])>1e-12:self.assertAlmostEqual(np.arctan2(world[1],world[0]),np.pi/4,places=12)
            last=directions
        for T,d in zip(transforms,last):self.assertAlmostEqual((T[:3,:3]@d)[2],0.,places=12)

    def test_other_sweep_still_blocks_material_restoration(self):
        seed=S.solid(trimesh.creation.box([.06,.06,.06]))
        oldmesh=trimesh.creation.box([.04,.02,.08]);oldmesh.apply_translation([.005,0,0]);old=S.solid(oldmesh)
        fixedmesh=trimesh.creation.box([.02,.02,.08]);fixedmesh.apply_translation([.01,0,0]);fixed=S.solid(fixedmesh)
        movedmesh=oldmesh.copy();movedmesh.apply_translation([0,.025,0]);moved=S.solid(movedmesh)
        before=remaining_support(seed,[old,fixed]);after=remaining_support(seed,[moved,fixed])
        regrown=after-before;lost=before-after
        self.assertGreater(material_volume(regrown),0.)
        self.assertGreater(material_volume(lost),0.)
        self.assertLess(material_volume(regrown^fixed),1e-14)
        self.assertLess(material_volume(after^fixed),1e-14)

    def test_red_lasts_half_second_at_output_rate_and_regrowth_wins(self):
        part=S.solid(trimesh.creation.box([.01,.01,.01]));empty=F.md.Manifold();queue=[(5,part)]
        self.assertGreater(material_volume(pending_red(queue,empty,14,10)),0.)
        self.assertEqual(material_volume(pending_red(queue,empty,15,10)),0.)
        self.assertEqual(material_volume(pending_red(queue,part,7,10)),0.)

    def test_display_fragments_expire_and_exclude_regrown_blue(self):
        part=S.solid(trimesh.creation.box([.01,.01,.01]));queue=[(5,part)]
        mesh=pending_red_mesh(queue,F.md.Manifold(),14,10)
        self.assertGreater(mesh.volume,0.)
        self.assertEqual(len(pending_red_mesh(queue,F.md.Manifold(),15,10).faces),0)
        self.assertEqual(len(pending_red_mesh(queue,part,7,10).faces),0)

if __name__=='__main__':unittest.main()
