from types import SimpleNamespace
import unittest
import numpy as np
import trimesh

from step4_connect_support.run_independent import placement_for
from step4_connect_support.fixture_view import local_to_world


class IndependentPlacementTests(unittest.TestCase):
    def test_different_exit_directions_produce_separate_fixture_placements(self):
        mesh=trimesh.creation.box(extents=[.1,.08,.12]);mesh.apply_translation([0,0,.06])
        task=SimpleNamespace(domain=SimpleNamespace(mesh=mesh))
        case=SimpleNamespace(tasks=[task,task],heads=[[[mesh.vertices]],[[mesh.vertices]]],
            demands=[np.array([[-.04,-.03],[.04,.03]])]*2,
            catalogues=[np.array([[1.,0,0]]),np.array([[0.,1.,0]])])
        p=placement_for(case,[0,0])
        self.assertFalse(np.allclose(p['bases'][0],p['bases'][1]))
        self.assertFalse(np.allclose(p['offsets'][0],p['offsets'][1]))
        mapped=[]
        for k in range(2):
            b,o=p['bases'][k],p['offsets'][k]
            points=mesh.vertices@b+o
            mapped.append(points)
            np.testing.assert_allclose(local_to_world(points,b,o),mesh.vertices,atol=1e-15)
            np.testing.assert_allclose(p['directions'][k]@b,[1,0,0],atol=1e-15)
            np.testing.assert_allclose(points[:,2],mesh.vertices[:,2],atol=1e-15)
        self.assertGreater(mapped[1][:,1].min()-mapped[0][:,1].max(),.04)
        # Idle material from pose2 is not transported back onto pose1's object.
        idle=local_to_world(mapped[1],p['bases'][0],p['offsets'][0])
        self.assertGreater(idle[:,1].min(),mesh.bounds[1,1])


if __name__=='__main__':unittest.main()
