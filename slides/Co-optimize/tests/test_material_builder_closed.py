from pathlib import Path
from unittest.mock import patch
import sys,unittest
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from material_builder_closed import Builder,backend,regularize_external_boundaries


class ClosedBuilderTests(unittest.TestCase):
    def test_internal_negative_boundary_remains_a_real_cavity(self):
        outer=trimesh.creation.box([.1]*3);outer.apply_translation([0.,0.,.05])
        cavity=trimesh.creation.icosphere(subdivisions=0,radius=.01)
        cavity.apply_translation([0.,0.,.05])
        body=backend.solid(outer)-backend.solid(cavity)
        result,removed=regularize_external_boundaries(body)
        self.assertFalse(removed)
        self.assertAlmostEqual(result.volume(),body.volume())
        self.assertEqual(len(result.decompose()),len(body.decompose()))
        self.assertEqual(len(backend.unpack(result).split(only_watertight=False)),2)

    def test_a_passed_recut_still_removes_an_external_negative_boundary(self):
        fixture=trimesh.creation.box([.1]*3);fixture.apply_translation([0.,0.,.05])
        allowed=trimesh.creation.box([2.]*3);allowed.apply_translation([0.,0.,1.])
        rows=[dict(pose='A',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),hull=np.array([[0.,0.,0.]]))]
        builder=Builder(fixture,allowed,rows)
        negative=trimesh.creation.icosphere(subdivisions=0,radius=.01)
        negative.apply_translation([.5,.5,.1]);negative.invert()
        builder.body=backend.md.Manifold.compose([backend.solid(fixture),backend.solid(negative)])
        uncanonical=backend.unpack(builder.body)
        self.assertEqual(len(uncanonical.split(only_watertight=False)),2)
        self.assertEqual(len(backend.components(builder.body)),1)
        with patch('material_builder_closed.ClippedBuilder.run',return_value=(uncanonical,dict(passed=True))):
            mesh,report=builder.run()
        self.assertTrue(report['passed']);self.assertTrue(mesh.is_watertight)
        self.assertEqual(len(mesh.split(only_watertight=False)),1)
        self.assertAlmostEqual(mesh.volume,fixture.volume)
        self.assertEqual(len(report['final_boundary_regularizations']),1)
        self.assertEqual(report['legality_reconnect_cycles'],0)

    def test_legal_pieces_are_reconnected_after_a_recut(self):
        left=trimesh.creation.box([.1]*3);left.apply_translation([-.3,0.,.05])
        right=left.copy();right.apply_translation([.6,0.,0.])
        fixture=trimesh.util.concatenate([left,right]);allowed=trimesh.creation.box([2.]*3)
        allowed.apply_translation([0.,0.,1.])
        rows=[dict(pose='A',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),hull=np.array([[-.3,0.,0.]]))]
        builder=Builder(fixture,allowed,rows)
        old=dict(passed=False,connected=False,outside_saved_legal_component_cm3=0.)
        with patch.object(backend.Builder,'run',return_value=(fixture,old)):
            mesh,report=builder.run()
        self.assertTrue(report['passed']);self.assertTrue(report['watertight'])
        self.assertEqual(report['legality_reconnect_cycles'],1)
        self.assertEqual(len(backend.components(builder.body)),1)
        self.assertLessEqual(report['outside_saved_legal_component_cm3'],1e-4)
        self.assertLessEqual(report['retained_legal_step4_material_loss_cm3'],1e-4)


if __name__=='__main__':unittest.main()
