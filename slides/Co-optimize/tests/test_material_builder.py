from pathlib import Path
from unittest.mock import patch
import sys,unittest
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from material_builder import Builder,backend


class BuilderTests(unittest.TestCase):
    def test_recut_removes_actual_illegal_material_and_preserves_legal_nominal(self):
        fixture=trimesh.creation.box([.5,.5,.5]);fixture.apply_translation([0.,0.,.25])
        allowed=trimesh.creation.box([2.,2.,2.]);allowed.apply_translation([0.,0.,1.])
        rows=[dict(pose='A',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),hull=np.array([[0.,0.,0.]]))]
        builder=Builder(fixture,allowed,rows)
        overhang=backend.box(np.array([.2,-.05,0.]),np.array([1.05,.05,.1]))
        builder.body=builder.body+overhang
        original=dict(passed=False,outside_saved_legal_component_cm3=500.)
        with patch.object(backend.Builder,'run',return_value=(backend.unpack(builder.body),original)):
            mesh,report=builder.run()
        self.assertTrue(report['passed']);self.assertTrue(report['connected']);self.assertTrue(report['watertight'])
        self.assertGreater(report['legality_recut_rounds'],0)
        self.assertLessEqual(report['outside_saved_legal_component_cm3'],1e-4)
        self.assertLessEqual(report['retained_legal_step4_material_loss_cm3'],1e-4)
        self.assertLessEqual(mesh.bounds[1,0],1.+1e-12)


if __name__=='__main__':unittest.main()
