from pathlib import Path
import sys,unittest
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from material_builder_bounded import Builder,backend


class BoundedBuilderTests(unittest.TestCase):
    def test_bounded_plan_retains_nominal_and_stays_in_original_witness(self):
        fixture=trimesh.creation.box([.1]*3);fixture.apply_translation([0.,0.,.05])
        allowed=trimesh.creation.box([2.]*3);allowed.apply_translation([0.,0.,1.])
        rows=[dict(pose='A',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),
            hull=np.array([[0.,0.,0.]]),object_vertices=fixture.vertices)]
        mesh,report=Builder(fixture,allowed,rows).run()
        self.assertTrue(report['passed']);self.assertTrue(report['watertight'])
        self.assertLessEqual(backend.volume(backend.solid(fixture)-backend.solid(mesh)),1e-10)
        self.assertLessEqual(backend.volume(backend.solid(mesh)-backend.solid(allowed)),1e-10)
        self.assertTrue(any(r['passed'] and r['world_box_margin_m'] is not None for r in report['material_plan_trials']))


if __name__=='__main__':unittest.main()
