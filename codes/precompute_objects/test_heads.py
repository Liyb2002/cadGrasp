"""Cache reuse, exact wrench equality and direction subset regression checks."""
import sys,unittest,copy,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from codes.precompute_objects import head_cache as C
from codes.precompute_objects.head_geometry import PairGeometry,FreePaths,transform_contact,horizontal_catalogue
from step3_scheculer.pair_tasks import read_task
from step2_local_support import circles as P,withdrawal as W

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=ROOT/'objects/B/poses/pose_1'
        cls.cache=C.load(cls.folder)
        if cls.cache is None:raise RuntimeError('Build B/pose_1 before these integration tests')
        cls.task=read_task('B','pose_1')
    def test_no_geometry_or_exit_recalculation(self):
        with patch.object(FreePaths,'__init__',side_effect=AssertionError('roadmap rebuilt')),patch.object(P.SurfaceCircles,'fit_area',side_effect=AssertionError('patch rebuilt')),patch.object(W.Analyzer,'analyze',side_effect=AssertionError('exit recomputed')):
            g=PairGeometry([self.task])
        self.assertEqual(len(g.initial),200)
    def test_exact_generator_supply(self):
        rows=self.cache['pools'][.01];contacts=[e['contact'] for e in rows if e['valid']][:4]
        raw=[{k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')} for c in contacts]
        np.testing.assert_array_equal(self.task.supply(contacts),self.task.supply(raw))
    def test_directions_subset_only(self):
        cat=horizontal_catalogue(self.task.domain.mesh);rows=C.pool(self.cache,.01,cat)
        old=np.array(self.cache['catalogue']['vectors']);new=np.array(cat['vectors'])
        for original,row in zip(self.cache['pools'][.01],rows):
            if 'directions' not in row:continue
            for i in row['directions'][0]:
                j=np.linalg.norm(old-new[i],axis=1).argmin()
                self.assertIn(int(j),original['directions'][0])
        self.assertTrue(all(e['valid']==bool(e.get('directions',[[]])[0]) for e in rows))
    def test_transformation_drops_native_wrenches(self):
        c=self.cache['pools'][.01][0]['contact'];T=np.eye(4);T[:3,3]=[1,2,3]
        moved=transform_contact(c,T)
        self.assertNotIn('wrench_generators',moved)
        np.testing.assert_allclose(moved['triangles_m'],c['triangles_m']+T[:3,3])
    def test_changed_inputs_rejected(self):
        with patch.object(C,'inputs',return_value={}):
            with self.assertRaises(ValueError):C.load(self.folder)
if __name__=='__main__':unittest.main()
