"""Tiny-triangle numerical failures must yield a finite on-surface center."""
import unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path('/home/yli581/Desktop/cadGrasp');sys.path.insert(0,'/home/yli581/Desktop/cadGrasp/slides/Co-optimize/data/results/B/data/system_pso_comparison_all52_20261013_01/frozen_sources');sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
from step2_local_support import surface as S
class FiniteCentersTests(unittest.TestCase):
    def test_closest_point_nonfinite_falls_back_on_same_surface(self):
        tri=np.array([[[0.,0.,.01],[.001,0.,.01],[0.,.002,.01]]])
        with patch.object(S,'closest_point',return_value=np.full((1,3),np.nan)):
            centers,faces,areas=S.equal_area_centers(tri,np.array([7]),1)
        np.testing.assert_allclose(centers,[tri[0].mean(axis=0)],atol=1e-18)
        self.assertEqual(faces,[7]);self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_allclose(areas,S.areas(tri))
if __name__=='__main__':unittest.main()
