"""Area allocation is invariant to layer spacing and triangle density."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import surface as S


def layers(areas,gap,subdivisions=None):
    polygons={};face_layer={};face=0
    for layer,area in enumerate(areas):
        grid=1 if subdivisions is None else subdivisions[layer]
        side=np.sqrt(area)
        for j in range(grid):
            for i in range(grid):
                x0,x1=side*i/grid,side*(i+1)/grid;y0,y1=side*j/grid,side*(j+1)/grid
                vertices=np.array([[x0,y0,1+layer*gap],[x1,y0,1+layer*gap],
                                   [x1,y1,1+layer*gap],[x0,y1,1+layer*gap]])
                for index in ([0,1,2],[0,2,3]):
                    polygons[face]=vertices[index];face_layer[face]=layer;face+=1
    return polygons,face_layer


class SurfaceSamplingTests(unittest.TestCase):
    def test_equal_area_layers_do_not_merge_when_close(self):
        for number in (3,5):
            reference=None
            for gap in (1e-6,.01,.02,1.):
                polygons,face_layer=layers([1.]*number,gap)
                points,faces,report=S.surface_centers(polygons,with_report=True)
                labels=np.array([face_layer[f] for f in faces])
                counts=np.bincount(labels,minlength=number)
                self.assertEqual(len(points),200)
                self.assertLessEqual(np.abs(counts-200/number).max(),1.)
                if number==5:np.testing.assert_array_equal(counts,[40]*5)
                if reference is None:reference=(counts,points[:,:2])
                else:
                    np.testing.assert_array_equal(counts,reference[0])
                    np.testing.assert_allclose(points[:,:2],reference[1],atol=1e-10)
                self.assertLess(max(c['max_cell_relative_area_error'] for c in report['charts']),1e-8)

    def test_unequal_area_not_triangle_count_controls_allocation(self):
        for subdivision in ((1,1,1),(1,4,8),(8,4,1)):
            polygons,face_layer=layers([1.,2.,3.],.005,subdivision)
            _,faces,report=S.surface_centers(polygons,with_report=True)
            counts=np.bincount([face_layer[f] for f in faces],minlength=3)
            np.testing.assert_array_less(np.abs(counts-np.array([1.,2.,3.])*200/6),1.)
            self.assertEqual(report['chart_count'],3)
            self.assertAlmostEqual(report['eligible_area_m2'],6.,places=11)

    def test_each_layer_spreads_points_across_its_surface(self):
        polygons,face_layer=layers([1.]*5,.01)
        points,faces=S.surface_centers(polygons)
        labels=np.array([face_layer[f] for f in faces])
        for layer in range(5):
            p=points[labels==layer]
            quadrants=(p[:,0]>.5).astype(int)+2*(p[:,1]>.5)
            np.testing.assert_array_equal(np.bincount(quadrants,minlength=4),[10]*4)
            np.testing.assert_allclose(p[:,2],1+layer*.01,atol=1e-14)
            self.assertTrue(((p[:,:2]>0)&(p[:,:2]<1)).all())


if __name__=='__main__':unittest.main()
