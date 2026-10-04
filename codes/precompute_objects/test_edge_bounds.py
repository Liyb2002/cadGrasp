import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/DSL_algo'))
from step2_local_support.circles import edge_interval,edge_interval_python
class EdgeBoundsTests(unittest.TestCase):
    def test_exact_interval_rule(self):
        rng=np.random.default_rng(312)
        theta=np.linspace(0,2*np.pi,20,endpoint=False);poly=np.c_[np.cos(theta),np.sin(theta),np.zeros(20)]
        for i in range(500):
            segment=rng.uniform(-2,2,(2,3));segment[:,2]=0
            np.testing.assert_allclose(edge_interval(poly,np.array([0.,0.,1.]),segment,1e-9),edge_interval_python(poly,np.array([0.,0.,1.]),segment,1e-9),atol=1e-13,rtol=1e-13)
        for s in [poly[:2],poly[[0,10]],np.zeros((2,3))]:
            np.testing.assert_allclose(edge_interval(poly,np.array([0.,0.,1.]),s,1e-9),edge_interval_python(poly,np.array([0.,0.,1.]),s,1e-9),atol=1e-13)
if __name__=='__main__':unittest.main()
