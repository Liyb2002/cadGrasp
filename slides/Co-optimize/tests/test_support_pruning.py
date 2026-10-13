"""Deletion preserves required contact ports, connected cores and cavities."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/'step5.2'),str(COOPT/'step5.3'),str(COOPT/'helper_func')]
from prune_support import Pruner,material_components
from deterministic_base import box,unpack,volume,union,solid

class Certificate:
    def __init__(self,ports):
        self.ports=ports;self.calls=self.classifications=self.reused=0
    def evaluate(self,mesh):
        self.calls+=1;self.classifications+=1
        passed=all(volume(solid(mesh)^port)>1e-13 for port in self.ports)
        mask=np.full(32768,passed)
        return dict(passed=passed,counts={'0':int(mask.sum())},masks={0:mask},
                    supplies={0:np.zeros((1,7))},contact_triangles={0:len(self.ports)})

class PruningTests(unittest.TestCase):
    def test_useless_island_deleted_and_needed_island_retained(self):
        main=box(np.array([0.,0.,0.]),np.array([.02,.02,.02]))
        needed=box(np.array([.04,0.,0.]),np.array([.05,.01,.01]))
        useless=box(np.array([.08,0.,0.]),np.array([.085,.005,.005]))
        original=union([main,needed,useless])
        allowed=box(np.array([-.01,-.01,-.01]),np.array([.1,.04,.04]))
        _,report,_=Pruner(unpack(original),unpack(allowed),Certificate([main,needed]),grids=()).run()
        self.assertTrue(report['passed']);self.assertEqual(report['component_count'],2)
        self.assertAlmostEqual(report['removed_material_cm3'],volume(useless)*1e6)
    def test_connected_core_is_not_cut_into_required_islands(self):
        original=box(np.array([0.,0.,0.]),np.array([.06,.02,.02]))
        ports=[box(np.array([0.,0.,0.]),np.array([.01,.02,.02])),
               box(np.array([.05,0.,0.]),np.array([.06,.02,.02]))]
        pruner=Pruner(unpack(original),unpack(original),Certificate(ports),grids=())
        pruner.run()
        candidate=original-box(np.array([.025,-.01,-.01]),np.array([.035,.03,.03]))
        self.assertFalse(pruner.accept(candidate,'split_required_core'))
        self.assertEqual(len(material_components(pruner.body)),1)
    def test_component_extraction_does_not_fill_a_real_cavity(self):
        outer=box(np.array([0.,0.,0.]),np.array([.05,.05,.05]))
        cavity=box(np.array([.01,.01,.01]),np.array([.04,.04,.04]))
        value=outer-cavity;assembled=union(material_components(value))
        self.assertLessEqual(volume(assembled-value),1e-13)
        self.assertAlmostEqual(volume(assembled),volume(value))
    def test_required_contact_cut_is_rejected(self):
        original=box(np.array([0.,0.,0.]),np.array([.04,.02,.02]))
        port=box(np.array([.03,0.,0.]),np.array([.04,.02,.02]))
        pruner=Pruner(unpack(original),unpack(original),Certificate([port]),grids=())
        pruner.run();self.assertFalse(pruner.accept(original-port,'remove_required_port'))
        self.assertAlmostEqual(volume(pruner.body),volume(original))
    def test_serialization_cannot_turn_a_deletion_into_added_material(self):
        original=box(np.zeros(3),np.array([.04,.02,.02]))
        port=box(np.zeros(3),np.array([.01,.02,.02]))
        pruner=Pruner(unpack(original),unpack(original),Certificate([port]),grids=())
        pruner.run()
        candidate=box(np.zeros(3),np.array([.03,.02,.02]))
        bad_roundtrip=unpack(box(np.zeros(3),np.array([.05,.02,.02])))
        with patch('prune_support.C.trimesh.load',return_value=bad_roundtrip):
            self.assertFalse(pruner.accept(candidate,'serialized_added_material'))
        self.assertAlmostEqual(volume(pruner.body),volume(original))
        self.assertIn('Serialized deletion',pruner.history[-1]['error'])

if __name__=='__main__':unittest.main()
