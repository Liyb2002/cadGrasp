"""Visibility counterexamples and persistence of a support-independent domain."""
from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import work_volume as W,visibility as V


def domain_with_blocker():
    base=trimesh.creation.box([.1, .1, .1]);base.apply_translation([0, 0, -.05])
    roof=trimesh.convex.convex_hull([[-3,-3,.5],[3,-3,.5],[0,3,.5],[0,0,.8]])
    mesh=trimesh.util.concatenate([base,roof])
    ids=np.flatnonzero(base.face_normals[:,2]>.9)
    return SimpleNamespace(mesh=mesh,work_ids=ids,normals=-base.face_normals[ids],ray_offset=1e-5,
                           data={'load':{'cone_half_deg':30.}})


class VisibilityTests(unittest.TestCase):
    def test_occluded_half_ray_removed_even_beyond_the_occluding_object(self):
        domain=domain_with_blocker()
        raw=W.WorkVolume(domain.mesh.triangles[domain.work_ids],-domain.normals,domain.work_ids,30.,6.)
        solid=trimesh.creation.box([.02, .02, .02]);solid.apply_translation([0, 0, 2.])
        self.assertFalse(raw.check_parts([solid])['passed'])
        work=W.WorkVolume.from_domain(domain)
        self.assertEqual(len(work.triangles),0)
        self.assertTrue(work.check_parts([solid])['passed'])
        self.assertGreater(work.visibility['occluded_parameter_measure'],0)

    def test_occluded_single_witness_is_not_mistaken_for_a_clear_piece(self):
        work=W.WorkVolume([[[0,0,0],[1,0,0],[0,1,0]]],[[0,0,1]],[1],30.,1.)
        solid=trimesh.creation.box([.01]*3);solid.apply_translation([.2, .2, 1.])
        witness=dict(classification='outer_envelope_intersection',reachable_ray_witness_found=False)
        with patch.object(work,'_intersection',return_value=witness):
            result=work.check_parts([solid])
        self.assertFalse(result['passed'])
        self.assertEqual(result['classification'],'visibility_or_envelope_unresolved')

    def test_shadow_bounds_contain_all_intermediate_directions(self):
        blocker=np.array([[-2,-2,1],[2,-2,1],[0,2,1.]])
        slopes=np.array([[-.2,-.2],[.2,-.2],[0,.2]])
        source=V.Polygon([[-.05,-.05],[.05,-.05],[0,.05]])
        possible,certain=V.shadows([(blocker,np.array([0,0,1.]))],slopes,source)
        self.assertTrue(certain.covers(source))
        rng=np.random.default_rng(913)
        for weights in rng.dirichlet(np.ones(3),100):
            slope=weights@slopes
            shadow=V.Polygon(blocker[:,:2]-blocker[:,2:]*slope)
            self.assertTrue(shadow.buffer(1e-12).covers(certain))
            self.assertTrue(possible.buffer(1e-12).covers(shadow))

    def test_line_sources_are_retained_because_their_cone_sweep_has_volume(self):
        parts=V.triangle_parts(V.shapely.LineString([(0,0),(1,0)]))
        self.assertEqual(len(parts),1)
        np.testing.assert_array_equal(parts[0],[[0,0],[1,0],[1,0]])

    def test_backward_ray_tolerance_is_uncertain_not_declared_visible(self):
        source=V.Polygon([[-.01,-.01],[.01,-.01],[0,.01]])
        roof=np.array([[[-1,-1,-.5e-6],[1,-1,-.5e-6],[0,1,-.5e-6]]])
        occluders=V.front_occluders(roof,np.zeros(3),np.eye(3),1.,0.,source,.6)
        possible,certain=V.shadows(occluders,np.array([[-.2,-.2],[.2,-.2],[0,.2]]),source)
        self.assertTrue(possible.covers(source))
        self.assertTrue(certain.is_empty)

    def test_reading_saved_domain_does_not_rebuild_visibility(self):
        work=W.WorkVolume.from_domain(domain_with_blocker())
        with tempfile.TemporaryDirectory(prefix='.work_volume_test_',dir=W.I.ROOT/'slides') as directory:
            root=Path(directory);source=root/'needs.json';source.write_text('{}')
            path=work.export(root,source)
            with patch.object(V,'build_cells',side_effect=AssertionError('Visibility may only be built in Step 2')):
                saved=W.WorkVolume.read(path)
            for key,value in work.arrays().items():np.testing.assert_array_equal(saved.arrays()[key],value)


if __name__=='__main__':unittest.main()
