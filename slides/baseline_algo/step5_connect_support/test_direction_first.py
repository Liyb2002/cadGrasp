"""Direction constraints, full sweeps and a loose frame must agree geometrically."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import direction_first as X, belt_geometry as B, rigid_path as P, surface_check as U
from step5_connect_support.test_whole_assembly import contacts
from step2_local_support.surface import areas


class DirectionFirstTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box([1., 1., 1.]);self.mesh.apply_translation([0., 0., 1.])
        self.work=np.flatnonzero(self.mesh.face_normals[:,0]>.9)
        self.domain=SimpleNamespace(mesh=self.mesh,work_ids=self.work,com=self.mesh.center_mass)
        cloud=np.array([[-.7,-.7],[.7,-.7],[.7,.7],[-.7,.7]])
        self.floor=dict(floor_demands_xy_m=cloud,continuous_floor_enclosure_xy_m=cloud,original_pivot_m=np.zeros(3))

    def test_step5_uses_only_scheduled_survivors_and_reports_untried_ids(self):
        from step2_local_support import withdrawal as W
        catalogue=dict(vectors=[[-1.,0,0],[0,1.,0],[1.,0,0]],
            preferred_withdrawal_direction=[-1.,0,0],global_allowed_directions=W.normalize([0,1]))
        record=dict(mode=W.MODE,direction_catalogue=catalogue,
            contacts=[dict(certified_directions=W.normalize([0,1,2]))],common_directions=W.normalize([0,1]))
        directions,report=X.scheduled_directions(record,1)
        np.testing.assert_array_equal(directions,[[-1.,0,0]])
        self.assertEqual(report['chosen_direction_ids'],[0]);self.assertEqual(report['untried_count'],1)
        with self.assertRaises(AssertionError): X.scheduled_directions(dict(record,common_directions=W.normalize([2])))

    def test_scheduled_full_sweeps_survive_through_frame_construction(self):
        from step2_local_support import withdrawal as W
        original=contacts(self.mesh)
        catalogue=W.make_catalogue(self.mesh,self.work,original)
        analyzer=W.Analyzer(self.mesh,.01,catalogue)
        records=[analyzer.analyze(c) for c in original]
        record=dict(mode=W.MODE,direction_catalogue=catalogue,contacts=records,
                    common_directions=W.common(records,catalogue['global_allowed_directions']))
        report,module=X.search(self.domain,original,self.floor,.01,4,direction_record=record)
        self.assertTrue(report['passed'],report)
        self.assertEqual(len(report['belt']['terminals_m']),len(original))
        self.assertTrue(report['belt']['terminals_grouped_by_contact'])
        self.assertTrue(P.replay(B.Scene(self.mesh),module['parts'],module['trajectory']))
        self.assertEqual(report['direction_search']['source'],'Step3 actual optimized common head directions')

    def test_work_back_is_a_preference_and_floor_is_a_constraint(self):
        np.testing.assert_allclose(X.preferred_direction(self.mesh,self.work),[-1,0,0])
        top=np.flatnonzero(self.mesh.face_normals[:,2]>.9)
        directions,report=X.propose_directions(self.mesh,contacts(self.mesh),top)
        np.testing.assert_allclose(report['preferred_withdrawal_direction'],[0,0,-1])
        self.assertTrue(directions)
        self.assertTrue(all(d[2]>=0 for d in directions))
        self.assertFalse(report['global_impossibility_claimed'])

    def test_incompatible_fixed_heads_are_rejected_before_frame_construction(self):
        cells=[]
        for i,triangle in enumerate(self.mesh.triangles):
            center=triangle.mean(axis=0);patch=(center+.3*(triangle-center))[None]
            cells.append(dict(candidate_id=str(i),center_face=i,center_m=center,source_faces=np.array([i]),
                triangles_m=patch,triangle_areas_m2=areas(patch)))
        directions,report=X.propose_directions(self.mesh,cells,[])
        self.assertEqual(directions,[])
        self.assertFalse(report['rotation_searched'])

    def test_clearance_rejects_a_noncolliding_but_too_close_member(self):
        part=trimesh.creation.box([.1, .1, .1]);part.apply_translation([-.551, 0, 1])
        scene=X.SweptScene(self.mesh,np.array([-1.,0,0]),.02)
        self.assertTrue(scene.clear(part,clearance=0))
        self.assertFalse(scene.clear(part))

    def test_full_ray_rejects_a_remote_obstacle_after_initial_separation(self):
        obstacle=trimesh.creation.box([.2, .2, .2]);obstacle.apply_translation([2., 0, .5])
        part=trimesh.creation.box([.1, .1, .1]);part.apply_translation([0., 0, .5])
        scene=X.SweptScene(obstacle,np.array([1.,0,0]))
        self.assertTrue(B.Scene.clear(scene,part))
        self.assertFalse(scene.clear(part))

    def test_loose_frame_preserves_contacts_and_has_one_whole_path(self):
        original=contacts(self.mesh)
        report,module=X.search(self.domain,original,self.floor,.01,4)
        self.assertTrue(report['passed'],report)
        self.assertTrue(module['solid']['one_solid'])
        self.assertTrue(module['ground']['passed'])
        self.assertFalse(report['contact_interfaces_changed'])
        self.assertTrue(P.replay(B.Scene(self.mesh),module['parts'],module['trajectory']))
        for contact in original:
            self.assertLess(U.surface_distances(module['joined'],contact['triangles_m'].reshape(-1,3)).max(),1e-9)
        scene=X.SweptScene(self.mesh,np.asarray(report['withdrawal_direction']),report['actual_frame_gap_m'])
        for part,label in zip(module['parts'],module['labels']):
            if label.startswith(('frame_','connector_','ground_strip_')):
                self.assertTrue(scene.clear(part,ground=label.startswith('ground_strip_')))
        wrong=module['trajectory'].copy();wrong['motion']=(-np.asarray(wrong['motion'])).tolist()
        with self.assertRaises(AssertionError):P.replay(B.Scene(self.mesh),module['parts'],wrong)


if __name__=='__main__':unittest.main()
