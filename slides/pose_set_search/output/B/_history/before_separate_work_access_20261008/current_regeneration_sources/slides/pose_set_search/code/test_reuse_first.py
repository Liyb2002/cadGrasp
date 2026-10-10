"""Regression checks for the user's volume objective and selective reuse."""
import tempfile
import unittest
from types import SimpleNamespace
from common import *
from model import Layout
from reuse_first import ReuseFirstSearch, registered, juxtaposed_count
from volume_guidance import VolumeGuidance
from exact_worker import write_result, read_result


def layout(n):
    return Layout(np.repeat(np.eye(4)[None], n, axis=0),
                  np.repeat([[0., 0., 1.]], n, axis=0), np.arange(n), tuple(range(n)))


class ReuseFirstTests(unittest.TestCase):
    def test_worker_export_preserves_the_accepted_boundary_and_volume(self):
        outer = C.trimesh.creation.box(extents=[.06, .05, .04])
        cut = C.trimesh.creation.box(extents=[.03, .02, .05])
        cut.apply_translation([.015, .013, .01])
        solid = C.S.solid(outer)-C.S.solid(cut)
        mesh = unpack_solid(solid)
        result = dict(layout=layout(1), remaining=solid, masks={0:np.ones(2, bool)},
                      supplies={0:np.eye(7)}, contacts={0:dict(triangles=mesh.triangles[:2], sources=np.array([0, 1]))},
                      counts={'0':2}, diagnostics={}, volume_cm3=C.material_volume(solid)*1e6,
                      maximum_projected_footprint_m2=.003, seconds=0., classifiers={0:{}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result'
            write_result(result, path)
            restored = read_result(path, 1)
            exported = result_mesh(restored)
            np.testing.assert_array_equal(mesh.vertices, exported.vertices)
            np.testing.assert_array_equal(mesh.faces, exported.faces)
            self.assertAlmostEqual(result['volume_cm3'], abs(exported.volume)*1e6, places=9)
            obj = Path(folder)/'support.obj'
            C.D.export_exact_obj(exported, obj)
            final = C.trimesh.load(obj, process=False)
            np.testing.assert_array_equal(final.vertices, mesh.vertices)
            np.testing.assert_array_equal(final.faces, mesh.faces)

    def test_material_volume_outweighs_support_placement_and_juxtapose_counts(self):
        native = layout(3)
        shared = native.copy()
        shared.hosts[:] = 0
        search = ReuseFirstSearch(None, None)
        a = dict(layout=native, masks={k:np.ones(2, bool) for k in native.active},
                 volume_cm3=200., maximum_projected_footprint_m2=.01)
        b = dict(a, layout=shared, volume_cm3=120., maximum_projected_footprint_m2=.04)
        self.assertLess(search.score(b), search.score(a))
        a['volume_cm3'] = 100.
        self.assertLess(search.score(a), search.score(b))

    def test_failed_loads_precede_volume(self):
        search = ReuseFirstSearch(None, None)
        feasible = dict(layout=layout(1), masks={0:np.array([True, True])},
                        volume_cm3=200., maximum_projected_footprint_m2=.01)
        missing = dict(feasible, masks={0:np.array([True, False])}, volume_cm3=1.)
        self.assertLess(search.score(feasible), search.score(missing))

    def test_registered_mode_requires_original_seating_relation(self):
        original = layout(2)
        self.assertTrue(all(registered(original, k) for k in original.active))
        moved = original.copy()
        moved.placements[1, 0, 3] = .003
        self.assertFalse(registered(moved, 1))
        self.assertEqual(juxtaposed_count(moved), 1)

    def test_registered_task_cannot_translate_inside_juxtapose_branch(self):
        current_layout = layout(2)
        current_layout.hosts[1] = 0
        current_layout.placements[1, 0, 3] = .01
        model = SimpleNamespace(poses=['pose_1', 'pose_2'], extent=.1,
            floor_normal=lambda layout,k:np.array([0., 0., 1.]),
            proxy=lambda layout,targets:dict(loss=1., sum_loss=1., span_m=0.))
        current = dict(layout=current_layout, masks={0:np.ones(2,bool), 1:np.zeros(2,bool)})
        proposals, _ = ReuseFirstSearch(model, None).local_proposals(
            current, [(1, 0)], {'pose_index':1}, translation_guests=(0, 1))
        translations = [candidate for kind,candidate,_ in proposals if kind.startswith('translation')]
        self.assertTrue(translations)
        for trial in translations:
            np.testing.assert_array_equal(trial.placements[0], np.eye(4))

    def test_new_pose_can_juxtapose_when_it_passes_but_damages_an_old_pose(self):
        model = SimpleNamespace(poses=['pose_1', 'pose_2', 'pose_3'])
        search = ReuseFirstSearch(model, None)
        current = dict(layout=layout(3), masks={0:np.array([True, False]),
                                              1:np.ones(2, bool), 2:np.ones(2, bool)},
                       counts={'0':1, '1':2, '2':2})
        proposed_guests = []
        search.targets = lambda result:([], None, [])
        search.juxtapose_proposals = lambda result,guests,targets:proposed_guests.extend(guests) or []
        search.shortlist = lambda proposals,targets:([], 0)
        search.record = lambda row:None
        search.rescue(current, [1, 2], uncommitted_guests=(2,))
        self.assertEqual(proposed_guests, [2])

    def test_nominal_volume_guidance_counts_overlapping_registered_wrap_once(self):
        body = C.trimesh.creation.box(extents=[.04, .04, .04])
        outer = C.trimesh.creation.box(extents=[.05, .05, .05])
        wrap = C.S.solid(outer)-C.S.solid(body)
        work = C.trimesh.creation.box(extents=[.005]*3)
        work.apply_translation([.1, .1, .1])
        model = SimpleNamespace(mesh=body, thickness=.005, length=.2,
                                ray=RayMeshIntersector(body),
                                wrap_rays=[RayMeshIntersector(unpack_solid(wrap)) for _ in range(2)],
                                work_rays=[RayMeshIntersector(work) for _ in range(2)])
        one, two = layout(1), layout(2)
        a = VolumeGuidance(model, [one]).estimate(one)
        b = VolumeGuidance(model, [two]).estimate(two)
        self.assertAlmostEqual(a, b, places=12)
        sweep = C.S.solid(C.S.swept_solid(body, [0., 0., .2]))
        actual = C.material_volume(wrap-sweep)*1e6
        self.assertLess(abs(a-actual), 1.5)

    def test_passing_blocker_requires_explicit_juxtapose_selection(self):
        model=SimpleNamespace(poses=['pose_1','pose_2'])
        search=ReuseFirstSearch(model,None)
        current=dict(layout=layout(2),masks={0:np.array([True,False]),1:np.ones(2,bool)},
                     counts={'0':1,'1':2})
        proposed=[]
        search.targets=lambda result:([],None,[])
        search.juxtapose_proposals=lambda result,guests,targets:proposed.extend(guests) or []
        search.shortlist=lambda proposals,targets:([],0)
        search.record=lambda row:None
        search.rescue(current,[1])
        self.assertEqual(proposed,[])
        search.rescue(current,[1],blocking_guests=(1,))
        self.assertEqual(proposed,[1])

    def test_coherent_fine_translation_preserves_world_height_and_registered_pose(self):
        from fast_search import FastReuseSearch
        native=np.array([np.eye(4),C.trimesh.transformations.rotation_matrix(.7,[0.,1.,0.]),
                         C.trimesh.transformations.rotation_matrix(-.6,[1.,0.,0.])])
        model=SimpleNamespace(poses=['pose_1','pose_2','pose_3'],extent=.15,native=native,
            floor_normal=lambda q,k:native[q.hosts[k],:3,:3].T @ [0.,0.,1.],
            proxy=lambda q,targets:dict(loss=0.,sum_loss=0.,span_m=0.))
        q=layout(3);q.placements[1,0,3]=.01;q.hosts[2]=0;q.placements[2]=native[2]
        current=dict(layout=q,masks={k:np.zeros(2,bool) for k in q.active})
        search=FastReuseSearch(model,None);search.fine_resolution=True
        proposals,_=search.local_proposals(current,[],dict(pose_index=1),translation_guests=(0,1,2))
        coherent=[trial for kind,trial,detail in proposals if kind=='translation-coherent']
        self.assertTrue(coherent)
        for trial in coherent:
            np.testing.assert_array_equal(trial.placements[0],q.placements[0])
            shifts=[native[q.hosts[k],:3,:3] @ (trial.placements[k,:3,3]-q.placements[k,:3,3]) for k in [1,2]]
            np.testing.assert_allclose(shifts[0],shifts[1],atol=1e-15)
            self.assertLess(abs(shifts[0][2]),1e-15)


if __name__ == '__main__':
    unittest.main()
