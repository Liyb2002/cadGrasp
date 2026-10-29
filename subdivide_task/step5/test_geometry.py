"""Ground enclosure and centered-execution evaluation invariants."""
import unittest
import numpy as np
from subdivide_task.step3.legacy import kernel as K
from .geometry import enclosure
from .geometry import SparseFeet,solid,unpack,volume,components
from subdivide_task.evaluation.metrics import static_bounds


class GeometryTests(unittest.TestCase):
    def test_shared_floor_design_preserves_demands_under_different_xy_poses(self):
        from .ground_design import shared_plane_rows
        from subdivide_task.operations.state import transform
        T=np.eye(4);T[:2,:2]=[[0.,-1.],[1.,0.]];T[:2,3]=[.2,-.3]
        hull=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
        rows=[dict(fixture_transform=M,hull=hull,pose=str(k)) for k,M in enumerate([np.eye(4),T])]
        original=[dict(r) for r in rows];grouped=shared_plane_rows(rows)
        np.testing.assert_array_equal(grouped[0]['fixture_transform'],grouped[1]['fixture_transform'])
        for a,b in zip(original,grouped):
            np.testing.assert_allclose(transform(a['hull'],np.linalg.inv(a['fixture_transform'])),
                transform(b['hull'],np.linalg.inv(b['fixture_transform'])),atol=1e-15)
        np.testing.assert_array_equal(rows[1]['fixture_transform'],T)

    def test_sparse_connectors_route_around_a_corner_without_a_large_bridge(self):
        from subdivide_task.step3 import independent_geometry as G
        allowed=G.box([0,0,0],[.3,.035,.04])+G.box([.265,0,0],[.3,.3,.04])
        core=G.box([.005,.005,.005],[.02,.025,.025])+G.box([.275,.275,.005],[.295,.295,.025])
        builder=SparseFeet(G.mesh(core),G.mesh(allowed),[])
        builder.connect()
        self.assertEqual(len(components(builder.body)),1)
        self.assertLessEqual(volume(builder.body-builder.allowed),1e-10)
        self.assertLessEqual(volume(core-builder.body),1e-10)
        self.assertLess(volume(builder.body),.5*volume(allowed))
        self.assertTrue(any(h['operation']=='boundary_surface_route' for h in builder.history))

    def test_removing_small_islands_preserves_an_enclosed_cavity(self):
        from subdivide_task.step3 import independent_geometry as G
        shell=G.box([0,0,0],[.2,.2,.2])-G.box([.05,.05,.05],[.15,.15,.15])
        cleaned=SparseFeet.regularize_material(object.__new__(SparseFeet),shell)
        self.assertAlmostEqual(volume(cleaned),volume(shell),places=12)
        self.assertLessEqual(volume(cleaned-shell),1e-13)

    def test_disjoint_feet_enclose_without_filled_ground_ring(self):
        feet=[]
        for x,y in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            mesh=K.trimesh.creation.box(extents=[.05,.05,.05]);mesh.apply_translation([x,y,.025]);feet.append(mesh)
        rows=[dict(pose='test',fixture_transform=np.eye(4),object_ground=np.empty((0,2)),
            hull=np.array([[-.9,-.9,0],[.9,-.9,0],[.9,.9,0],[-.9,.9,0]]))]
        self.assertTrue(enclosure(K.trimesh.util.concatenate(feet),rows)[0]['passed'])
        self.assertFalse(enclosure(K.trimesh.util.concatenate(feet[:3]),rows)[0]['passed'])

    def test_evaluation_counts_actual_translated_installations(self):
        body=K.trimesh.creation.box().vertices
        T=np.eye(4);T[0,3]=1.
        placements=np.repeat(np.eye(4)[None],2,axis=0);placements[1,0,3]=-1
        layout=dict(active=np.array([0,1]),native_world=np.repeat(np.eye(4)[None],2,axis=0),
            hosts=np.array([0,0]),placements=placements)
        execution=[dict(world_from_support=np.eye(4)),dict(world_from_support=T)]
        result=static_bounds(body,layout,full_vertices=body,execution=execution)
        self.assertEqual(result['support_pose_count'],2)
        self.assertEqual(result['object_pose_counts_per_support_pose'],[1,1])
        np.testing.assert_allclose(result['objects_in_final_layout']['dimensions_m'],[1,1,1])
        np.testing.assert_allclose(result['with_full_support']['dimensions_m'],[2,1,1])

    def test_unsupported_baseline_does_not_borrow_support_translations(self):
        body=K.trimesh.creation.box(extents=[2,3,4]).vertices
        native=np.repeat(np.eye(4)[None],2,axis=0);native[1,0,3]=12.
        layout=dict(active=np.array([0,1]),native_world=native,hosts=np.array([0,1]),placements=np.repeat(np.eye(4)[None],2,axis=0))
        result=static_bounds(body,layout,full_vertices=body)
        np.testing.assert_allclose(result['object_static']['dimensions_m'],[2,3,4])
        self.assertEqual(result['objects_in_final_layout']['dimensions_m'][0],14)


if __name__=='__main__':unittest.main()
