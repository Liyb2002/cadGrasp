"""Regression checks for complete-body withdrawal and enclosing cavities."""
import unittest
import numpy as np
import trimesh
from subdivide_task.common.surface_hull import hull_vertex_ids
from . import independent_geometry as G


class IndependentGeometryTests(unittest.TestCase):
    def test_ground_selection_preserves_enclosed_cavity(self):
        stock=G.box([-.2,-.2,0],[.2,.2,.4])
        cavity=G.box([-.05,-.05,.1],[.05,.05,.2])
        floating=G.box([.5,.5,.5],[.6,.6,.6])
        result=G.grounded_component(G.union([stock-cavity,floating]))
        self.assertLess(G.volume(result^cavity),1e-12)
        self.assertLess(G.volume(result^floating),1e-12)
        self.assertAlmostEqual(G.volume(result),.4*.4*.4-.1**3,places=12)

    def test_sweep_includes_protrusions_between_endpoints(self):
        body=trimesh.creation.box([.1,.06,.08])
        body.apply_translation([0,0,.2])
        d=np.array([0.,0.,1.]);sweep,_=G.whole_object_sweep(body,d,.3)
        # A collision at an off-centre corner halfway through withdrawal
        # must be found even though both endpoint placements are clear.
        obstacle=G.box([.04,-.01,.32],[.06,.01,.34])
        self.assertLess(G.volume(G.solid(body)^obstacle),1e-12)
        end=body.copy();end.apply_translation(.3*d)
        self.assertLess(G.volume(G.solid(end)^obstacle),1e-12)
        self.assertGreater(G.volume(sweep^obstacle),1e-7)

    def test_planar_application_hull_drops_only_interior_points(self):
        points=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],float)
        ids,proof=hull_vertex_ids(points)
        self.assertEqual(set(ids),{0,1,2,3})
        self.assertEqual(proof['dimension'],2)
        self.assertLess(proof['maximum_residual_m'],1e-12)

    def test_resting_pose_uses_lowest_vertex_and_preserves_work_geometry(self):
        from .run_independent import resting_pose
        body=trimesh.creation.box([.1,.07,.16]);reference=np.eye(4)
        reference[:3,3]=[.3,-.5,.4]
        axis=np.array([0.,-1.,-1.]);axis/=np.linalg.norm(axis)
        workspace=dict(nominal_axis_world=axis,reference_head_front_world_m=[.3,-.4,.5])
        expected_head=np.asarray(workspace['reference_head_front_world_m'])-reference[:3,3]
        for angle in (0,67,180):
            T,ws,_=resting_pose(body,reference,workspace,angle,.008)
            vertices=body.vertices@T[:3,:3].T+T[:3,3]
            self.assertAlmostEqual(vertices[:,2].min(),.008,places=14)
            actual_head=(np.asarray(ws['reference_head_front_world_m'])-T[:3,3])@T[:3,:3]
            np.testing.assert_allclose(actual_head,expected_head,atol=1e-14)
            np.testing.assert_allclose(T[:3,:3].T@axis,axis,atol=1e-14)

    def test_cast_backing_follows_outline_instead_of_filling_bounding_box(self):
        parts=[]
        for x,z in [(-.06,.04),(.06,.09)]:
            piece=trimesh.creation.box([.03,.03,.02]);piece.apply_translation([x,0,z]);parts.append(piece)
        outer=trimesh.util.concatenate(parts);cast=G.casting_stock(outer,.008)
        gap=G.box([-.01,-.01,.015],[.01,.01,.025])
        backing=G.box([.055,-.005,.02],[.065,.005,.03])
        self.assertLess(G.volume(cast^gap),1e-12)
        self.assertAlmostEqual(G.volume(cast^backing),.01**3,places=12)
        actual=G.mesh(cast)
        self.assertAlmostEqual(actual.bounds[0,2],.008,places=12)
        self.assertAlmostEqual(actual.bounds[1,2],outer.bounds[1,2],places=12)

    def test_raising_a_seat_changes_ground_pressure_for_same_com_wrench(self):
        # COM wrench is invariant under joint translation, but the torque
        # about the ground changes. Padding therefore needs a fresh base test.
        demand=np.array([[.2,.3,1.,0.,0.,0.]])
        low=G.pressure_points(demand,np.array([0.,0.,.1]))
        high=G.pressure_points(demand,np.array([0.,0.,.2]))
        np.testing.assert_allclose(low,[[-.02,-.03]],atol=1e-14)
        np.testing.assert_allclose(high,[[-.04,-.06]],atol=1e-14)


if __name__=='__main__':unittest.main()
