import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from step3_scheculer import shared_direction_paths as E

class ExitTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box(extents=[1,1,1]);self.mesh.apply_translation([0,0,.6])
        self.analyzer=E.PathAnalyzer(self.mesh)
    def cell(self,size,center):
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center)
        return SimpleNamespace(vertices=mesh.vertices)
    def plan(self,nodes):return dict(object_translation_waypoints_world_m=nodes)
    def test_upward_object_virtual_head_can_cross_floor(self):
        cell=self.cell([.3,.3,.02],[0,0,.09])
        result=self.analyzer.test([cell],self.plan([[0,0,0],[0,0,2]]))
        self.assertTrue(result['clear'],result)
    def test_real_object_floor_rejected(self):
        cell=self.cell([.3,.3,.02],[0,0,.09])
        result=self.analyzer.test([cell],self.plan([[0,0,0],[0,0,-2]]))
        self.assertEqual(result['reason'],'moving_object_floor_collision')
    def test_perpendicular_contact_heads_allow_lift_slide(self):
        cells=[self.cell([.3,.3,.02],[0,0,.09]),self.cell([.02,.3,.3],[.51,0,.6])]
        result=self.analyzer.test(cells,self.plan([[0,0,0],[0,0,.05],[-2,0,.05]]))
        self.assertTrue(result['clear'],result)
    def test_continuous_segment_catches_tunneling(self):
        cell=self.cell([.1,.1,.1],[1.5,0,.6])
        result=self.analyzer.test([cell],self.plan([[0,0,0],[3,0,0],[3,0,2]]))
        self.assertFalse(result['clear']);self.assertEqual(result['reason'],'finite_segment_collision')
    def test_different_lift_heights_are_retained(self):
        plans=[dict(kind='lift_then_slide',initial_object_exit_world=[0,0,1],terminal_object_exit_world=[1,0,0],lift_height_m=h) for h in [.001,.004,.01]]
        self.assertEqual(len(E.diverse(plans,8)),3)
    def test_right_angle_directions_are_retained(self):
        plans=[dict(kind='ray',initial_object_exit_world=d,terminal_object_exit_world=d) for d in [[1,0,0],[0,1,0]]]
        self.assertEqual(len(E.diverse(plans,8)),2)
    def test_step4_accepts_perpendicular_paths_for_different_poses(self):
        from step4_connect_support import select_exit_paths as P, build_coupled_saddle as S
        tasks=[SimpleNamespace(pose=str(i),domain=SimpleNamespace(mesh=self.mesh)) for i in range(2)]
        plans=[]
        for i,d in enumerate([[2.,0,0],[0,2.,0]]):
            plans.append(dict(id=i,kind='ray',support_withdrawal_world=(-np.asarray(d)/2).tolist(),object_translation_waypoints_world_m=[[0,0,0],d]))
        reports=[dict(exit_options=[dict(plan=p)]) for p in plans]
        root=trimesh.creation.box(extents=[.1,.1,.1]);root.apply_translation([4,4,1])
        roots=[S.solid(root)]*2
        window=dict(min_m=[-1,-1,0],max_m=[5,5,3])
        states,detail=P.select(tasks,reports,roots,np.array([np.eye(3)]*2),np.zeros((2,3)),window)
        self.assertTrue(detail['passed'],detail)
        self.assertTrue(detail['angle_is_soft_and_never_a_feasibility_gate'])
        self.assertEqual([p['plan']['id'] for p in states[0]['selected']],[0,1])
    def test_bend_sweep_is_union_not_global_convex_hull(self):
        mesh=E.sweep_mesh(self.mesh,self.plan([[0,0,0],[1,0,0],[1,0,1]]))
        self.assertAlmostEqual(abs(mesh.volume),3.,places=7)

if __name__=='__main__':unittest.main()
