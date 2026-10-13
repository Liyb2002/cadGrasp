"""Construction invariants and static pose/state occupied-space math."""
from pathlib import Path
import sys,unittest,tempfile
import numpy as np
import trimesh

COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'step5.3'))
sys.path.insert(0,str(COOPT/'evaluation'))
from deterministic_base import Builder,solid,unpack,box,volume
from metrics import static_bounds, unsupported_transforms, transformed


class StaticBoundsTests(unittest.TestCase):
    def layout(self):
        T=np.eye(4);T[:3,3]=[1,2,3]
        return dict(active=np.array([0]),native_world=np.array([T]),hosts=np.array([0]),
            placements=np.array([np.eye(4)]),directions=np.array([[1.,0,0]]))

    def test_only_seated_object_is_counted(self):
        vertices=np.array([[0,0,0],[1,1,1]],float)
        result=static_bounds(vertices,self.layout())['objects_in_final_layout']
        self.assertEqual(result['min_m'],[1,2,3]);self.assertEqual(result['max_m'],[2,3,4])
        self.assertEqual(result['volume_cm3'],1e6)

    def test_host_rotation_rotates_object(self):
        layout=self.layout();layout['native_world'][0,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        vertices=np.array([[0,0,0],[2,1,1]],float)
        result=static_bounds(vertices,layout)['object_static']
        np.testing.assert_allclose(result['dimensions_m'],[1,2,1])

    def test_support_is_counted_at_every_installed_state(self):
        layout=self.layout()
        result=static_bounds(np.array([[0,0,0],[1,1,1]],float),layout,
            np.array([[-2,-2,-2],[2,2,2]],float),np.array([[-3,-3,-3],[3,3,3]],float))
        self.assertGreater(result['with_step4_support']['volume_cm3'],result['object_static']['volume_cm3'])
        self.assertGreater(result['with_full_support']['volume_cm3'],result['with_step4_support']['volume_cm3'])

    def test_direction_does_not_change_static_occupancy(self):
        layout=self.layout();vertices=np.array([[0,0,0],[1,1,1]],float)
        expected=static_bounds(vertices,layout)
        layout.pop('directions')
        self.assertEqual(static_bounds(vertices,layout),expected)

    def test_separate_pose_spaces_and_shared_state_are_distinct(self):
        layout=self.layout();layout['active']=np.array([0,1]);layout['hosts']=np.array([0,0])
        layout['native_world']=np.repeat(layout['native_world'],2,axis=0)
        layout['placements']=np.repeat(layout['placements'],2,axis=0)
        layout['placements'][1,0,3]=.5
        r=static_bounds(np.array([[0,0,0],[1,1,1]],float),layout)
        self.assertEqual(r['independent_pose_object_sum_litres'],2000)
        self.assertEqual(r['object_static']['volume_litres'],1000)
        self.assertEqual(r['state_space_sums_litres']['objects_in_final_layout'],1500)
        self.assertEqual(r['state_count'],1);self.assertEqual(r['multi_pose_fixture_state_count'],1)
        self.assertEqual(r['juxtaposed_pose_count'],1);self.assertEqual(r['rotating_reuse_pose_count'],1)
        self.assertEqual(r['fixture_states'][0]['poses'],['pose_1','pose_2'])

    def test_no_support_baseline_ignores_fixture_offsets_and_hosts(self):
        layout=self.layout();layout['active']=np.array([0,1]);layout['hosts']=np.array([0,0])
        layout['native_world']=np.repeat(layout['native_world'],2,axis=0)
        layout['placements']=np.repeat(layout['placements'],2,axis=0)
        body=trimesh.creation.box([2.,1.,.5]).vertices
        before={k:v.copy() for k,v in layout.items()}
        plain=static_bounds(body,layout)
        np.testing.assert_allclose(plain['object_static']['dimensions_m'],[2,1,.5])
        for key in before:np.testing.assert_array_equal(layout[key],before[key])
        layout['placements'][1,:3,3]=[100,200,300]
        layout['native_world'][1,:3,3]=[-900,500,70]
        spaced=static_bounds(body,layout)
        self.assertEqual(spaced['object_static'],plain['object_static'])
        self.assertGreater(spaced['objects_in_final_layout']['volume_litres'],
                           plain['objects_in_final_layout']['volume_litres'])
        self.assertEqual(spaced['object_static']['min_m'][2],0.)

    def test_overlapping_box_attains_rotation_extent_lower_bound(self):
        body=trimesh.creation.box([2.,1.,.5]).vertices + [3,8,-5]
        layout=self.layout();layout['active']=np.array([0,1]);layout['hosts']=np.array([0,0])
        layout['native_world']=np.repeat(layout['native_world'],2,axis=0)
        layout['native_world'][1,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        layout['placements']=np.repeat(layout['placements'],2,axis=0)
        matrices=unsupported_transforms(body,layout)
        extents=[]
        for k,T in matrices.items():
            np.testing.assert_array_equal(T[:3,:3],layout['native_world'][k,:3,:3])
            vertices=transformed(body,T);lo,hi=vertices.min(0),vertices.max(0)
            np.testing.assert_allclose((lo[:2]+hi[:2])/2,0.,atol=1e-12)
            self.assertAlmostEqual(lo[2],0.)
            extents.append(hi-lo)
        r=static_bounds(body,layout)
        np.testing.assert_allclose(r['object_static']['dimensions_m'],np.max(extents,axis=0))
        np.testing.assert_allclose(r['object_static']['dimensions_m'],[2,2,.5])
        self.assertEqual(r['object_static']['volume_litres'],2000.)

    def test_duplicate_transforms_share_state_but_distinct_states_count_support(self):
        layout=self.layout();layout['active']=np.array([0,1]);layout['hosts']=np.array([0,1])
        layout['native_world']=np.repeat(layout['native_world'],2,axis=0)
        layout['placements']=np.repeat(layout['placements'],2,axis=0)
        v=np.array([[0,0,0],[1,1,1]],float);support=np.array([[-1,-1,-1],[2,2,2]],float)
        r=static_bounds(v,layout,support,support)
        self.assertEqual(r['state_count'],1)
        self.assertEqual(r['state_space_sums_litres']['with_full_support'],27000)
        layout['native_world'][1,0,3]+=10
        r=static_bounds(v,layout,support,support)
        self.assertEqual(r['state_count'],2);self.assertEqual(r['singleton_fixture_state_count'],2)
        self.assertEqual(r['state_space_sums_litres']['with_full_support'],54000)
        self.assertEqual(r['with_full_support']['volume_litres'],117000)
        self.assertEqual(r['object_static']['volume_litres'],1000)

    def test_support_pose_membership_uses_installation_not_host_or_object_seat(self):
        layout=self.layout();layout['active']=np.arange(5);layout['hosts']=np.array([0,1,2,3,3])
        layout['native_world']=np.repeat(layout['native_world'],5,axis=0)
        layout['native_world'][2,0,3]+=2
        layout['native_world'][3,0,3]+=4
        layout['placements']=np.repeat(layout['placements'],5,axis=0)
        layout['placements'][1,0,3]=.5
        layout['placements'][4,0,3]=-.5
        names=['pose_2','pose_4','pose_6','pose_8','pose_20']
        r=static_bounds(np.array([[0,0,0],[1,1,1]],float),layout,pose_names=names)
        self.assertEqual(r['support_pose_count'],3)
        self.assertEqual(r['object_pose_counts_per_support_pose'],[2,1,2])
        self.assertEqual([s['poses'] for s in r['fixture_states']],
                         [['pose_2','pose_4'],['pose_6'],['pose_8','pose_20']])
        self.assertEqual(sum(r['object_pose_counts_per_support_pose']),r['pose_count'])


class DeterministicBaseTests(unittest.TestCase):
    def example(self,obstacle=False):
        legal=box(np.array([-.06,-.06,0]),np.array([.06,.06,.07]))
        if obstacle:legal=legal-box(np.array([-.008,-.012,.005]),np.array([.008,.012,.065]))
        original=box(np.array([-.04,-.005,.025]),np.array([-.03,.005,.035]))+box(np.array([.03,-.005,.025]),np.array([.04,.005,.035]))
        hull=np.array([[-.045,-.045,0],[.045,-.045,0],[.045,.045,0],[-.045,.045,0]])
        row=dict(pose='test',fixture_transform=np.eye(4),hull=hull,object_ground=np.empty((0,2)))
        return Builder(unpack(original),unpack(legal),[row]),legal,original

    def test_compact_feet_and_one_connected_solid(self):
        builder,legal,original=self.example()
        mesh,report=builder.run()
        self.assertTrue(report['passed']);self.assertTrue(mesh.is_watertight)
        self.assertEqual(report['retained_legal_step4_material_loss_cm3'],0.)
        self.assertLess(report['added_material_cm3'],15.)
        self.assertTrue(report['history'][0]['compact_feet_used'])

    def test_connections_avoid_obstacle_and_keep_contacts(self):
        builder,legal,original=self.example(True)
        mesh,report=builder.run()
        self.assertTrue(report['passed']);self.assertLess(volume(solid(mesh)-legal),1e-10)
        self.assertLess(volume(original-solid(mesh)),1e-10)
        self.assertTrue(report['floor_pressure_enclosure_passed'])

    def test_foot_window_uses_task_floor_axes_after_fixture_rotation(self):
        # A long narrow support's common-axis box becomes a much wider box
        # on a rotated floor. Feet must follow the task hull, not its corners.
        angle=np.pi/4
        T=np.eye(4);T[:3,:3]=[[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],[0,0,1]]
        original=box(np.array([-.05,-.005,.02]),np.array([.05,.005,.03]))
        legal=box(np.array([-.2,-.2,0]),np.array([.2,.2,.06]))
        hull=np.array([[-.02,-.01,0],[.02,-.01,0],[.02,.01,0],[-.02,.01,0]])
        row=dict(pose='tilted',fixture_transform=T,hull=hull,object_ground=np.empty((0,2)))
        builder=Builder(unpack(original),unpack(legal),[row]);builder.ground()
        feet=[np.array(p['world_xy_m']) for p in builder.history[0]['foot_locations']]
        self.assertTrue(builder.history[0]['compact_feet_used'])
        self.assertTrue(all(np.all(np.abs(p)<=[.0250001,.0150001]) for p in feet))

    def test_export_remains_one_closed_component(self):
        builder,_,_=self.example(True)
        mesh,_=builder.run()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'support.obj';mesh.export(path,digits=16)
            saved=trimesh.load(path,force='mesh',process=False)
            self.assertTrue(saved.is_watertight)
            self.assertEqual(len(saved.split(only_watertight=False)),1)
            self.assertAlmostEqual(saved.volume,mesh.volume,places=14)

    def test_illegal_nominal_is_removed_without_losing_legal_contacts(self):
        legal=box(np.array([-.06,-.06,0]),np.array([.06,.06,.07]))
        original=box(np.array([-.01,-.01,-.002]),np.array([.01,.01,.03]))
        hull=np.array([[-.005,-.005,0],[.005,-.005,0],
                       [.005,.005,0],[-.005,.005,0]])
        row=dict(pose='clipped',fixture_transform=np.eye(4),hull=hull,
                 object_ground=np.empty((0,2)))
        builder=Builder(unpack(original),unpack(legal),[row])
        mesh,report=builder.run()
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['removed_illegal_nominal_material_cm3'],.8)
        self.assertEqual(report['retained_legal_step4_material_loss_cm3'],0.)
        self.assertGreaterEqual(mesh.bounds[0,2],-1e-12)


if __name__=='__main__':unittest.main()
