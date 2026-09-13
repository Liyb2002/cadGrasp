"""Fixed pad geometry, whole-body paths, and isolation of failed siblings."""
from pathlib import Path
import sys
import unittest
import copy
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import fixed_feet as H,layout as L
from step5_connect_support.test_connection import contact
from step2_local_support import insertion as D,work_volume as W
from step4_floor_contact.footprints import rectangle


class FixedFeetTests(unittest.TestCase):
    def setup_case(self):
        mesh=trimesh.creation.box([1., 1., 1.]);mesh.apply_translation([0, 1., 0])
        patch=contact(mesh,[0,-1,0],0)
        pads=[rectangle(np.array(c)-.02,np.array(c)+.02).tolist() for c in [(-1.1,-.3),(-.8,-.3),(-.8,.3),(-1.1,.3)]]
        foot=dict(candidate_id='C0',pads_xz_m=pads,height_m=.018,fixed_for_step5=True,
                  center_xz_m=[-.95,0],bearing_area_m2=4*.04**2)
        work=W.WorkVolume([[[4.,4.,4.],[5.,4.,4.],[4.,5.,4.]]],[[0,0,1]],[0],30.,1.)
        direction=dict(certified_directions=D.normalize(isolated=[0.]))
        return mesh,patch,direction,foot,work

    def test_routes_reach_all_fixed_pads_without_moving_them(self):
        mesh,patch,direction,foot,work=self.setup_case();before=copy.deepcopy(foot)
        result,module=H.build_one(mesh,patch,direction,foot,.01,work,edge_budget=50)
        self.assertTrue(result['passed'],result)
        self.assertEqual(foot,before)
        np.testing.assert_array_equal(module['plan']['ground_polygons_xz_m'],foot['pads_xz_m'])
        self.assertEqual(len(module['plan']['routes']),4)
        self.assertTrue(module['solid']['one_solid'])
        self.assertTrue(result['object_sweep']['passed'])

    def test_fixed_foot_collision_fails_without_resizing_or_suppressing_next_support(self):
        mesh,patch,direction,foot,work=self.setup_case()
        failed={**foot,'candidate_id':'bad','pads_xz_m':[[[4.,4.],[4.1,4.],[4.1,4.1],[4.,4.1]]]}
        # A reserved ray volume beginning at the ground blocks this actual pad.
        reserved=W.WorkVolume([[[3.,0.,3.],[6.,0.,3.],[3.,0.,6.]]],[[0,1,0]],[0],30.,1.)
        rejected,_=H.build_one(mesh,{**patch,'candidate_id':'bad'},direction,failed,.01,reserved)
        self.assertFalse(rejected['passed'])
        calls=[]
        def builder(mesh,c,d,f,depth,w):
            calls.append(c['candidate_id'])
            if c['candidate_id']=='bad':raise RuntimeError('one support failed')
            return H.build_one(mesh,c,d,f,depth,w,edge_budget=50)
        results,modules=H.search(mesh,[{**patch,'candidate_id':'bad'},patch],[direction,direction],[failed,foot],.01,work,builder=builder)
        self.assertEqual(calls,['bad','C0'])
        self.assertFalse(results[0]['passed']);self.assertTrue(results[1]['passed'])
        self.assertEqual([m['plan']['candidate_id'] for m in modules],['C0'])

    def test_individual_paths_survive_pair_order_cycle(self):
        obstacle=trimesh.creation.box([.2, .2, .2]);obstacle.apply_translation([0, 1, 3])
        modules=[]
        for x,a in [(1.,[1.,0,0]),(-1.,[-1.,0,0])]:
            solid=trimesh.creation.box([.2, .4, .2]);solid.apply_translation([x, .2, 0])
            modules.append(dict(joined=solid,parts=[solid],plan=dict(direction=np.array(a))))
        result=L.check_assembly(obstacle,modules)
        self.assertFalse(result['passed'])
        self.assertTrue(all(r['passed'] for r in result['object_sweeps']))
        subset,order=H.compatible_subset(result,2)
        self.assertEqual(subset,[0]);self.assertEqual(order,[0])
        self.assertEqual(len(modules),2)

    def test_overlapping_solids_are_not_presented_as_compatible_subset(self):
        report=dict(volume_tolerance_m3=1e-10,precedence_edges=[],pair_checks=[
            dict(first=0,second=1,final_intersection_m3=.1),
            dict(first=0,second=2,final_intersection_m3=0),
            dict(first=1,second=2,final_intersection_m3=0)])
        subset,_=H.compatible_subset(report,3)
        self.assertEqual(subset,[0,2])

    def test_subset_excludes_a_failed_whole_object_sweep(self):
        report=dict(volume_tolerance_m3=1e-10,precedence_edges=[],pair_checks=[],
                    object_sweeps=[dict(passed=False),dict(passed=True)])
        subset,order=H.compatible_subset(report,2)
        self.assertEqual(subset,[1]);self.assertEqual(order,[1])

    def test_missing_direction_record_is_not_silently_dropped(self):
        mesh,patch,direction,foot,work=self.setup_case()
        with self.assertRaisesRegex(ValueError,'Every contact'):
            H.search(mesh,[patch],[],[foot],.01,work)

    def test_success_is_published_before_the_next_support_is_attempted(self):
        events=[]
        def builder(mesh,contact,direction,foot,depth,work):
            cid=contact['candidate_id'];events.append(('build',cid))
            if cid=='bad':raise RuntimeError('failed sibling')
            return dict(candidate_id=cid,passed=True,status='verified'),dict(candidate_id=cid)
        def publish(record,module):
            events.append(('publish',record['candidate_id']))
        contacts=[dict(candidate_id=cid) for cid in ['good','bad','missing']]
        records,modules=H.search(None,contacts,[{}]*3,contacts[:2],.1,None,builder=builder,on_result=publish)
        self.assertEqual(events,[('build','good'),('publish','good'),('build','bad'),('publish','bad'),('publish','missing')])
        self.assertEqual(len(modules),1)
        self.assertEqual([r['passed'] for r in records],[True,False,False])

    def test_no_success_still_draws_all_fixed_targets_and_contacts(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch as mock
        from PIL import Image
        from step5_connect_support import draw
        mesh,contact,direction,foot,work=self.setup_case()
        foot['hull_xz_m']=rectangle([-1.12,-.32],[-.78,.32]).tolist()
        domain=SimpleNamespace(mesh=mesh,work_ids=np.array([],int))
        floor=dict(report=dict(ground_footprints=[foot]))
        report=dict(support_count=0,contact_count=1,supports=[],assemblable_subset_ids=[],
                    geometric_assembly_verified=False,step4_bearing_certificate_verified=False,
                    support_results=[dict(candidate_id='C0',passed=False,status='no_connector_in_fixed_foot_search_menu')])
        schedule=dict(rounds=[dict(candidate_id='C0',round=1)])
        with TemporaryDirectory() as directory:
            out=Path(directory)
            draw.overview('box',domain,[contact],[],floor,report,out)
            with mock.object(draw.V,'reference_camera',return_value=(draw.V.VIEW,{})):
                draw.positions('box',domain,[contact],[],floor,report,out,schedule)
            draw.ground_sheet('box',domain,floor,report,out)
            for file in ['connection.png','contact_positions.png','ground_feet.png','supports/C0/support.png']:
                with Image.open(out/file) as picture:
                    self.assertGreater(np.asarray(picture).std(),1.)

    def test_interruption_keeps_success_failure_previews_and_overview(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from PIL import Image
        from step5_connect_support import connect,draw,solids
        mesh,patch,direction,foot,work=self.setup_case()
        contacts=[{**patch,'candidate_id':cid} for cid in ('good','bad','interrupted')]
        feet=[{**foot,'candidate_id':c['candidate_id']} for c in contacts]
        domain=SimpleNamespace(mesh=mesh,work_ids=np.array([],int))
        floor=dict(report=dict(ground_footprints=feet))
        records=[];previews=[]
        def builder(mesh,c,d,f,depth,w):
            if c['candidate_id']=='interrupted':raise KeyboardInterrupt()
            if c['candidate_id']=='bad':raise RuntimeError('failed sibling')
            return H.build_one(mesh,c,d,f,depth,w,edge_budget=50)
        with TemporaryDirectory() as directory:
            out=Path(directory)
            def publish(record,module):
                cid=record['candidate_id'];destination=out/'supports'/cid
                connect.I.save(destination/'status.json',record)
                if module is not None:
                    connect.export_geometry(destination,module)
                    previews.append((dict(candidate_id=cid,ground_color=[41,139,147]),
                        solids.pack_parts(module['parts'],module['labels'],module['joined'])))
                records.append(record)
                draw.progress_snapshot('box',domain,contacts,previews,floor,records,out)
            with self.assertRaises(KeyboardInterrupt):
                H.search(mesh,contacts,[direction]*3,feet,.01,work,builder=builder,on_result=publish)
            self.assertEqual([r['passed'] for r in records],[True,False])
            self.assertTrue((out/'supports/good/support.stl').is_file())
            self.assertFalse((out/'supports/bad/support.stl').exists())
            for filename in ('connection.png','supports/good/support.png','supports/bad/support.png'):
                with Image.open(out/filename) as picture:
                    self.assertGreater(np.asarray(picture).std(),1.)

    def test_routed_bar_cannot_add_unrecorded_ground_contact(self):
        from step5_connect_support import routing
        mesh,_,_,_,work=self.setup_case()
        router=routing.Router(mesh,work,0.)
        touching=routing.beam(np.array([-1.,router.radius,0.]),
                              np.array([-1.2,router.radius,0.]),router.radius,router.radius)
        self.assertFalse(router.clear(touching))
        touching.apply_translation([0, router.scale*1e-8, 0])
        self.assertTrue(router.clear(touching))

    def test_later_direct_connection_precedes_early_expensive_spatial_search(self):
        from unittest.mock import patch as mock
        mesh,contact,_,foot,work=self.setup_case()
        direction=dict(certified_directions=D.normalize(isolated=[0.,180.]))
        events=[]
        class Router(H.T.Router):
            def __init__(self,mesh,work,angle,**kwargs):
                super().__init__(mesh,work,angle,**kwargs);self.angle=angle
            def connect(self,heads,anchor,height,allow_roadmap=True):
                events.append((self.angle,allow_roadmap))
                if self.angle==180.:
                    if allow_roadmap:raise AssertionError('Tried expensive search before the later direct path')
                    return None
                return super().connect(heads,anchor,height,allow_roadmap=allow_roadmap)
        with mock.object(H.T,'Router',Router),mock.object(H,'angles',return_value=[180.,0.]):
            result,module=H.build_one(mesh,contact,direction,foot,.01,work,edge_budget=50)
        self.assertTrue(result['passed'])
        self.assertEqual(module['plan']['bearing_deg'],0.)
        self.assertTrue(all(not spatial for _,spatial in events))
        self.assertEqual(len(module['plan']['routes']),4)


if __name__=='__main__':unittest.main()
