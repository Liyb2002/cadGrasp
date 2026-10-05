"""Connection witnesses: multi-floor counterexample, actual solids and hard gates."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import connection as B, directions as T, contacts as I
from step2_local_support import withdrawal as D
from step3_scheculer.stage_imports import load_stage
A=load_stage('optimize','adjust')
S=load_stage('optimize','size_search')


def contact(mesh, normal, index):
    face=int(np.argmax(mesh.face_normals@np.asarray(normal)))
    triangle=mesh.triangles[face]
    center=triangle.mean(axis=0)
    triangle=center+.2*(triangle-center)
    area=.5*np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))
    return dict(candidate_id=f'C{index}',candidate_index=index,source_faces=np.array([face]),
                triangles_m=triangle[None],triangle_areas_m2=np.array([area]),center_m=center,
                center_face=face,radius_m=.2)


def setup(planes=None):
    mesh=trimesh.creation.box(extents=[2.,2.,2.])
    mesh.apply_translation([0,0,2.])
    cat=dict(vectors=[[1.,0,0],[-1.,0,0]],global_allowed_directions=D.normalize([0,1]),
             preferred_withdrawal_direction=[1.,0,0])
    checker=B.Checker(mesh,.1,cat,floor_planes=planes,half_width=.08,gap=.02)
    heads=[contact(mesh,[0,0,1],0),contact(mesh,[0,0,-1],1)]
    return mesh,checker,heads


class ConnectionTests(unittest.TestCase):
    def test_final_output_checks_actual_sweep_after_sufficient_connection_passes(self):
        from step3_scheculer import verification as V
        mesh,checker,heads=setup()
        problem=SimpleNamespace(domain=SimpleNamespace(mesh=mesh),_connection_checker=checker)
        directions=dict(common_directions=D.normalize([0,1]))
        catalogue=dict(normal_depth_m=.1,direction_catalogue=checker.catalogue)
        self.assertTrue(checker.check(heads,directions['common_directions'])['passed'])
        self.assertTrue(V.final_connection_check(problem,heads,directions,catalogue)['passed'])
        with patch.object(D.Analyzer,'test',return_value=dict(clear=False)):
            result=V.final_connection_check(problem,heads,directions,catalogue)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'],'final_connection_geometry_failed')
        self.assertFalse(V.final_connection_check(problem,[],directions,catalogue)['passed'])

    def test_nearly_tangent_floor_cannot_supply_missing_joint_height(self):
        # Real B/pose_2 direction 1042 and transformed initial-floor normal:
        # their nominal positive dot is only roundoff, formerly yielding 1e13 m.
        n=np.array([.000746755254538918,.7069845898378712,.7072285571782061])
        u=np.array([-.6114418024501076,-.5592997194748645,.5597523970577543])
        self.assertLess(abs(float(u@n)),64*np.finfo(float).eps)
        mesh,_,contacts=setup()
        cat=dict(vectors=[u.tolist(),n.tolist()],global_allowed_directions=D.normalize([0,1]))
        checker=B.Checker(mesh,.1,cat,floor_planes=[[*n,0]],half_width=.08,gap=.02)
        def heads(c):
            box=trimesh.creation.box(extents=[.01,.01,.01])
            box.apply_translation(.04*n+(2*c['candidate_index']-1)*.1*u)
            return [box]
        with patch.object(checker.analyzer,'heads',side_effect=heads):
            # Each small head is above the floor; its wider rear joint needs lift.
            self.assertTrue(all(checker.head(c)['valid'] for c in contacts))
            result=checker.check(contacts,D.normalize([0]))
            self.assertFalse(result['passed'])
            self.assertEqual(result['status'],'no_connection_witness')
            # A genuine upward direction still permits a finite solid witness.
            result=checker.check(contacts,D.normalize([1]))
            self.assertTrue(result['passed'])
            self.assertLess(abs(result['witness']['plane_offset_m']),10.)
            vertices=np.vstack([p.vertices for p in checker.parts(contacts,result['witness'])])
            self.assertGreaterEqual(np.min(vertices@n),-checker.eps)

    def test_floor_roundoff_guard_is_a_lower_bound_not_flat_relaxation(self):
        n=np.array([0.,0.,1.])
        vectors=np.array([[1.,0.,1e-17],[1.,0.,-1e-17],[0.,0.,1.]])
        lower=B.floor_slope_lower(vectors,n)
        self.assertTrue(np.all(lower<=vectors@n))
        self.assertTrue(np.all(lower[:2]<0.))
        self.assertGreater(lower[2],.999999999999)
        # With any nonnegative extrusion distance the guarded prediction never
        # claims a higher floor clearance than the nominal dot product.
        distances=np.array([0.,.1,100.])
        self.assertTrue(np.all(distances*lower<=distances*(vectors@n)))

    def test_same_heads_have_a_thick_connected_and_removable_witness(self):
        mesh,c,heads=setup()
        result=c.check(heads,D.normalize([0,1]))
        self.assertTrue(result['passed'])
        parts=c.parts(heads,result['witness'])
        self.assertGreater(min(p.vertices[:,2].min() for p in parts),0)
        analyzer=D.Analyzer(mesh,.1,c.catalogue)
        u=result['witness']['withdrawal_direction']
        self.assertTrue(analyzer.test(parts,u)['clear'])
        actual_heads=[p for head in heads for p in analyzer.heads(head)]
        obstacle=D.solid(mesh,np.zeros(3),2.)
        solid=None
        for part in actual_heads+parts:
            p=D.solid(part,np.zeros(3),2.)
            self.assertLess(abs((p ^ obstacle).volume()),1e-12)
            solid=p if solid is None else solid+p
        self.assertEqual(len([p for p in solid.decompose() if abs(p.volume())>1e-12]),1)

    def test_individually_clear_heads_and_shared_rays_do_not_imply_connection(self):
        planes=[[1,0,0,1],[-1,0,0,1],[0,1,0,1],[0,-1,0,1]]
        mesh,c,heads=setup(planes)
        analyzer=D.Analyzer(mesh,.1,c.catalogue)
        for h in heads:
            self.assertTrue(c.check([h],D.normalize([0,1]))['passed'])
            self.assertTrue(analyzer.test(analyzer.heads(h),[1.,0,0])['clear'])
        result=c.check(heads,D.normalize([0,1]))
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'],'no_connection_witness')

    def test_filter_rejects_candidate_before_scoring_when_connection_is_blocked(self):
        _,c,heads=setup([[1,0,0,1],[-1,0,0,1],[0,1,0,1],[0,-1,0,1]])
        rows=[dict(candidate_index=i,candidate_id=h['candidate_id'],certified_directions=D.normalize([0,1])) for i,h in enumerate(heads)]
        eligible,records=T.candidate_filter(rows,[0],np.ones(2,bool),common_allowed=D.normalize([0,1]),
            connection_checker=c,selected_contacts=heads[:1],candidate=lambda i:heads[i])
        self.assertFalse(eligible[1])
        self.assertEqual(records[1]['reason'],'no_common_connection_witness')
        self.assertFalse(records[1]['connection']['passed'])

    def test_tilted_floor_checks_actual_cube_width(self):
        _,c,heads=setup([[0,0,1,0],[1,0,1,5.]])
        result=c.check(heads,D.normalize([0,1]))
        self.assertTrue(result['passed'])
        points=np.vstack([p.vertices for p in c.parts(heads,result['witness'])])
        self.assertGreaterEqual(np.min(points@c.planes[:,:3].T+c.planes[:,3]),-1e-10)

    def test_actual_head_floor_collision_is_rejected_even_for_one_head(self):
        _,c,heads=setup([[0,0,1,-1.]])
        self.assertFalse(c.check(heads[1:],D.normalize([0]))['passed'])

    def test_shrinking_rechecks_connections_instead_of_inheriting_a_witness(self):
        scalar=A.SizeProblem.__new__(A.SizeProblem)
        scalar.direction_cache={}
        scalar.geometry=lambda r:dict(row=dict(geometry_valid=True))
        scalar.contact=lambda r:dict(radius_m=r)
        scalar.fixed=[];scalar.allowed_directions=D.normalize([0])
        scalar.inherited_direction_radius=1.;scalar.inherited_directions=D.normalize([0])
        calls=[]
        def check(contacts,allowed):
            calls.append(contacts[-1]['radius_m'])
            return dict(passed=False,directions=D.normalize())
        scalar.connection_checker=SimpleNamespace(check=check)
        self.assertEqual(scalar.insertion_directions(.5),D.normalize())
        self.assertEqual(calls,[.5])

    def test_old_greedy_ratio_search_cannot_accept_disconnected_best_size(self):
        evaluate=lambda r:dict(area_m2=r,covered_count=100,total_area_m2=r)
        radius,report=S.maximize_efficiency(evaluate,[.25,1.,2.],1.,.01,max_evaluations=16,
                                           admissible=lambda r:r>=.8)
        self.assertGreaterEqual(radius,.8)
        self.assertLessEqual(radius,1.)

    def test_shape_cache_uses_actual_geometry_and_not_just_candidate_id(self):
        _,c,heads=setup()
        first=c.head(heads[0])
        changed=dict(heads[0])
        changed['triangles_m']=heads[0]['center_m']+.5*(heads[0]['triangles_m']-heads[0]['center_m'])
        changed['triangle_areas_m2']=heads[0]['triangle_areas_m2']*.25
        changed['radius_m']=.1
        self.assertIs(c.head(heads[0]),first)
        second=c.head(changed)
        self.assertNotEqual(first['signature'],second['signature'])
        self.assertEqual(len(c.cache),2)

    def test_optional_work_volume_is_not_silently_ignored(self):
        _,c,heads=setup()
        c.work=SimpleNamespace(check_parts=lambda parts:dict(passed=False))
        self.assertFalse(c.check(heads,D.normalize([0]))['passed'])

    def test_downstream_uses_only_joint_connection_and_head_directions(self):
        from step6_connect_support.direction_first import scheduled_directions
        _,checker,_=setup()
        record=dict(mode=D.MODE,direction_catalogue=checker.catalogue,
                    contacts=[dict(certified_directions=D.normalize([0,1]))],
                    common_directions=D.normalize([0,1]),
                    connection=dict(directions=D.normalize([1])))
        vectors,report=scheduled_directions(record)
        self.assertEqual(report['chosen_direction_ids'],[1])
        np.testing.assert_array_equal(vectors,[[-1.,0,0]])
        record['connection']['directions']=D.normalize([0,1])
        record['direction_catalogue']['preferred_withdrawal_direction']=[-1.,0,0]
        _,report=scheduled_directions(record)
        self.assertEqual(report['chosen_direction_ids'],[1,0])
        self.assertEqual(record['connection']['directions']['ids'],[0,1])


if __name__=='__main__':unittest.main()
