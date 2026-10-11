from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'step5.2'));sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
import numpy as np
import stage5_common as C
import base_seating as B
from base_access_seating import AccessSeating
from base_bounds import object_z_certificates
from base_floor_space import clip
import base_xy


class PlacementPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=C.Inputs('B');cls.data=cls.inputs.load(cls.inputs.cases[0]);B.load_demands(cls.data,cls.inputs)

    def seating(self):
        n=len(self.data['rows']);rng=np.random.default_rng(17)
        heights=rng.uniform(.03,.07,n);relative=rng.uniform(-.02,.02,(n,3))
        total=relative.copy();total[:,2]+=heights
        return dict(fixture_lift_m=heights.tolist(),object_relative_world_translation_m=relative.tolist(),
                    total_object_world_translation_m=total.tolist())

    def test_forward_wrench_projection_all_demands(self):
        seating=self.seating();moved=B.shifted(self.data,seating)
        for old,new,delta in zip(self.data['rows'],moved['rows'],seating['total_object_world_translation_m']):
            delta=np.asarray(delta)
            expected=old['cloud'][:,:2]+delta[:2]-delta[2]*old['horizontal_force_ratio']
            np.testing.assert_allclose(new['cloud'][:,:2],expected,atol=1e-14)
            self.assertEqual(len(new['cloud']),32768)

    def test_floor_cut_matches_direct_rigid_transform(self):
        seat=self.seating();moved=B.shifted(self.data,seat)
        h=np.asarray(seat['fixture_lift_m']);r=np.asarray(seat['object_relative_world_translation_m'])
        for i,j,a,k,s,b,margin in B.Seating(self.data).banks:
            q=C.F.transform_points(moved['rows'][i]['cloud'][137],np.linalg.inv(moved['states'][i]))
            direct=C.F.transform_points(q,moved['states'][j])[2]
            predicted=b[137]-k[137]*h[i]+h[j]-s[137]*r[i,2]+a@r[i,:2]
            self.assertAlmostEqual(direct,predicted,places=12)

    def test_access_facet_gradient_matches_forward_geometry(self):
        access=object.__new__(AccessSeating);access.data=self.data;access.n=len(self.data['rows'])
        access.rotations=np.array(self.data['states'])[:,:3,:3];access.inverse_rotations=access.rotations.transpose(0,2,1)
        seat=self.seating();moved=B.shifted(self.data,seat)
        h=np.asarray(seat['fixture_lift_m']);r=np.asarray(seat['object_relative_world_translation_m'])
        equation=np.array([.3,-.7,.2,.013])
        for i,j in [(0,1),(2,4),(1,1)]:
            hc,rc,c=access.coefficients(i,137,j,equation)
            q=C.F.transform_points(moved['rows'][i]['cloud'][137],np.linalg.inv(moved['states'][i]))
            direct=equation[:3]@(q-access.inverse_rotations[j]@r[j])+equation[3]
            self.assertAlmostEqual(float(hc@h+(rc*r).sum()+c),direct,places=12)

    def test_xy_whole_assembly_shift_preserves_common_ground_and_floors(self):
        old=self.data['rows'][0];delta=np.array([.06,-.04,0.])
        transform=old['fixture_transform'].copy();transform[:3,3]+=delta
        common=C.F.transform_points(old['cloud'],np.linalg.inv(old['fixture_transform']))
        shifted_common=C.F.transform_points(old['cloud']+delta,np.linalg.inv(transform))
        np.testing.assert_allclose(common,shifted_common,atol=1e-14)

    def test_raise_only_certificates_have_contradictory_height_bounds(self):
        from growth import floor_certificates
        failed={q['task_index'] for q in floor_certificates(self.data['rows'])}
        certificates=object_z_certificates(self.data,failed)
        self.assertTrue(certificates)
        for q in certificates:
            self.assertTrue(q['constant_failure'] or q['allowed_maximum_lift_mm']<q['required_minimum_lift_mm'])
            for key in ['lower_witness','upper_witness']:
                w=q[key]
                if w:
                    bound=q['required_minimum_lift_mm'] if key=='lower_witness' else q['allowed_maximum_lift_mm']
                    self.assertAlmostEqual(w['b_m']-w['k']*bound/1000,0.,places=12)

    def test_polygon_clip_preserves_exact_halfspace(self):
        p=clip(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]),np.array([1.,1.]),-.5)
        self.assertTrue(len(p)>=3);self.assertTrue(np.all(p.sum(1)>=.5-1e-12))

    @classmethod
    def tearDownClass(cls):cls.inputs.verify()


if __name__=='__main__':unittest.main()
