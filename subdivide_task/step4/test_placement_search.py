"""Floor LP and placement bookkeeping for the SO(3) endgame search."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from subdivide_task.operations.state import Layout
from subdivide_task.step4.placement_search import floor_translation,merged_layout


def family(normals,offsets,points):
    material=[]
    for n,o in zip(normals,offsets):
        n=np.asarray(n,float);n/=np.linalg.norm(n)
        a=np.array([1.,0,0]) if abs(n[0])<.9 else np.array([0,1.,0])
        x=np.cross(a,n);x/=np.linalg.norm(x);y=np.cross(n,x)
        M=np.eye(4);M[:3,:3]=np.column_stack([x,y,n]);M[:3,3]=o*n;material.append(M)
    material=np.asarray(material);k=len(material)
    return dict(layout=Layout(np.tile(np.eye(4),(k,1,1)),material,np.tile([0,0,1.],(k,1))),
        pressure=[np.asarray(points,float)]*k,floor_normals=material[:,:3,2],
        floor_offsets=np.einsum('ij,ij->i',material[:,:3,2],material[:,:3,3]))


class PlacementSearchTests(unittest.TestCase):
    def test_translation_satisfies_every_cross_floor_inequality(self):
        host=family([[0,0,1],[1,0,.2]],[0.,-.3],[[0,0,0],[.1,.1,0],[-.2,0,.05]])
        guest=family([[0,0,1],[0,1,.3]],[0.,-.2],[[0,0,0],[.05,-.1,.02]])
        R=Rotation.from_rotvec([.2,-.1,.4]).as_matrix()
        target=np.array([.4,-.3,.2]);t=floor_translation(host,guest,R,target)
        self.assertIsNotNone(t)
        moved=guest['pressure'][0]@R.T+t
        for n,o in zip(host['floor_normals'],host['floor_offsets']):
            self.assertGreaterEqual((moved@n-o).min(),2e-4-1e-9)
        for n,o in zip(guest['floor_normals']@R.T,guest['floor_offsets']+(guest['floor_normals']@R.T)@t):
            self.assertGreaterEqual((host['pressure'][0]@n-o).min(),2e-4-1e-9)

    def test_shared_parallel_floor_has_no_margin_but_tilted_floors_do(self):
        # Exactly parallel floors may be shared (t_z = 0), as in floor_repair.
        host=family([[0,0,1]],[0.],[[0,0,0],[1,0,0],[0,1,0]])
        guest=family([[0,0,1]],[0.],[[0,0,0],[1,0,0],[0,1,0]])
        t=floor_translation(host,guest,np.eye(3),np.array([.3,0,.2]))
        self.assertAlmostEqual(t[2],0.,places=12)
        # Opposed floors are distinct: each hull clears the other by 0.2 mm.
        flip=Rotation.from_rotvec([np.pi,0,0]).as_matrix()
        t=floor_translation(host,guest,flip,np.zeros(3))
        self.assertGreaterEqual(t[2],2e-4-1e-12)

    def test_merged_layout_moves_only_guest_installations(self):
        host=family([[0,0,1]],[0.],[[0,0,0]]);guest=family([[0,1,1]],[.1],[[0,0,0]])
        C=np.eye(4);C[:3,:3]=Rotation.from_rotvec([0,.3,0]).as_matrix();C[:3,3]=[.1,.2,.3]
        q=merged_layout(host,guest,C)
        np.testing.assert_array_equal(q.material[0],host['layout'].material[0])
        np.testing.assert_allclose(q.material[1],C@guest['layout'].material[0])
        np.testing.assert_array_equal(q.poses,np.concatenate([host['layout'].poses,guest['layout'].poses]))


if __name__=='__main__':
    unittest.main()
