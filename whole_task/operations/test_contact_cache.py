import unittest
from unittest.mock import Mock
import numpy as np
from whole_task.step3 import independent_geometry as G
from whole_task.step3 import independent_physics as P
from whole_task.step3.legacy import kernel as K
from .contact_cache import query_contacts,cached_certificate_contacts
import trimesh


class ContactCacheTests(unittest.TestCase):
    def test_reordered_same_boundary_preserves_physics_without_requerying(self):
        body=G.mesh(G.box([-.1,-.1,.2],[.1,.1,.4]))
        support=G.mesh(G.box([-1.,-1.,0.],[1.,1.,.2]))
        permuted=trimesh.Trimesh(support.vertices,support.faces[::-1][:,[1,2,0]],process=False)
        excluded=np.array([],int);cache={};primitive=Mock(wraps=P.contacts)
        a=query_contacts(body,support,excluded,body.center_mass,cache,primitive)
        b=query_contacts(body,permuted,excluded,body.center_mass,cache,primitive)
        self.assertEqual(primitive.call_count,1)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        fresh=P.contacts(body,permuted,excluded,body.center_mass)
        targets=np.zeros((10,6));targets[:,2]=1.;targets[:,3]=np.linspace(-.02,.02,10)
        np.testing.assert_array_equal(K.J.classify(b[2],targets*b[3])[0],K.J.classify(fresh[2],targets*fresh[3])[0])
        bottom=np.flatnonzero(body.face_normals[:,2]<-.9)
        with self.assertRaises(RuntimeError):query_contacts(body,support,bottom,body.center_mass,cache,primitive)
        self.assertEqual(primitive.call_count,2)

    def test_certificate_query_wrapper_is_restored_after_failure(self):
        original=P.contacts
        try:
            with cached_certificate_contacts({}):
                self.assertIsNot(P.contacts,original)
                raise ValueError('test')
        except ValueError:pass
        self.assertIs(P.contacts,original)


if __name__=='__main__':unittest.main()
