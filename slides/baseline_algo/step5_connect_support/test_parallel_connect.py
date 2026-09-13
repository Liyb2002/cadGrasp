"""Process isolation and publishing completed supports ahead of a slow sibling."""
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
from threading import Event
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import parallel_connect as P
from step5_connect_support import test_fixed_feet as fixtures, solids as S


def good_payload(cid):
    mesh,contact,direction,foot,work=fixtures.FixedFeetTests().setup_case()
    contact={**contact,'candidate_id':cid};foot={**foot,'candidate_id':cid}
    record,module=P.B.bounded_one(mesh,contact,direction,foot,.01,work,max_edges=500)
    return record,P.pack(module)


def process_case(index):
    if index==0:raise ValueError('One worker cannot construct this support')
    return good_payload('good')


class ParallelConnectionTests(unittest.TestCase):
    def test_spawned_worker_error_retains_sibling_real_solid(self):
        contacts=[dict(candidate_id=cid) for cid in ('bad','good')];seen=[]
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(process_case,i):i for i in range(2)}
            records,modules=P.collect(contacts,futures,lambda r,m:seen.append(r['candidate_id']))
        self.assertEqual([r['candidate_id'] for r in records],['bad','good'])
        self.assertEqual(records[0]['status'],'individual_worker_error')
        self.assertFalse(records[0]['failure_is_global_impossibility_proof'])
        self.assertTrue(records[1]['passed'])
        self.assertEqual(set(seen),{'bad','good'})
        self.assertEqual(len(modules),1);module=modules[0]
        original=good_payload('good')[1]['arrays']
        replay=S.pack_parts(module['parts'],module['labels'],module['joined'])
        for key in original:np.testing.assert_array_equal(original[key],replay[key])

    def test_fast_support_is_published_before_slow_earlier_contact(self):
        fast=good_payload('fast');released=Event();seen=[]
        def slow():
            released.wait(timeout=5)
            return dict(candidate_id='slow',passed=False,status='search_unresolved'),None
        def publish(record,module):
            seen.append(record['candidate_id'])
            if record['candidate_id']=='fast':released.set()
        contacts=[dict(candidate_id=cid) for cid in ('slow','fast')]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(slow):0,pool.submit(lambda:fast):1}
            records,modules=P.collect(contacts,futures,publish)
        self.assertEqual(seen,['fast','slow'])
        self.assertEqual([r['candidate_id'] for r in records],['slow','fast'])
        self.assertEqual(modules[0]['plan']['candidate_id'],'fast')

    def test_worker_rejects_a_different_input_snapshot(self):
        inputs=(None,[],None,{}, {},.01,[])
        with patch.dict(os.environ),patch.object(P.C,'read_inputs',return_value=inputs), \
                patch.object(P.C.I,'hashes',return_value={'changed':'hash'}):
            with self.assertRaisesRegex(RuntimeError,'parent snapshot'):
                P.initialize('test','pose_1',200,{'original':'hash'})


if __name__=='__main__':unittest.main()
