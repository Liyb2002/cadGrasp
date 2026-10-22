"""Continuous nonconvex sweep and signed-cavity regression checks."""
import unittest
import numpy as np
from . import independent_geometry as G
from .exit_geometry import whole_object_sweep,grounded_material,continuous_checks
from .exit_search import independent_search,AssemblyRejected


class ExitInitializationTests(unittest.TestCase):
    def test_nonconvex_sweep_keeps_the_reentrant_gap(self):
        boxes=[G.box([0,0,.01],[.06,.02,.05]),G.box([0,0,.01],[.02,.06,.05])]
        body=G.mesh(G.union(boxes));direction=np.array([0.,0.,1.])
        exact,meta=whole_object_sweep(body,direction,.08)
        reference=G.union([G.whole_object_sweep(G.mesh(b),direction,.08)[0]for b in boxes])
        self.assertLess(G.volume(exact-reference),1e-12)
        self.assertLess(G.volume(reference-exact),1e-12)
        gap=G.box([.027,.027,.03],[.033,.033,.06])
        convex=G.whole_object_sweep(body,direction,.08)[0]
        self.assertLess(G.volume(exact^gap),1e-12)
        self.assertGreater(G.volume(convex^gap),1e-8)
        self.assertTrue(meta['all_positive_leading_prisms_retained'])

    def test_ground_selection_preserves_internal_void_without_orphan_cavity(self):
        base=G.box([-.1,-.1,0],[.1,.1,.1]);void=G.box([-.03,-.03,.02],[.03,.03,.08])
        floating=G.box([.3,.3,.2],[.4,.4,.3]);floating_void=G.box([.32,.32,.22],[.38,.38,.28])
        original=G.union([base-void,floating-floating_void])
        selected=grounded_material(original)
        self.assertAlmostEqual(G.volume(selected),.2*.2*.1-.06**3,places=12)
        self.assertLess(G.volume(selected^void),1e-12)
        self.assertLess(G.volume(selected^floating),1e-12)
        self.assertLess(G.volume(selected-original),1e-12)

    def test_each_path_is_checked_with_its_own_direction(self):
        body=G.mesh(G.box([-.01,-.01,0],[.01,.01,.02]))
        obstacle=G.box([.03,-.01,.03],[.05,.01,.05])
        vertical=np.array([0.,0.,1.]);slanted=np.array([1.,0.,1.])/np.sqrt(2)
        rows,_=continuous_checks(obstacle,[('vertical',body,vertical,.1),('slanted',body,slanted,.1)],1e-10)
        self.assertTrue(rows[0]['passed'])
        self.assertFalse(rows[1]['passed'])

    def test_final_support_failure_only_adjusts_failed_task_and_rechecks_all(self):
        # Both local proposals pass. The first combined fixture fails B;
        # local success must not bypass the final gate or change A's path.
        tried=[];audited=[];events=[]
        def local(id,index,direction):
            tried.append((id,index));return direction
        def assembly(selected):
            audited.append(dict(selected))
            if selected['B']=='back':raise AssemblyRejected(['B'],'lost supporting contact after assembly carve')
            return 'fully_audited_support'
        selected,result,indices=independent_search(['A','B'],{'A':['back','tilted_A'],'B':['back','tilted_B']},local,assembly,events.append)
        self.assertEqual(tried,[('A',0),('B',0),('B',1)])
        self.assertEqual(audited,[{'A':'back','B':'back'},{'A':'back','B':'tilted_B'}])
        self.assertEqual(selected,{'A':'back','B':'tilted_B'})
        self.assertEqual(result,'fully_audited_support');self.assertEqual(indices,{'A':0,'B':1})

    def test_exhausted_final_audit_does_not_accept_local_success(self):
        events=[]
        def reject(selected):raise AssemblyRejected(['A'],'force demand infeasible')
        with self.assertRaises(RuntimeError):
            independent_search(['A'],{'A':['back']},lambda *args:'local_pass',reject,events.append)
        self.assertFalse(any(e['status']=='accepted'for e in events))

    def test_local_load_failure_advances_only_that_task(self):
        attempts=[];events=[]
        def local(id,index,direction):
            attempts.append((id,index))
            if id=='B'and index==0:raise RuntimeError('paired moment cannot be supported after this cut')
            return direction
        selected,_,_=independent_search(['A','B'],{'A':['back'],'B':['back','slightly_tilted']},
            local,lambda paths:dict(paths),events.append)
        self.assertEqual(attempts,[('A',0),('B',0),('B',1)])
        self.assertEqual(selected,{'A':'back','B':'slightly_tilted'})
        self.assertEqual(events[-1]['status'],'accepted')


if __name__=='__main__':unittest.main()
