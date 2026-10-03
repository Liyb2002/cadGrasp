import unittest
from types import SimpleNamespace
import numpy as np
from collect_states import conditional,joint_value,sample_states
from prepare import compatible

def pool(n):return [dict(valid=True,directions=[i],components=[0]) for i in range(n)]
class Tests(unittest.TestCase):
 def test_two_pose_exact_and_state_count(self):
  p=pool(2);r=joint_value([p,p],[[(0,),(1,)],[(0,),(1,)]],{(0,1):np.array([[1.,0.],[0.,1.]])},0)
  self.assertEqual(r['value'],1);self.assertEqual(r['remaining_heads'],1);self.assertEqual(r['dispersion'],0)
 def test_three_pose_average(self):
  p=pool(1);r=joint_value([p]*3,[[(0,)]]*3,{(0,1):np.array([[.3]]),(0,2):np.array([[.6]]),(1,2):np.array([[.9]])},1)
  self.assertAlmostEqual(r['dispersion'],.6);self.assertAlmostEqual(r['value'],1.6)
 def test_conditional_preserves_all_state_heads(self):
  p=[dict(valid=True,directions=[0],components=[0]) for _ in range(4)]
  oracle=SimpleNamespace(check=lambda g:len(g)>=3)
  bank={(0,1,2)}
  found=conditional(p,oracle,{0,3},bank,np.random.default_rng(1),8)
  self.assertTrue(found)
  self.assertTrue(all({0,3}.issubset(g) for g in found))
 def test_states_can_be_sampled_without_successful_seeds(self):
  p=[dict(valid=True,directions=[0],components=[0]) for _ in range(12)]
  states=sample_states([p,p],[set(),set()],20,42)
  self.assertEqual(len(states),20);self.assertEqual(len(set(states)),20)
  self.assertEqual(states[0],((),()))
  self.assertTrue(all(compatible(p,list(g)) for state in states for g in state))
 def test_certified_completions_are_reused_without_resolving(self):
  p=[dict(valid=True,directions=[0],components=[0]) for _ in range(3)]
  oracle=SimpleNamespace(check=lambda g:self.fail('unnecessary solve'))
  self.assertEqual(conditional(p,oracle,{0,1},{(0,1,2)},np.random.default_rng(1)),[(0,1,2)])
 def test_dead_path_never_runs_terminal(self):
  oracle=SimpleNamespace(check=lambda g:self.fail('dead branch checked'))
  self.assertEqual(conditional(pool(2),oracle,{0,1},{(0,)},np.random.default_rng(1)),[])
if __name__=='__main__':unittest.main()
