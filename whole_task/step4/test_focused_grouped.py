"""Large work-pose rolls must not silently rotate independent withdrawal axes."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from whole_task.step1.common import ROOT
from .angular_model import AngularTaskModel,AngularLayout
from .focused_grouped import spin_with_world_exit
from fixture_state import installed_state


class IndependentExitTests(unittest.TestCase):
    def test_large_guest_roll_preserves_all_world_exits_and_other_world_tasks(self):
        m=AngularTaskModel()
        with np.load(ROOT/'whole_task/data/continuous13/B/step4/grouped13_partial_checkpoint.npz') as a:
            q=AngularLayout.from_arrays(a)
        m.activate(q)
        worlds={k:installed_state(m.native,q,int(q.hosts[k]))@q.placements[k] for k in q.active}
        exits={k:installed_state(m.native,q,int(q.hosts[k]))[:3,:3]@q.directions[k] for k in q.active}
        owner=12
        for delta in [180.,90.,-90.]:
            trial=spin_with_world_exit(m,q,owner,q.angles_deg[owner]+delta)
            for k in q.active:
                t=installed_state(m.native,trial,int(trial.hosts[k]));u=t@trial.placements[k]
                np.testing.assert_allclose(t[:3,:3]@trial.directions[k],exits[k],atol=1e-12)
                if k!=owner:np.testing.assert_allclose(u,worlds[k],atol=1e-12)
                else:
                    expected=Rotation.from_rotvec(np.radians(delta)*m.rotation_axis).as_matrix()@worlds[k][:3,:3]
                    np.testing.assert_allclose(u[:3,:3],expected,atol=1e-12)
            m.check_station(trial)


if __name__=='__main__':unittest.main()
