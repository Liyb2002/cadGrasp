"""step4/fast_contacts: the kernel's contact region and the same judgement of every paired demand."""
import unittest
import numpy as np
import trimesh

from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import contacts
from whole_task_2.step3.legacy import kernel as K
from .combine import body_and_keepout, juxtaposed, largest, membrane, pose, slot, default_exit
from .merge import move
from .fast_contacts import contacts as fast
from .seats import context

OBJECT = 'B'


class FastContactTests(unittest.TestCase):
    def check(self, support, slots):
        c = context(OBJECT)
        for s in slots:
            i = s['members'][0]; P = pose(OBJECT, i); data = c.base.world_data(c.base.records[i], P)
            body, domain = data['body_world'], data['domain']
            near = contact_surface(body, moved_mesh(support, P@np.linalg.inv(np.asarray(s['X']))))
            (a, _, A, scale), (b, _, B, _) = contacts(body, near, domain.work_ids, domain.com), fast(body, near, domain.work_ids, domain.com)
            self.assertAlmostEqual(trimesh.triangles.area(a).sum()/trimesh.triangles.area(b).sum(), 1., places=12)
            targets = np.vstack([data['endpoints']['endpoint_need_wrench'], data['samples']['need_wrench']])*scale
            np.testing.assert_array_equal(K.J.classify(A, targets)[0], K.J.classify(B, targets)[0])

    def test_a_task_alone(self):
        i = 0; keep, _ = largest(membrane(OBJECT, i)-body_and_keepout(OBJECT, i))
        self.check(G.mesh(keep), [slot(i, np.eye(4), default_exit(OBJECT, i))])

    def test_a_task_cut_by_another_tasks_keep_out(self):
        g = dict(slots=[slot(0, np.eye(4), default_exit(OBJECT, 0))]); X = juxtaposed(OBJECT, g, 0, 3, 60., np.zeros(3))
        keep, _ = largest(membrane(OBJECT, 0)-body_and_keepout(OBJECT, 0)-move(X, body_and_keepout(OBJECT, 3)))
        self.check(G.mesh(keep), g['slots'])


if __name__ == '__main__':
    unittest.main()
