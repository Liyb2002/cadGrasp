"""Delta/full parity, shared blockers, movement effects and certificate reuse."""
import unittest
from types import SimpleNamespace
from common import *
from model import Layout
from delta_guidance import ContactDelta, VolumeDelta
from volume_guidance import VolumeGuidance
from classify import classify
from model import Model


def layout(n):
    return Layout(np.repeat(np.eye(4)[None],n,axis=0),
                  np.repeat([[0.,0.,1.]],n,axis=0),np.arange(n),tuple(range(n)))


class FakeContacts:
    def __init__(self):
        self.points=np.zeros((8,3));self.sources=np.arange(8)
        self.allowed=[np.arange(8)]*3
        self.calls=0

    def material_at_points(self, current, owner, provider):
        index=np.arange(8)
        return (index+owner+provider+int(current.placements[owner,0,3]*10)) % 3 != 0

    def contact_allowed(self,current,owner):
        return np.ones(len(self.points),bool)

    def locks(self,current,owner,blocker):
        self.calls+=1
        index=np.arange(8)
        shift=int(current.placements[owner,0,3]*10)-int(current.placements[blocker,0,3]*10)
        return (index+shift+int(current.directions[blocker,0]*10)) % (blocker+2) == 0

    def rebuild(self,current):
        result={}
        for k in current.active:
            coverage=np.logical_or.reduce([self.material_at_points(current,k,j) for j in current.active])
            locked=np.logical_or.reduce([self.locks(current,k,j) for j in current.active])
            result[k]=coverage & ~locked
        return result


class DeltaTests(unittest.TestCase):
    def test_nearly_tangent_leading_faces_cannot_supply_a_reaction(self):
        model=Model.__new__(Model)
        model.points=np.zeros((2,3));model.sources=np.arange(2)
        model.allowed=[np.arange(2)]
        model.mesh=SimpleNamespace(face_normals=np.array([[1.,0.,0.],[-1.,0.,0.]]))
        model.material_at_points=lambda *args:np.ones(2,bool)
        model.locks=lambda *args:np.zeros(2,bool)
        current=layout(1);current.directions[0]=[1e-6,0.,1.]
        flags=ContactDelta(model).state(current).available[0]
        np.testing.assert_array_equal(flags,[False,True])

    def test_direction_updates_one_blocker_column_and_preserves_other_locks(self):
        model=FakeContacts();engine=ContactDelta(model);original=layout(3)
        base=engine.state(original)
        old={k:v.copy() for k,v in base.lock_counts.items()}
        trial=original.copy();trial.directions[1]=[.2,0.,.98]
        before=model.calls
        changed=engine.state(trial)
        self.assertEqual(model.calls-before,3)
        full=model.rebuild(trial)
        for k in trial.active:
            np.testing.assert_array_equal(changed.available[k],full[k])
            np.testing.assert_array_equal(base.lock_counts[k],old[k])

    def test_translation_changes_owner_row_and_blocker_column_without_mutation(self):
        model=FakeContacts();engine=ContactDelta(model);original=layout(3)
        engine.state(original)
        trial=original.copy();trial.placements[1,0,3]=.1
        changed=engine.state(trial);full=model.rebuild(trial)
        for k in trial.active:
            np.testing.assert_array_equal(changed.available[k],full[k])
        engine.commit(trial)
        trial2=trial.copy();trial2.directions[2]=[.3,0.,.95]
        changed=engine.state(trial2);full=model.rebuild(trial2)
        for k in trial2.active:
            np.testing.assert_array_equal(changed.available[k],full[k])

    def test_incremental_insertion_and_removal_match_a_full_rebuild(self):
        model=FakeContacts();engine=ContactDelta(model);one=layout(3);one.active=(0,)
        engine.commit(one)
        two=one.copy();two.active=(0,1)
        changed=engine.state(two);full=model.rebuild(two)
        for k in two.active:
            np.testing.assert_array_equal(changed.available[k],full[k])
        engine.commit(two)
        changed=engine.state(one);full=model.rebuild(one)
        np.testing.assert_array_equal(changed.available[0],full[0])

    def test_volume_delta_matches_full_sampling_and_counts_shared_material_once(self):
        body=C.trimesh.creation.box(extents=[.04]*3)
        outer=C.trimesh.creation.box(extents=[.05]*3)
        wrap=unpack_solid(C.S.solid(outer)-C.S.solid(body))
        work=C.trimesh.creation.box(extents=[.005]*3);work.apply_translation([.1]*3)
        model=SimpleNamespace(mesh=body,thickness=.005,length=.2,ray=RayMeshIntersector(body),
                              wrap_rays=[RayMeshIntersector(wrap)]*2,work_rays=[RayMeshIntersector(work)]*2)
        base=layout(2)
        moved=base.copy();moved.placements[1,0,3]=.012
        turned=moved.copy();turned.directions[1]=np.array([.2,0.,1.])/np.linalg.norm([.2,0.,1.])
        layouts=[base,moved,turned]
        delta=VolumeDelta(model,layouts,power=12);full=VolumeGuidance(model,layouts,power=12)
        initial=delta.estimate(base);delta.commit(base)
        for trial in layouts:
            self.assertAlmostEqual(delta.estimate(trial),full.estimate(trial),places=12)
            change=delta.delta(trial)
            self.assertAlmostEqual(change['net_cm3'],delta.estimate(trial)-initial,places=12)
        old=delta.base.material.copy();delta.state(turned)
        np.testing.assert_array_equal(delta.base.material,old)
        delta.commit(moved)
        self.assertAlmostEqual(delta.estimate(turned),full.estimate(turned),places=12)

    def test_reaction_basis_reuse_rejects_removed_contacts(self):
        full=np.eye(7)[:2];targets=np.eye(6)[:2]
        bank=[]
        mask,info=classify(full,targets,basis_cache=bank,column_ids=np.array([10,11]))
        self.assertTrue(mask.all());self.assertTrue(bank)
        again,info=classify(full,targets,basis_cache=bank,column_ids=np.array([10,11]))
        self.assertTrue(again.all());self.assertEqual(info['equilibrium_lps'],0)
        removed,info=classify(full[:1],targets,basis_cache=bank,column_ids=np.array([10]))
        np.testing.assert_array_equal(removed,[True,False])


if __name__=='__main__':
    unittest.main()
