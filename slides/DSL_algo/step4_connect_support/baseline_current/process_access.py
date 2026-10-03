"""Step4 working-surface guard, matching Step3's working-face exclusions.

The compatibility filename is retained for callers from the previous revision.
Tool rays/cones and shared non-working/working edges are NOT excluded volumes.
"""
from pathlib import Path

import numpy as np
import trimesh

from step3_scheculer import contacts as I
from step2_local_support import support_policy as POLICY
from step4_connect_support.baseline_current import build_coupled_saddle as S, working_surface as SURFACE

SCHEMA = 'whole_fixture_working_surface_v2'


class AccessRejected(RuntimeError):
    def __init__(self, report):
        self.access_report = report
        super().__init__(report['status'])


def sources():
    return [Path(__file__),Path(SURFACE.__file__),Path(POLICY.__file__),
        Path(SURFACE.triangle_overlaps.__code__.co_filename)]


def contact_checks(tasks, groups, bases=None, offsets=None, owner_only=False):
    if bases is None:
        bases = np.repeat(np.eye(3)[None],len(tasks),axis=0)
        offsets = np.zeros((len(tasks),3))
    rows = []
    for k,task in enumerate(tasks):
        for owner,contacts in enumerate(groups):
            if owner_only and owner != k:
                continue
            for contact in contacts:
                if owner == k:
                    # Exactly the native Step3 mask; adjacent non-working faces
                    # are eligible even when they meet a working-face edge.
                    overlap = np.intersect1d(contact['source_faces'],task.domain.work_ids)
                    check = dict(passed=not overlap.size,overlapping_work_faces=overlap.tolist(),
                        method='Step3 source_faces intersection with owner work_ids')
                else:
                    triangles = (contact['triangles_m']@bases[owner]+offsets[owner]-offsets[k])@bases[k].T
                    patch = trimesh.Trimesh(triangles.reshape(-1,3),
                        np.arange(triangles.size//3).reshape(-1,3),process=False)
                    check = SURFACE.check(patch,task,closed_solid=False)
                rows.append(dict(pose=task.pose,owner_pose=tasks[owner].pose,
                    candidate_id=contact['candidate_id'],owner_contact=owner==k,
                    passed=bool(check['passed']),check=check))
    return rows


def preflight(case):
    rows = contact_checks(case.tasks,case.groups,owner_only=True)
    passed = all(r['passed'] for r in rows)
    report = dict(schema=SCHEMA,complete=True,object=case.name,poses=case.poses,passed=passed,
        status='owner_working_faces_clear' if passed else 'owner_contact_on_working_face',checks=rows,
        same_surface_rule_as_step3=True,shared_edge_or_vertex_contact_allowed=True,
        processing_ray_volume_enforced=False,head_or_load_inputs_changed=False,
        supersedes_previous_strict_access_cone_rejection=True,
        previous_claim_that_heads_must_be_reselected_withdrawn=True,
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes(sources())))
    I.save(case.output/'data/process_access_preflight.json',report)
    if not passed:
        raise AccessRejected(report)
    return report


class Guard:
    def __init__(self,case,placement):
        self.case = case
        self.bases,self.offsets = np.asarray(placement['bases']),np.asarray(placement['offsets'])
        preflight(case)
        self.contact_checks = contact_checks(case.tasks,case.groups,self.bases,self.offsets)
        if not all(r['passed'] for r in self.contact_checks):
            raise AccessRejected(dict(schema=SCHEMA,complete=True,passed=False,
                status='installed_contact_on_working_face',checks=self.contact_checks))
        self.root_checks = []
        for k,task in enumerate(case.tasks):
            for owner,root in enumerate(case.root_solids):
                mesh = S.unpack(root)
                mesh.vertices = (mesh.vertices@self.bases[owner]+self.offsets[owner]-self.offsets[k])@self.bases[k].T
                self.root_checks.append(dict(pose=task.pose,owner_pose=case.poses[owner],
                    **SURFACE.check(mesh,task)))
        if not all(r['passed'] for r in self.root_checks):
            raise AccessRejected(dict(schema=SCHEMA,complete=True,passed=False,
                status='generated_root_on_working_face',checks=self.root_checks,
                roots_are_constructor_owned=True,root_reshape_may_resolve=True))

    def verify(self,mesh):
        rows = []
        for task,basis,offset in zip(self.case.tasks,self.bases,self.offsets):
            installed = trimesh.Trimesh((mesh.vertices-offset)@basis.T,mesh.faces,process=False)
            rows.append(dict(pose=task.pose,**SURFACE.check(installed,task)))
        return dict(schema=SCHEMA,enforced=True,passed=all(r['passed'] for r in rows),checks=rows,
            working_surface_enforced=True,processing_ray_volume_enforced=False,
            shared_edge_or_vertex_contact_allowed=True,same_surface_rule_as_step3=True,
            support_scope='All actual installed material, including active/idle heads, roots, feet and connectors',
            contact_checks=self.contact_checks,root_checks=self.root_checks)
