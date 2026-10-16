import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from whole_task.step1.common import ROOT
from .poses import load_inputs, formalize
from .transforms import rigid_transform
from .wrenches import build_domain, sample_domain, endpoint_demands, demand


class WrenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, source, arrays, config, _ = load_inputs()
        cls.poses, _ = formalize(cls.mesh, source, arrays, config)
        cls.physical = json.loads((ROOT/"objects/B/meta.json").read_text())
        cls.policy = json.loads(Path(__file__).with_name("load_policy.json").read_text())
        cls.domain = cls.make_domain(cls.poses["poses"][0])

    @classmethod
    def make_domain(cls, pose):
        return build_domain(cls.mesh, pose, cls.poses["fixed_station"], cls.physical, cls.policy)

    def test_known_support_reaction_sign_and_moment_pair(self):
        np.testing.assert_array_equal(demand([0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 0, -1]),
                                      [-1, 0, 1, 0, 0, 2])

    def test_translation_keeps_com_moment_but_moves_world_origin_moment(self):
        first = self.domain.evaluate(0, .2, .3, .4, check_visibility=False)
        pose = copy.deepcopy(self.poses["poses"][0])
        transform = np.asarray(pose["world_from_object"])
        shift = np.array([.3, -.1, .2])
        transform[:3, 3] += shift
        pose["world_from_object"] = transform.tolist()
        moved = self.make_domain(pose)
        second = moved.evaluate(0, .2, .3, .4, check_visibility=False)
        np.testing.assert_allclose(first["need_wrench"], second["need_wrench"], atol=1e-14)
        np.testing.assert_allclose(second["pt_m"]-first["pt_m"], shift, atol=1e-14)
        origin1 = first["need_wrench"][3:]+np.cross(self.domain.com, first["need_wrench"][:3])
        origin2 = second["need_wrench"][3:]+np.cross(moved.com, second["need_wrench"][:3])
        np.testing.assert_allclose(origin2-origin1, np.cross(shift, first["need_wrench"][:3]), atol=1e-14)

    def test_new_rotation_recomputes_geometry_while_gravity_and_needle_stay_world_fixed(self):
        pose = copy.deepcopy(self.poses["poses"][0])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        transform = rigid_transform(rotation, [.1, -.4, .25])
        pose["world_from_object"] = transform.tolist()
        changed = self.make_domain(pose)
        np.testing.assert_array_equal(changed.gravity, self.domain.gravity)
        np.testing.assert_array_equal(changed.axis, self.domain.axis)
        sample = changed.evaluate(0, .2, .3, .4, check_visibility=False)
        local_tri = self.mesh.triangles[changed.work_ids[0]]
        local_point = .5*local_tri[0]+.2*local_tri[1]+.3*local_tri[2]
        expected_arm = rotation @ (local_point-self.physical["com_mesh_frame"])
        np.testing.assert_allclose(sample["need_wrench"][3:], -np.cross(expected_arm, .4*self.domain.axis), atol=1e-14)

    def test_compiled_formula_matches_original_quadratic_implementation(self):
        sys.path.insert(0, str(ROOT/"objects"))
        from r6_quadratic import QuadraticR6
        tri = self.domain.triangles
        legacy = QuadraticR6(SimpleNamespace(a=tri[:, 0]-self.domain.com,
            e=tri[:, 1]-tri[:, 0], f=tri[:, 2]-tri[:, 0], gravity=self.domain.gravity))
        np.testing.assert_array_equal(legacy.matrices, self.domain.B)
        force = .3*self.domain.axis
        np.testing.assert_allclose(legacy.cartesian(0, .2, .3, force),
            self.domain.evaluate(0, .2, .3, .3, check_visibility=False)["need_wrench"], atol=1e-14)

    def test_exact_jacobian_and_closed_magnitude_endpoints(self):
        at = np.array([.2, .3, .25]); step = 1e-6
        jac = self.domain.jacobian(0, *at)
        for k in range(3):
            delta = np.eye(3)[k]*step
            plus = self.domain.evaluate(0, *(at+delta), check_visibility=False)["need_wrench"]
            minus = self.domain.evaluate(0, *(at-delta), check_visibility=False)["need_wrench"]
            np.testing.assert_allclose(jac[:, k], (plus-minus)/(2*step), atol=1e-10, rtol=0)
        zero = self.domain.evaluate(0, 0, 0, 0, check_visibility=False)["need_wrench"]
        np.testing.assert_array_equal(zero, np.r_[-self.domain.gravity, np.zeros(3)])
        self.domain.evaluate(0, 1, 0, self.domain.k, check_visibility=False)
        with self.assertRaises(ValueError):
            self.domain.evaluate(0, .8, .8, .2)
        with self.assertRaises(ValueError):
            self.domain.evaluate(0, .1, .1, self.domain.k+.001)

    def test_sample_prefix_is_reproducible_and_si_scale_preserves_pair(self):
        small = sample_domain(self.domain, 64, 12)
        large = sample_domain(self.domain, 128, 12)
        for key in ("parameters", "work_face_index", "pt_m", "need_wrench"):
            np.testing.assert_array_equal(small[key], large[key][:64])
        np.testing.assert_allclose(small["need_wrench_SI"], small["need_wrench"]*self.physical["mass_kg"]*9.81, atol=1e-14)
        np.testing.assert_array_equal(small["targets_7d"][:, 6], np.zeros(64))

    def test_full_filled_region_is_kept_and_flagged(self):
        pose = self.poses["poses"][1]
        domain = self.make_domain(pose)
        np.testing.assert_array_equal(domain.work_ids, pose["working_face_ids"])
        self.assertEqual(int((~domain.face_screened).sum()), len(pose["geometry_audit"]["unresolved_face_ids"]))
        self.assertGreater(int((~domain.face_screened).sum()), 0)
        face = np.flatnonzero(~domain.face_screened)[0]
        result = domain.evaluate(face, .2, .3, .3)
        self.assertFalse(result["geometrically_screened"])
        self.assertEqual(result["need_wrench"].shape, (6,))

    def test_vertex_endpoints_span_every_surface_magnitude(self):
        domain = self.domain
        endpoints = endpoint_demands(domain)
        lookup = {v: i for i, v in enumerate(endpoints["source_vertex_ids"])}
        for face in (0, len(domain.work_ids)-1):
            bary = np.array([.2, .3, .5]); magnitude = .17
            vertices = domain.mesh.faces[domain.work_ids[face]]
            rows = [lookup[v] for v in vertices]
            wanted = (1-magnitude/domain.k)*endpoints["endpoint_need_wrench"][0]
            wanted += magnitude/domain.k*(bary @ endpoints["endpoint_need_wrench"][rows])
            result = domain.evaluate(face, bary[1], bary[2], magnitude, check_visibility=False)
            np.testing.assert_allclose(result["need_wrench"], wanted, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
