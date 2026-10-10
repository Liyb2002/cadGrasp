"""Forward expressions for the saved continuous demand sets in R^6.

For a working triangle, let a be its first vertex relative to COM, e/f its
edges, n its inward normal, g gravity, and P the applied processing force.

    b(u,v,P) = (-g-P, -(a+u*e+v*f) cross P).

The parameter domain is u,v>=0, u+v<=1, ||P||<=K,
n.P>=cos(alpha)||P||, and the original object-ray visibility predicate.
This is a forward, bilinear map; saved sample rows are NOT inputs to export.

An equivalent expression directly in b=(F,tau) eliminates the surface point:
    P=-F-g, h=n.a, r=(tau cross n+h*P)/(n.P).
For P!=0, require P.tau=0, the force cone, r inside the working triangle,
and visibility of q=COM+r along -P/||P||. The P=0 branch is (-g,0).
Angles below 90 degrees make n.P strictly positive on every nonzero branch.

The sets have intrinsic dimension at most five, NOT positive R^6 volume.
Integration uses the original area/solid-angle/magnitude pushforward measure.
The Jacobians below differentiate load parameters, not fixture exit direction.

Commands (from the repository root):
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python objects/r6_demand.py build
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python objects/r6_demand.py verify
Only new r6_* files are written; existing dataset files are never replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
SCHEMA = "cadgrasp_continuous_r6_demand_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_json(path, data):
    """Create a new artifact; never overwrite even an earlier r6 artifact."""
    if path.exists() and data.get("schema") in {
        "cadgrasp_r6_domain_index_v1", "cadgrasp_r6_expression_verification_v1"
    }:
        previous = json.loads(path.read_text())
        # A repeated run sees our new artifacts too. Keep the count from the
        # first audit while still requiring every substantive field to match.
        if "preexisting_files_preserved" in previous:
            data = dict(data, preexisting_files_preserved=previous["preexisting_files_preserved"])
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Different existing expression: {path}")
        return
    with path.open("x") as stream:
        stream.write(encoded)


def cross_matrix(vectors):
    v = np.asarray(vectors, float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def expression_from_domain(data, source_digest):
    """Use ONLY source geometry/domain constants; no sampled loads or fits."""
    geometry, load = data["geometry"], data["load"]
    vertices = np.asarray(geometry["vertices_m"], float)
    faces = np.asarray(geometry["faces"], int)
    ids = np.asarray(geometry["work_face_ids"], int)
    triangles = vertices[faces[ids]]
    com = np.asarray(data["frame"]["moment_origin_m"], float)
    n = np.asarray(geometry["inward_normals"], float)
    e1, e2 = np.asarray(geometry["tangent1"]), np.asarray(geometry["tangent2"])
    a = triangles[:, 0] - com
    edges = np.stack([triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]], axis=-1)
    # Some saved faces have aspect ratios near 10^6. Normal equations square
    # that condition number; an SVD computes the same left inverse stably.
    barycentric = np.linalg.pinv(edges, rcond=1e-15)
    alpha = float(load["cone_half_deg"])
    if not 0 < alpha < 90 or load["magnitude_range_mg"] != [0., float(load["K"])]:
        raise ValueError("Expected the saved inward cone and [0,K] interval")
    if data["mapping"]["need_force_mg"] != "-gravity_force_mg - F_push_mg":
        raise ValueError("Unexpected saved force convention")
    if data["mapping"]["need_moment_mgm"] != "-cross(q_m-COM_m,F_push_mg)":
        raise ValueError("Unexpected saved moment convention")
    np.testing.assert_allclose(load["gravity_application_point_m"], com, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1., atol=1e-12, rtol=0)
    outward = np.cross(edges[:, :, 0], edges[:, :, 1])
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    # Rotated saved normals and normals recomputed from rounded world vertices
    # differ by up to 3.4e-9 on those very thin faces. Preserve the saved normals
    # for the original force cone, and check their plane orthogonality below.
    np.testing.assert_allclose(outward, -n, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.einsum("fi,fij->fj", n, edges), 0., atol=1e-12, rtol=0)
    for tangent in [e1, e2]:
        np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.sum(tangent*n, axis=1), 0., atol=1e-12, rtol=0)
    return dict(
        schema=SCHEMA, object=data["object"], pose_id=data["pose_id"],
        cone_half_deg=alpha, cone_opening_deg=2*alpha,
        representation="exact_forward_face_union_embedded_in_R6",
        coordinate_frame=data["frame"], units=data["units"],
        ground_state_independent=True, gravity_retained=True,
        pair_order=data["mapping"]["pair_order"], source=dict(file="needs.json", sha256=source_digest),
        construction=dict(method="forward substitution and exact vector-identity elimination",
                          sample_rows_used=0, sample_fitting=False, convex_hull_used=False),
        constants=dict(gravity_force_mg=load["gravity_force_mg"],
                       maximum_processing_force_mg=float(load["K"]),
                       cos_cone_half_angle=float(np.cos(np.radians(alpha)))),
        forward_expression=dict(
            parameters=["u", "v", "P_x", "P_y", "P_z"],
            moment_arm="r=a_f+u*edge_u_f+v*edge_v_f", processing_force="P",
            need_force="F=-g-P", need_moment="tau=-cross(r,P)",
            vector_components=["-g_x-P_x", "-g_y-P_y", "-g_z-P_z",
                               "r_z*P_y-r_y*P_z", "r_x*P_z-r_z*P_x", "r_y*P_x-r_x*P_y"],
            parameter_constraints=["u>=0", "v>=0", "u+v<=1", "norm(P)<=K",
                                   "dot(n_f,P)>=cos(alpha)*norm(P)",
                                   "original object visibility; for P=0 use an admissible direction witness"],
            spherical_force="P=m*(cos(theta)*n_f+sin(theta)*(cos(phi)*tangent1_f+sin(phi)*tangent2_f))",
            spherical_parameters=["u", "v", "theta_rad", "phi_rad", "magnitude_mg"],
            spherical_bounds=dict(u_v="closed unit triangle", theta_rad=[0., float(np.radians(alpha))],
                                  phi_rad="[0,2*pi), periodic", magnitude_mg=[0., float(load["K"])])),
        direct_R6_expression=dict(
            set="D_alpha = {(-g,0)} union UNION_f D_f", processing_force="P=-F-g",
            nonzero_branch_constraints=["0<norm(P)<=K", "dot(n_f,P)>=cos(alpha)*norm(P)",
                                        "dot(P,tau)=0", "u_f(F,tau)>=0", "v_f(F,tau)>=0",
                                        "u_f(F,tau)+v_f(F,tau)<=1", "visible_f(F,tau)"],
            plane_offset="h_f=dot(n_f,a_f)",
            reconstructed_moment_arm="r_f=(cross(tau,n_f)+h_f*P)/dot(n_f,P)",
            reconstructed_barycentrics="[u_f,v_f]=barycentric_inverse_f@(r_f-a_f)",
            reconstructed_application_point="q=COM+r_f",
            visibility="no source-object hit on ray (q-ray_offset_m*n_f)+t*(-P/norm(P)), t>=0",
            zero_branch="F=-g AND tau=0; admissible zero-force witness is recorded below",
            division_domain="nonzero P; denominator is positive since alpha<90 degrees",
            equivalence="tau=P cross r and r lies in face f; force/moment pairing retained"),
        analytic_derivatives=dict(
            Cartesian_column_order=["u", "v", "P_x", "P_y", "P_z"],
            Cartesian_jacobian="J=[[0,0,-I_3],[-cross(edge_u,P),-cross(edge_v,P),-cross_matrix(r)]]",
            spherical_column_order=["u", "v", "theta", "phi", "m"],
            direction_theta="-sin(theta)*n+cos(theta)*(cos(phi)*tangent1+sin(phi)*tangent2)",
            direction_phi="sin(theta)*(-sin(phi)*tangent1+cos(phi)*tangent2)",
            spherical_u="[0_3;-m*cross(edge_u,d)]", spherical_v="[0_3;-m*cross(edge_v,d)]",
            spherical_theta="[-m*d_theta;-m*cross(r,d_theta)]",
            spherical_phi="[-m*d_phi;-m*cross(r,d_phi)]", spherical_m="[-d;-cross(r,d)]",
            scope="analytic load-map derivatives within a fixed working face; not fixture-exit derivatives",
            boundary_scope="visibility, face unions and changing fixture contacts are not globally smooth"),
        measure=dict(
            kind="pushforward of original surface-area, solid-angle and uniform-magnitude measure",
            unnormalized_parameter_density="2*face_area_f*sin(theta)/K, restricted to visible parameters",
            normalized_parameter_density="2*face_area_f*sin(theta)/(A_work*2*pi*(1-cos(alpha))*K*Z_visibility)",
            visibility_normalizer="Z_visibility is the reachable probability under the unconditioned measure",
            image_measure="mu(B)=integral of parameter density over forward_map^{-1}(B)",
            original_definition=data["sampling_measure"],
            ambient_dimension=6, intrinsic_dimension_upper_bound=5, six_dimensional_Lebesgue_volume=0,
            necessary_R6_equality="dot(tau,-F-g)=0",
            note="An exact coupled set in R6, not a six-dimensional filled volume or independent F/tau box"),
        reachability=data["reachability"],
        coefficients=dict(work_face_ids=ids.tolist(), a_minus_com_m=a.tolist(),
                          edge_u_m=edges[:, :, 0].tolist(), edge_v_m=edges[:, :, 1].tolist(),
                          inward_normals=n.tolist(), tangent1=e1.tolist(), tangent2=e2.tolist(),
                          face_plane_offset_m=np.sum(n*a, axis=1).tolist(),
                          barycentric_inverse_per_m=barycentric.tolist(),
                          work_face_areas_m2=geometry["work_face_areas_m2"]))


class R6Demand:
    def __init__(self, expression, geometry=None):
        if expression["schema"] != SCHEMA:
            raise ValueError("Unexpected R6 expression schema")
        self.expression = expression
        co = expression["coefficients"]
        self.a, self.e, self.f = (np.asarray(co[k], float) for k in ["a_minus_com_m", "edge_u_m", "edge_v_m"])
        self.n, self.t1, self.t2 = (np.asarray(co[k], float) for k in ["inward_normals", "tangent1", "tangent2"])
        self.h = np.asarray(co["face_plane_offset_m"])
        self.barycentric = np.asarray(co["barycentric_inverse_per_m"])
        self.com = np.asarray(expression["coordinate_frame"]["moment_origin_m"])
        self.gravity = np.asarray(expression["constants"]["gravity_force_mg"])
        self.maximum = expression["constants"]["maximum_processing_force_mg"]
        self.alpha = np.radians(expression["cone_half_deg"])
        self.geometry, self._mesh = geometry, None

    @classmethod
    def read(cls, path):
        path = Path(path)
        expression = json.loads(path.read_text())
        source = path.parent / expression["source"]["file"]
        if digest(source) != expression["source"]["sha256"]:
            raise ValueError(f"Source domain changed: {source}")
        return cls(expression, json.loads(source.read_text())["geometry"])

    def visible(self, points, directions, inward_normals):
        if self._mesh is None:
            if self.geometry is None:
                raise ValueError("Visibility requires the original complete mesh")
            self._mesh = trimesh.Trimesh(self.geometry["vertices_m"], self.geometry["faces"], process=False)
        points, directions, inward_normals = np.broadcast_arrays(points, directions, inward_normals)
        origins = points - self.expression["reachability"]["ray_offset_m"] * inward_normals
        return (~self._mesh.ray.intersects_any(origins.reshape(-1, 3),
                                             -directions.reshape(-1, 3))).reshape(points.shape[:-1])

    def _parameters(self, face_index, parameters):
        p = np.asarray(parameters, float)
        if p.ndim < 1 or p.shape[-1] != 5 or not np.isfinite(p).all():
            raise ValueError("Expected finite [...,5] parameters (u,v,theta,phi,m)")
        raw = np.asarray(face_index)
        raw = np.broadcast_to(raw, p.shape[:-1])
        if not np.isfinite(raw).all() or np.any(raw != raw.astype(int)):
            raise ValueError("Integer working-face indices required")
        index = raw.astype(int)
        if np.any(index < 0) or np.any(index >= len(self.a)):
            raise ValueError("Working-face index out of bounds")
        u, v, theta, phi, m = np.moveaxis(p, -1, 0)
        if np.any((u < 0) | (v < 0) | (u+v > 1+1e-14) | (theta < 0) |
                  (theta > self.alpha+1e-14) | (m < 0) | (m > self.maximum)):
            raise ValueError("Parameters outside the original domain")
        r = self.a[index]+u[..., None]*self.e[index]+v[..., None]*self.f[index]
        tangent = np.cos(phi)[..., None]*self.t1[index]+np.sin(phi)[..., None]*self.t2[index]
        d = np.cos(theta)[..., None]*self.n[index]+np.sin(theta)[..., None]*tangent
        return index, r, d, tangent, theta, phi, m

    def evaluate(self, face_index, parameters, check_visibility=True):
        index, r, d, _, _, _, m = self._parameters(face_index, parameters)
        force = m[..., None]*d
        points = self.com+r
        need = np.concatenate((-self.gravity-force, -np.cross(r, force)), axis=-1)
        reachable = self.visible(points, d, self.n[index]) if check_visibility else None
        return dict(need_wrench=need, pt_m=points, force_push_mg=force, reachable=reachable)

    def jacobian_cartesian(self, face_index, u, v, processing_force):
        force = np.asarray(processing_force, float)
        if force.ndim < 1 or force.shape[-1] != 3 or not np.isfinite(force).all():
            raise ValueError("Finite processing force [...,3] required")
        index, u, v = np.broadcast_arrays(np.asarray(face_index), u, v)
        shape = np.broadcast_shapes(index.shape, force.shape[:-1])
        raw = np.broadcast_to(index, shape)
        if np.any(raw != raw.astype(int)) or np.any(raw < 0) or np.any(raw >= len(self.a)):
            raise ValueError("Working-face index out of bounds")
        index = raw.astype(int); force = np.broadcast_to(force, shape+(3,))
        u, v = np.broadcast_to(u, shape), np.broadcast_to(v, shape)
        r = self.a[index]+u[..., None]*self.e[index]+v[..., None]*self.f[index]
        result = np.zeros(shape+(6, 5))
        result[..., :3, 2:] = -np.eye(3)
        result[..., 3:, 0] = -np.cross(self.e[index], force)
        result[..., 3:, 1] = -np.cross(self.f[index], force)
        result[..., 3:, 2:] = -cross_matrix(r)
        return result

    def jacobian_spherical(self, face_index, parameters):
        index, r, d, tangent, theta, phi, m = self._parameters(face_index, parameters)
        dt = -np.sin(theta)[..., None]*self.n[index]+np.cos(theta)[..., None]*tangent
        dp = np.sin(theta)[..., None]*(-np.sin(phi)[..., None]*self.t1[index]+
                                      np.cos(phi)[..., None]*self.t2[index])
        result = np.zeros(d.shape[:-1]+(6, 5))
        result[..., 3:, 0] = -m[..., None]*np.cross(self.e[index], d)
        result[..., 3:, 1] = -m[..., None]*np.cross(self.f[index], d)
        for column, derivative in [(2, m[..., None]*dt), (3, m[..., None]*dp), (4, d)]:
            result[..., :3, column] = -derivative
            result[..., 3:, column] = -np.cross(r, derivative)
        return result

    def contains(self, wrench, check_visibility=True, atol=1e-10):
        """Direct R6 membership via exact force/moment/triangle elimination.

        atol is a numerical comparison tolerance, not a thickened demand set.
        The zero-force singular branch is handled without division.
        """
        b = np.asarray(wrench, float)
        if b.ndim < 1 or b.shape[-1] != 6 or not np.isfinite(b).all():
            raise ValueError("Finite [...,6] paired wrench required")
        flat = b.reshape(-1, 6); p = -flat[:, :3]-self.gravity; tau = flat[:, 3:]
        magnitude = np.linalg.norm(p, axis=1)
        zero = magnitude == 0
        result = zero & (np.linalg.norm(tau, axis=1) <= atol)
        if result.any() and "zero_demand_witness" not in self.expression:
            raise ValueError("Zero branch needs an admissible original-domain witness")
        eligible = (~zero) & (magnitude <= self.maximum+1e-12)
        # Relative moment compatibility: do not accept an impossible torsion
        # merely because the processing force is tiny.
        eligible &= np.abs(np.sum(p*tau, axis=1)) <= atol*magnitude
        for face in range(len(self.a)):
            ids = np.flatnonzero(eligible & ~result)
            if not len(ids): break
            denominator = p[ids]@self.n[face]
            keep = denominator >= (np.cos(self.alpha)-1e-12)*magnitude[ids]
            ids, denominator = ids[keep], denominator[keep]
            if not len(ids): continue
            r = (np.cross(tau[ids], self.n[face])+self.h[face]*p[ids])/denominator[:, None]
            uv = (r-self.a[face])@self.barycentric[face].T
            edge_scale = max(np.linalg.norm(self.e[face]), np.linalg.norm(self.f[face]))
            tol = atol/edge_scale
            keep = (uv[:, 0] >= -tol) & (uv[:, 1] >= -tol) & (uv.sum(axis=1) <= 1+tol)
            # Check the recovered physical moment, including numerical plane
            # elimination and the original coupled moment sign.
            keep &= np.max(np.abs(np.cross(p[ids], r)-tau[ids]), axis=1) <= atol
            ids, r = ids[keep], r[keep]
            if check_visibility and len(ids):
                keep = self.visible(self.com+r, p[ids]/magnitude[ids, None], self.n[face])
                ids = ids[keep]
            result[ids] = True
        return result.reshape(b.shape[:-1])


def inventory():
    records = {}
    for p in ROOT.rglob("*"):
        if not (p.is_file() or p.is_symlink()): continue
        stat = p.lstat()
        records[str(p.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns, stat.st_ino,
                                            os.readlink(p) if p.is_symlink() else None]
    return records


def check_preserved(before):
    after = inventory()
    changed = [key for key, value in before.items() if after.get(key) != value]
    if changed:
        raise RuntimeError(f"Preexisting files changed externally or disappeared: {changed[:10]}")
    return len(before)


def source_domains():
    index = json.loads((ROOT/"load_variants.json").read_text())
    if not index["complete"] or index["schema"] != "cadgrasp_all_load_angles_v2":
        raise ValueError("Expected the current complete angle-only dataset")
    paths = []
    for pose in sorted(ROOT.glob("*/poses/pose_*/load_variants.json")):
        manifest = json.loads(pose.read_text())
        if not manifest["complete"]: raise ValueError(f"Incomplete pose: {pose}")
        for row in manifest["variants"]:
            if row["cone_half_deg"] not in index["cone_half_angles_deg"]:
                raise ValueError("Unexpected angle variant")
            folder = pose.parent/row["folder"]
            if digest(folder/"variant.json") != row["variant_sha256"]:
                raise ValueError(f"Stale angle manifest: {folder}")
            record = json.loads((folder/"variant.json").read_text())
            if digest(folder/"needs.json") != record["artifacts"]["needs.json"]:
                raise ValueError(f"Stale source domain: {folder}")
            paths.append(folder/"needs.json")
    if len(paths) != index["variants"]: raise ValueError("Angle-domain inventory incomplete")
    return paths


def build():
    before = inventory(); paths = source_domains(); rows = []; zero_witnesses = {}
    for number, path in enumerate(paths, 1):
        data = json.loads(path.read_text())
        expression = expression_from_domain(data, digest(path))
        domain = R6Demand(expression, data["geometry"])
        pose_key = path.parent.parent
        if pose_key not in zero_witnesses:
            parameters = np.tile([1/3, 1/3, 0., 0., 0.], (len(domain.a), 1))
            result = domain.evaluate(np.arange(len(domain.a)), parameters)
            visible = np.flatnonzero(result["reachable"])
            if not len(visible): raise RuntimeError(f"No normal-ray zero-demand witness: {path}")
            face = int(visible[0])
            zero_witnesses[pose_key] = dict(work_face_index=face, parameters=parameters[face].tolist(),
                                           object_normal_ray_checked=True)
        expression["zero_demand_witness"] = zero_witnesses[pose_key]
        target = path.parent/"r6_domain.json"
        add_json(target, expression)
        rows.append(dict(object=data["object"], pose_id=data["pose_id"], cone_half_deg=expression["cone_half_deg"],
                         expression=str(target.relative_to(ROOT)), sha256=digest(target),
                         source=str(path.relative_to(ROOT)), source_sha256=digest(path),
                         work_face_count=len(domain.a)))
        if number % 90 == 0 or number == len(paths):
            print(f"EXPRESSIONS {number}/{len(paths)}", flush=True)
    preserved = check_preserved(before)
    add_json(ROOT/"r6_domain_index.json", dict(
        schema="cadgrasp_r6_domain_index_v1", complete=True,
        objects=len({r["object"] for r in rows}), poses=len(zero_witnesses), expressions=len(rows),
        cone_half_angles_deg=[15,30,60], sample_rows_used_for_construction=0,
        expression_type="forward bilinear parameterization and equivalent direct R6 membership conditions",
        preexisting_files_preserved=preserved,
        preservation_check="all preexisting paths, sizes, modification timestamps, inodes and symlink targets unchanged",
        generator=dict(file="r6_demand.py", sha256=digest(__file__)), entries=rows))
    print(f"COMPLETE: {len(rows)} expressions; {preserved} preexisting files unchanged", flush=True)


def verify_one(row):
    path = ROOT/row["expression"]
    if digest(path) != row["sha256"]: raise ValueError(f"Expression changed: {path}")
    domain = R6Demand.read(path)
    # Saved rows are read ONLY for independent replay after expression export.
    with np.load(path.parent/"samples.npz", allow_pickle=False) as saved:
        indices, parameters = saved["work_face_index"], saved["parameters"]
        replay = domain.evaluate(indices, parameters, check_visibility=False)
        error = max(float(np.max(np.abs(replay[key]-saved[key])))
                    for key in ["need_wrench", "pt_m", "force_push_mg"])
        if error > 1e-12: raise AssertionError(f"Forward replay mismatch {path}: {error}")
        expected = saved["need_wrench"]
        # Substitute into the ELIMINATED R6 expression on each originating
        # face; this independently tests equality with the forward domain.
        p = -expected[:, :3]-domain.gravity; tau = expected[:, 3:]
        denominator = np.einsum("ij,ij->i", p, domain.n[indices])
        nonzero = np.linalg.norm(p, axis=1) > 1e-14
        i = indices[nonzero]; pp = p[nonzero]; tt = tau[nonzero]
        recovered = (np.cross(tt, domain.n[i])+domain.h[i, None]*pp)/denominator[nonzero, None]
        uv = np.einsum("nij,nj->ni", domain.barycentric[i], recovered-domain.a[i])
        inverse_error = float(np.max(np.abs(uv-parameters[nonzero, :2]), initial=0.))
        # Inverting a total force subtracts gravity; relative roundoff grows
        # for tiny processing forces. Thin triangles amplify position error
        # into barycentric error. Check a condition-aware floating-point bound
        # instead of treating either amplification as a change to the domain.
        reference_r = saved["pt_m"][nonzero]-domain.com
        radius = np.linalg.norm(reference_r, axis=1)
        force_size = np.linalg.norm(pp, axis=1)
        plane_residual = np.abs(np.einsum("ij,ij->i", domain.n[i], reference_r)-domain.h[i])
        arithmetic_scale = (radius+np.linalg.norm(domain.a[i], axis=1)+np.linalg.norm(domain.com)+
                            (np.linalg.norm(tt, axis=1)+np.abs(domain.h[i])*force_size+
                             np.linalg.norm(domain.gravity)*radius)/denominator[nonzero])
        position_bound = (64*np.finfo(float).eps*arithmetic_scale+
                          force_size*plane_residual/denominator[nonzero])
        inverse_bound = 64*np.finfo(float).eps+np.linalg.norm(domain.barycentric[i], axis=(1,2))*position_bound
        inverse_difference = np.max(np.abs(uv-parameters[nonzero, :2]), axis=1)
        normalized_inverse_error = float(np.max(inverse_difference/inverse_bound, initial=0.))
        position_error = float(np.max(np.abs(recovered-reference_r), initial=0.))
        if normalized_inverse_error > 1:
            raise AssertionError(f"Direct R6 substitution exceeds roundoff bound: {path}")
        np.testing.assert_allclose(np.einsum("ij,ij->i", pp, tt), 0., atol=1e-13, rtol=0)
        selection = np.linspace(0, len(expected)-1, 8, dtype=int)
        if not domain.contains(expected[selection], check_visibility=True).all():
            raise AssertionError(f"Original-ray R6 membership mismatch: {path}")
        checked = len(expected)
    # Independent central differences of the forward mapping, not sampled
    # target fitting, verify all five analytic derivative columns.
    ids = np.arange(min(3, len(domain.a)))
    parameters = np.tile([.21, .27, domain.alpha*.43, .73, domain.maximum*.61], (len(ids), 1))
    jac = domain.jacobian_spherical(ids, parameters); derivative_error = 0.
    for column in range(5):
        plus, minus = parameters.copy(), parameters.copy(); step = 1e-6
        plus[:, column] += step; minus[:, column] -= step
        numerical = (domain.evaluate(ids, plus, False)["need_wrench"]-
                     domain.evaluate(ids, minus, False)["need_wrench"])/(2*step)
        difference = float(np.max(np.abs(numerical-jac[:, :, column])))
        derivative_error = max(derivative_error, difference)
    if derivative_error > 2e-9: raise AssertionError(f"Jacobian mismatch: {path}")
    return dict(expression=row["expression"], saved_rows_replayed=checked, passed=True,
                maximum_forward_error=error, maximum_inverse_barycentric_error=inverse_error,
                maximum_inverse_position_error_m=position_error,
                maximum_inverse_error_over_roundoff_bound=normalized_inverse_error,
                maximum_derivative_error=derivative_error, original_ray_membership_checks=8)


def verify(jobs):
    before = inventory(); index = json.loads((ROOT/"r6_domain_index.json").read_text()); rows = []
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for number, result in enumerate(pool.map(verify_one, index["entries"]), 1):
            rows.append(result)
            if number % 90 == 0 or number == len(index["entries"]):
                print(f"VERIFIED {number}/{len(index['entries'])} ({time.monotonic()-began:.1f}s)", flush=True)
    preserved = check_preserved(before)
    report = dict(schema="cadgrasp_r6_expression_verification_v1", complete=True, passed=True,
                  expressions=len(rows), saved_loads_replayed=sum(r["saved_rows_replayed"] for r in rows),
                  sampled_loads_used_only_for_verification=True, sample_fitting_used=False,
                  preexisting_files_preserved=preserved, all_existing_files_unchanged=True,
                  maximum_forward_error=max(r["maximum_forward_error"] for r in rows),
                  maximum_inverse_barycentric_error=max(r["maximum_inverse_barycentric_error"] for r in rows),
                  maximum_inverse_position_error_m=max(r["maximum_inverse_position_error_m"] for r in rows),
                  maximum_inverse_error_over_roundoff_bound=max(r["maximum_inverse_error_over_roundoff_bound"] for r in rows),
                  maximum_derivative_error=max(r["maximum_derivative_error"] for r in rows),
                  scope="forward expressions, eliminated R6 constraints, analytic load Jacobians and original visibility",
                  fixture_optimization_or_acceptance_run=False, entries=rows)
    add_json(ROOT/"r6_domain_verification.json", report)
    print(json.dumps({key:value for key,value in report.items() if key != "entries"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1: parser.error("--jobs must be positive")
    if args.command == "build": build()
    else: verify(args.jobs)


if __name__ == "__main__":
    main()
