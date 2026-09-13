"""Step 3: the contacts needed to hold the workpiece against the process.

The disturbances are now only the pushes the process can make: one per point of
the work region, along the inward normal there. Everything else on the surface is
a place a support may touch -- except the work region itself, which has to stay
clear. Gravity is added as one more wrench that must be balanced.

Frictionless. The push is capped at K times the workpiece's own weight, which is
what lets gravity count for anything: an unbounded push swamps any finite weight,
and then the workpiece being heavy -- the whole premise -- buys nothing. So the
contacts have to balance gravity plus K times any one push, and gravity alone.

    python slides/tools/supports.py A1-f
    python slides/tools/supports.py A1-f --pose 0 --region 3
"""
from __future__ import annotations

import argparse

import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw

import scipy.sparse as sp
from scipy.optimize import linprog

from common import mat_to_quat_wxyz, obj_path, read_json, write_json
from disturbances import _font, sample_surface
from opposition import opposable, wrenches
from work_regions import SCENE, SPRAY_INCIDENCE

CONTACT_EPS = 1.5e-3
# A contact whose push points downward is resting on an upward-facing patch of
# the workpiece, so whatever applies it sits above the workpiece: it is a clamp.
# The threshold is therefore zero, not a tolerance. Allowing even a little slack
# lets two near-horizontal contacts on opposite sides cancel and net a downward
# force, which is the very thing "no clamping" is meant to exclude.
DOWNWARD = 0.0
T_MAX = 1e4            # a multiple this large is unbounded for any real purpose
FREE_TOL = 1e-6        # a wrench component this small counts as absent
# A direction the contacts resist only with forces a million times larger is not
# resisted. Healthy contact sets here sit at 1e-2 of the leading singular value
# and degenerate ones below 1e-6, so the cut lies in open space between them.
POOL = 8               # points drawn per point kept
FELL_BACK = [0]        # how often the direct program had to be bisected instead
ARROW_FRAC = 0.30
ARROW_WIDTH = 2.6e-3
PRESSES_DOWN = (0.88, 0.22, 0.18, 1.0)
PUSHES_UP = (0.20, 0.45, 0.85, 1.0)


def free_motions(W: np.ndarray, tol: float = FREE_TOL):
    """Split wrench space into what these contacts can produce and what they cannot.

    The contacts do not always span all six directions, and the shortfall is
    physical rather than numerical: a frictionless body of revolution is free to
    spin about its axis, and no normal force anywhere on it has a moment about
    that axis. Written as an equality in all six directions the program is
    degenerate, and the solver returns errors on it. Projecting onto what the
    contacts can reach makes it well posed, and the directions left over are
    checked separately -- anything the disturbance or gravity needs there cannot
    be supplied at all.
    """
    U, S, _ = np.linalg.svd(W)
    r = int((S > S[0] * tol).sum()) if len(S) and S[0] > 0 else 0
    return U[:, :r].T, U[:, r:].T


def max_push(W: np.ndarray, w_d: np.ndarray, w_g: np.ndarray, basis=None) -> float:
    """Largest multiple of the weight of this one push the contacts can take.

    Solved directly: the multiple is just one more non-negative unknown alongside
    the contact forces, so one linear program gives the exact answer instead of
    bisecting.

    The multiple is bounded rather than left free. Unbounded is the common case
    here, and HiGHS returns a solve error on some of those instead of certifying
    the ray; read as a failure that becomes "this push cannot be resisted at
    all", and since K* is a minimum over pushes, one such hiccup drags the whole
    number to zero. With the bound in place the program is always well posed and
    hitting the bound is what unbounded looks like.
    """
    P, N = free_motions(W) if basis is None else basis
    if not P.shape[0]:
        return 0.0
    if len(N):                            # nothing can be supplied along these
        if np.abs(N @ w_g).max() > FREE_TOL or np.abs(N @ w_d).max() > FREE_TOL:
            return 0.0
    A = np.hstack([P @ W, (P @ w_d).reshape(-1, 1)])
    c = np.zeros(A.shape[1])
    c[-1] = -1.0
    bounds = [(0, None)] * W.shape[1] + [(0, T_MAX)]
    r = linprog(c, A_eq=A, b_eq=-(P @ w_g), bounds=bounds, method="highs")
    if r.status == 2:               # infeasible: cannot even hold the weight
        return 0.0
    if not r.success:               # a solver failure is not an answer
        FELL_BACK[0] += 1
        return bisect_push(W, w_d, w_g)
    t = float(r.x[-1])
    return np.inf if t >= T_MAX * (1 - 1e-9) else t


def bisect_push(W: np.ndarray, w_d: np.ndarray, w_g: np.ndarray, rounds: int = 30) -> float:
    """`max_push` the slow way, for the handful of programs the fast way fails on.

    The multiples that can be held form an interval: the cone is convex and the
    wrench to be answered is affine in the multiple, so feasibility is monotone
    and can be bisected. Each test is the least-absolute-residual program, which
    always returns something rather than erroring out.
    """
    if not opposable(W, -w_g)[0]:
        return 0.0
    if opposable(W, -(T_MAX * w_d + w_g))[0]:
        return np.inf
    lo, hi = 0.0, T_MAX
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if opposable(W, -(mid * w_d + w_g))[0] else (lo, mid)
    return lo


def sparse_contact_set(W: np.ndarray, targets: list[np.ndarray],
                       rounds: int = 6, tol: float = 1e-7) -> np.ndarray:
    """Smallest set of contacts that can answer every target wrench.

    Every target needs its own force amounts, but they all have to come from one
    set of contacts, so the thing to minimise is how many contacts get used by
    *any* target. That is a count, which is not linear -- it is approximated the
    standard way, by penalising each contact's largest usage and reweighting so
    that contacts already near zero get pushed the rest of the way.

    Removing contacts one at a time from a working set instead gets stuck: a set
    can need two removed together before it shrinks.
    """
    n, K = W.shape[1], len(targets)
    blocks = [sp.hstack([sp.csr_matrix((6, n)),
                         sp.csr_matrix((6, n * k)), sp.csr_matrix(W),
                         sp.csr_matrix((6, n * (K - k - 1)))]) for k in range(K)]
    A_eq = sp.vstack(blocks).tocsc()
    b_eq = np.concatenate(targets)

    eye = sp.identity(n, format="csr")
    A_ub = sp.vstack([sp.hstack([-eye] + [eye if j == k else sp.csr_matrix((n, n))
                                          for j in range(K)]) for k in range(K)]).tocsc()
    b_ub = np.zeros(n * K)

    w = np.ones(n)
    z = np.zeros(n)
    for _ in range(rounds):
        c = np.concatenate([w, np.zeros(n * K)])
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=(0, None), method="highs")
        if not r.success:
            return np.arange(n)
        z = r.x[:n]
        w = 1.0 / (z + 1e-4)

    keep = np.flatnonzero(z > tol * max(z.max(), 1e-12))
    # tighten: drop anything the rest can still cover
    changed = True
    while changed and len(keep) > 1:
        changed = False
        for j in list(keep):
            trial = keep[keep != j]
            if len(trial) and all(opposable(W[:, trial], b)[0] for b in targets):
                keep, changed = trial, True
                break
    return keep


def region_mask(mesh, T, dirs, pts_local, nrm_local) -> np.ndarray:
    """Points of the surface the spray reaches: facing a gun and in its sight."""
    R = T[:3, :3]
    n_world = nrm_local @ R.T
    origins = pts_local + nrm_local * 1e-5
    inside = np.zeros(len(pts_local), bool)
    for u in dirs:
        facing = n_world @ u > np.cos(np.radians(SPRAY_INCIDENCE))
        clear = ~mesh.ray.intersects_any(
            ray_origins=origins, ray_directions=np.tile(u @ R, (len(origins), 1)))
        inside |= facing & clear
    return inside


def sample_sets(mesh, T, dirs, n_dist, n_cand, seed) -> dict:
    """Split the surface into what the process pushes on and what a support may touch.

    Both sets are prefixes of one pool of area-uniform points, so raising either
    count only ever adds points. That matters because K* is a maximum over the
    cone the candidates span and a minimum over the pushes: with independent
    draws per setting the two sets are not nested, and K* jumps around instead of
    converging, which makes it impossible to tell a real zero from a sparse one.
    """
    R, t = T[:3, :3], T[:3, 3]
    pl, nl = sample_surface(mesh, max(n_dist, n_cand) * POOL, seed)
    pw, nw = pl @ R.T + t, nl @ R.T
    inside = region_mask(mesh, T, dirs, pl, nl)

    # disturbances: the process pushing on the work region
    pd, push_d = pw[inside][:n_dist], -nw[inside][:n_dist]
    # candidates: anywhere a support may touch, i.e. off the work region
    pc, push_c = pw[~inside][:n_cand], -nw[~inside][:n_cand]

    # The ground contact is a line once the workpiece is tipped, so it has no
    # area and sampling the surface by area essentially never lands on it. It is
    # the one contact we get for free, so it is added explicitly.
    Vw = mesh.vertices @ R.T + t
    touching = Vw[Vw[:, 1] <= CONTACT_EPS]
    if len(touching):
        step = max(1, len(touching) // 24)
        floor = touching[::step]
        pc = np.vstack([floor, pc])
        push_c = np.vstack([np.tile([0.0, 0.0, 1.0], (len(floor), 1)), push_c])

    scale = float(np.linalg.norm(mesh.extents))
    ref = mesh.center_mass @ R.T + t
    return {"pd": pd, "push_d": push_d, "pc": pc, "push_c": push_c,
            "ground": pc[:, 1] <= CONTACT_EPS,
            "W": wrenches(pc, push_c, ref, scale),
            "D": wrenches(pd, push_d, ref, scale),
            # one unit of weight at the centre of mass
            "gravity": np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            # the samples are area-uniform, so this estimates the area fraction;
            # supports.py itself reports the exact one, off the refined mesh
            "region_area_fraction_sampled": float(inside.mean())}


def ceiling(v_z_max: float) -> float:
    """The push no support geometry can raise, from vertical equilibrium alone.

    A support that never presses down contributes a non-negative vertical force,
    so summing the vertical components of `sum f_i + t*v + weight = 0` gives
    `t * v_z <= 1`. The tightest bound comes from the most downward-facing part
    of the work region, and it involves no contacts and no sampling: it is a
    property of the region by itself. A patch tilted just past horizontal is
    nearly harmless; one facing straight down caps the push at exactly the
    workpiece's own weight.
    """
    return 1.0 / v_z_max if v_z_max > 0 else np.inf


def region_ceiling(mesh, T, dirs, target_edge=None) -> dict:
    """`ceiling` evaluated on the mesh, so it does not depend on the sampling."""
    from work_regions import BOUNDARY_EDGE, split_along_boundary
    if target_edge is None:
        target_edge = BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents))
    refined, mask = split_along_boundary(
        mesh, lambda m: region_mask(mesh, T, dirs, m.triangles_center, m.face_normals),
        target_edge)
    # the push is along the inward normal, so its upward part is -n_z
    n_z = (refined.face_normals @ T[:3, :3].T)[:, 1]
    v_z = -n_z[mask]
    area = refined.area_faces
    return {"region_area_fraction": float(area[mask].sum() / area.sum()),
            "region_area_fraction_facing_down":
                float(area[mask][v_z > 0].sum() / max(area[mask].sum(), 1e-12)),
            "v_z_max": float(v_z.max()) if len(v_z) else -1.0,
            "k_star_ceiling": ceiling(float(v_z.max()) if len(v_z) else -1.0)}


def k_star(s: dict) -> dict:
    """How hard a push the supports can take without ever reaching over the part.

    Only candidates that push up or sideways are allowed, so every support could
    stand on the ground; K* is then the worst case over the pushes the process
    can make, in multiples of the workpiece's own weight.

    K* is a maximum over the cone a finite set of candidates spans, so it is an
    underestimate that only rises as the sampling is refined -- it is a lower
    bound on what the best support design achieves. `ceiling` bounds it from the
    other side, exactly. Read the pair, not either one alone.
    """
    W, D, gravity, push_c, push_d = s["W"], s["D"], s["gravity"], s["push_c"], s["push_d"]
    from_below = push_c[:, 1] >= DOWNWARD
    W_below = W[:, from_below]
    if not from_below.any() or not D.shape[1]:
        strength, n_free = np.array([0.0]), 0
    else:
        basis = free_motions(W_below)
        n_free = len(basis[1])
        FELL_BACK[0] = 0
        strength = np.array([max_push(W_below, D[:, i], gravity, basis)
                             for i in range(D.shape[1])])
    failed = int(np.isnan(strength).sum())
    ok = strength[~np.isnan(strength)]
    if not len(ok):
        ok = np.array([np.nan])
    worst = int(np.nanargmin(strength)) if len(ok) and np.isfinite(ok).any() else 0
    # K* = 0 for two quite different reasons: some push cannot be answered at all,
    # or the supports cannot even hold the workpiece up on their own.
    holds_gravity = bool(from_below.any() and opposable(W_below, -gravity)[0])
    v_z_max = float(push_d[:, 1].max()) if len(push_d) else -1.0
    return {"k_star_no_clamping": float(np.min(ok)),
            "k_star_median": float(np.median(ok)),
            "k_star_ceiling_sampled": ceiling(v_z_max),
            "v_z_max_sampled": v_z_max,
            "n_candidates_from_below": int(from_below.sum()),
            "n_free_motions": int(n_free),
            "holds_gravity_from_below": holds_gravity,
            "n_pushes_unanswerable": int((ok <= 0).sum()),
            "n_lp_failures": failed,
            "n_lp_bisected": int(FELL_BACK[0]),
            "worst_push_point": [float(v) for v in s["pd"][worst]] if len(s["pd"]) else None,
            "worst_push_dir": [float(v) for v in push_d[worst]] if len(push_d) else None,
            "n_disturbances": int(len(s["pd"])),
            "fraction_of_region_pushing_upward":
                float((push_d[:, 1] > 0.0).mean()) if len(push_d) else 0.0}


def analyse(mesh, T, dirs, n_dist, n_cand, seed, k):
    s = sample_sets(mesh, T, dirs, n_dist, n_cand, seed)
    pc, push_c, W, D, gravity = s["pc"], s["push_c"], s["W"], s["D"], s["gravity"]
    stats = k_star(s)

    targets = [-(k * D[:, i] + gravity) for i in range(D.shape[1])] + [-gravity]
    ok, used = np.zeros(len(targets), bool), np.zeros(len(pc), bool)
    for i, b in enumerate(targets):
        good, lam = opposable(W, b)
        ok[i] = good
        if good:
            used |= lam > 1e-9

    live = [b for b, g in zip(targets, ok) if g]
    keep = sparse_contact_set(W, live)

    return {**stats,
            "n_opposable": int(ok[:-1].sum()),
            "gravity_balanced": bool(ok[-1]),
            "contacts": [{"p": [float(v) for v in pc[i]],
                          "push": [float(v) for v in push_c[i]],
                          "on_ground": bool(s["ground"][i])} for i in keep],
            "_pc": pc, "_push_c": push_c, "_keep": keep, "_inside_pts": s["pd"]}


# ------------------------------------------------------------------ render ---

def paint_region(mesh, T, dirs, out_dir, tag):
    """Split the surface into work region and the rest, for colour."""
    from work_regions import BOUNDARY_EDGE, paint, split_along_boundary
    R = T[:3, :3]

    def label(m):
        return region_mask(m, T, dirs, m.triangles_center, m.face_normals)

    target = BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents))
    refined, mask = split_along_boundary(mesh, label, target)
    return paint(refined, mask, out_dir, tag), float(
        refined.area_faces[mask].sum() / refined.area)


def render(name, T, parts, contacts, px, azimuth):
    obj = obj_path(name)
    xml = SCENE.format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>' for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmp = obj / ".sup_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mesh = trimesh.load(obj / "mesh.stl", force="mesh")
    Wv = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = Wv.min(axis=0), Wv.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    L = ARROW_FRAC * float(np.linalg.norm(mesh.extents))

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -17.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.85 * size / 2 / np.tan(np.deg2rad(model.vis.global_.fovy / 2))

    with YUpRenderer(model, px, px, max_geom=2000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene
        for c in contacts:
            p, u = np.asarray(c["p"]), np.asarray(c["push"])
            rgba = np.array(PRESSES_DOWN if u[1] < DOWNWARD else PUSHES_UP, np.float32)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH, p - u * L, p)
            scn.ngeom += 1
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, ARROW_WIDTH * 1.6),
                                p, np.eye(3).ravel(), rgba)
            scn.ngeom += 1
        return Image.fromarray(r.render())


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--pose", type=int, default=0)
    ap.add_argument("--region", type=int, default=None, help="only this region index")
    ap.add_argument("--disturbances", type=int, default=120)
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--k", type=float, default=1.0,
                    help="the push is at most K times the workpiece's own weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    spec = read_json(d / "work_regions" / f"regions_pose{args.pose}.json")
    ex = read_json(d / "tips" / "tips.json")["examples"][args.pose]
    T = np.asarray(ex["T_world_mesh"])

    out = d / "supports"
    rows, records = [], []
    for i, reg in enumerate(spec["regions"]):
        if reg["kind"] != "spray" or (args.region is not None and i != args.region):
            continue
        dirs = [np.asarray(u) for u in reg["gun_directions"]]
        res = analyse(mesh, T, dirs, args.disturbances, args.candidates, args.seed, args.k)
        parts, area = paint_region(mesh, T, dirs, d / "work_regions" / "paint",
                                   f"sup_p{args.pose}_r{i}")

        n_down = sum(1 for c in res["contacts"] if c["push"][1] < DOWNWARD)
        rec = {"region_index": i, "n_passes": reg["n_passes"],
               "region_area_fraction": area,
               **{k: v for k, v in res.items() if not k.startswith("_")},
               "n_contacts": len(res["contacts"]),
               "n_contacts_pressing_down": n_down}
        records.append(rec)
        rows.append((rec, [render(args.object, T, parts, res["contacts"], args.size, az)
                           for az in (135.0, 315.0)]))
        ks = rec["k_star_no_clamping"]
        print(f"region {i} ({reg['n_passes']} pass, {100 * area:4.1f}% of the surface): "
              f"without clamping, holds a push of {ks:6.2f} x its own weight "
              f"(median over pushes {rec['k_star_median']:6.2f}); "
              f"{rec['n_contacts']} contacts at K={args.k:g} ({n_down} press down)",
              flush=True)

    write_json(out / f"supports_pose{args.pose}_k{args.k:g}.json",
               {"object": args.object, "pose": args.pose, "k": args.k,
                "regions": records})

    px = args.size
    label_w, head_h = int(px * 0.95), int(px * 0.18)
    sheet = Image.new("RGB", (label_w + 2 * px, head_h + px * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.050)), _font(int(px * 0.048))
    dr.text((label_w + 8, int(head_h * 0.28)),
            "green = the work region a push can land anywhere on\n"
            "arrows = the contacts; red would have to press down from above",
            fill=(20, 20, 20), font=f)
    for r, (rec, imgs) in enumerate(rows):
        y = head_h + r * px
        for c, im in enumerate(imgs):
            sheet.paste(im, (label_w + c * px, y))
        ks, cap = rec["k_star_no_clamping"], rec["k_star_ceiling"]
        verdict = (f"without clamping it holds\na push of {ks:.2f}x its own weight\n"
                   + ("nothing caps it: the work\nregion is all upper surface"
                      if np.isinf(cap) else
                      f"and no support could pass\n{cap:.2f}x -- the region reaches\nunderneath"))
        dr.text((10, y + int(px * 0.10)),
                f"{rec['n_passes']} spray pass(es)\n"
                f"work region: {100 * rec['region_area_fraction']:.1f}% of the surface\n"
                f"{100 * rec['fraction_of_region_pushing_upward']:.0f}% of its pushes "
                f"point upward\n\n"
                f"{verdict}\n\n"
                f"(at most {rec['n_contacts']} contacts, an upper bound)",
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))
    sheet.save(out / f"sheet_pose{args.pose}_k{args.k:g}.png")
    print(f"\n{out / f'sheet_pose{args.pose}_k{args.k:g}.png'}  "
          f"{sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
