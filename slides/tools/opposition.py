"""Step 2: can each disturbance be pushed back against, and by how many contacts.

A push can be arbitrarily large, so a support answers it only if it can push back
along the same line -- no finite force, gravity included, helps. Written out:
the reversed wrench of the disturbance has to lie in the cone of wrenches the
contacts can produce. Gravity therefore drops out of this step entirely.

For each target pose this asks two things:

    with every surface point available as a contact, which disturbances can be
    answered at all?

    and how few contacts suffice to answer all of them?

    python slides/tools/opposition.py A1-f
"""
from __future__ import annotations

import argparse

import numpy as np
import trimesh
from scipy.optimize import linprog

import mujoco
from mujoco import Renderer
from PIL import Image, ImageDraw

from common import mat_to_quat_wxyz, obj_path, read_json, write_json
from disturbances import CONTACT_EPS, SCENE, _font, sample_surface

RESIDUAL_TOL = 1e-6     # a reversed wrench this close to the cone counts as inside
ARROW_FRAC = 0.28       # arrow length as a fraction of the workpiece size
ARROW_WIDTH = 2.6e-3
DOWNWARD = -0.3         # a push with this much -z has to come from above the part
PRESSES_DOWN = (0.88, 0.22, 0.18, 1.0)
PUSHES_UP = (0.20, 0.45, 0.85, 1.0)


def wrenches(p: np.ndarray, push: np.ndarray, ref: np.ndarray, scale: float) -> np.ndarray:
    """6 x n wrench matrix: unit pushes at points p, moments about ref.

    Moments are divided by the workpiece size so the two halves of a wrench are
    the same order of magnitude and the solver is well conditioned.
    """
    return np.vstack([push.T, np.cross(p - ref, push).T / scale])


def opposable(W: np.ndarray, target: np.ndarray) -> tuple[bool, np.ndarray]:
    """Is `target` a non-negative combination of the columns of W?

    Solved as a least-absolute-residual LP so the answer degrades gracefully
    instead of flipping on a numerical hair, and so the residual is reported.
    """
    m, n = W.shape
    # variables: [lambda (n) | s+ (m) | s- (m)],  W l + s+ - s- = target
    # m is 6 for the full problem and 3 when only translation is being balanced
    A = np.hstack([W, np.eye(m), -np.eye(m)])
    c = np.concatenate([np.zeros(n), np.ones(2 * m)])
    r = linprog(c, A_eq=A, b_eq=target, bounds=(0, None), method="highs")
    if not r.success:
        return False, np.zeros(n)
    return bool(r.fun < RESIDUAL_TOL), r.x[:n]


def analyse(mesh: trimesh.Trimesh, T: np.ndarray, n_dist: int, n_cand: int,
            seed: int = 0, cand_seed: int | None = None) -> dict:
    R, t = T[:3, :3], T[:3, 3]
    scale = float(np.linalg.norm(mesh.extents))
    ref = mesh.center_mass @ R.T + t

    def surface(n, s):
        pts, nrm = sample_surface(mesh, n, s)
        p = pts @ R.T + t
        push = -(nrm @ R.T)
        return p, push

    # disturbances: every push except on the patch already resting on the ground
    pd, push_d = surface(n_dist, seed)
    free = pd[:, 2] > CONTACT_EPS
    pd, push_d = pd[free], push_d[free]

    # candidate contacts: anywhere on the surface, plus the ground contact itself
    pc, push_c = surface(n_cand, seed + 991 if cand_seed is None else cand_seed)
    ground = pc[:, 2] <= CONTACT_EPS
    push_c[ground] = np.array([0.0, 0.0, 1.0])       # the floor can only push up

    W = wrenches(pc, push_c, ref, scale)
    D = wrenches(pd, push_d, ref, scale)

    ok = np.zeros(len(pd), bool)
    used = np.zeros(len(pc), bool)
    for i in range(D.shape[1]):
        good, lam = opposable(W, -D[:, i])
        ok[i] = good
        if good:
            used |= lam > 1e-9

    return {"points_d": pd, "push_d": push_d, "points_c": pc, "push_c": push_c,
            "ground": ground, "W": W, "D": D, "opposable": ok, "touched": used,
            "scale": scale, "ref": ref}


def minimal_set(res: dict) -> np.ndarray:
    """Shrink the contact set until dropping any one of them breaks something."""
    W, D, ok = res["W"], res["D"], res["opposable"]
    keep = np.flatnonzero(res["touched"])
    targets = [-D[:, i] for i in np.flatnonzero(ok)]

    changed = True
    while changed:
        changed = False
        for j in list(keep):
            trial = keep[keep != j]
            if len(trial) == 0:
                continue
            Wt = W[:, trial]
            if all(opposable(Wt, b)[0] for b in targets):
                keep = trial
                changed = True
                break
    return keep


def render_contacts(name: str, T: np.ndarray, contacts: list[dict], px: int,
                    azimuth: float) -> Image.Image:
    """The workpiece with the contacts of the minimal set drawn on it."""
    obj = obj_path(name)
    xml = SCENE.format(assets='    <mesh name="m" file="mesh.stl"/>',
                       geoms='      <geom type="mesh" mesh="m" material="holds"/>',
                       pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
                       quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    xml = xml.replace('name="holds" rgba="0.36 0.55 0.82 1"',
                      'name="holds" rgba="0.86 0.80 0.70 1"')
    tmp = obj / ".opp_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mesh = trimesh.load(obj / "mesh.stl", force="mesh")
    W = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = W.min(axis=0), W.max(axis=0)
    L = ARROW_FRAC * float(np.linalg.norm(mesh.extents))

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -16.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.8 * float(np.linalg.norm(hi - lo)) / 2 / np.tan(
        np.deg2rad(model.vis.global_.fovy / 2))

    with Renderer(model, px, px, max_geom=2000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene
        for c in contacts:
            p, u = np.asarray(c["p"]), np.asarray(c["push"])
            rgba = np.array(PRESSES_DOWN if u[2] < DOWNWARD else PUSHES_UP, np.float32)
            for kind, a, b, w in ((mujoco.mjtGeom.mjGEOM_ARROW, p - u * L, p, ARROW_WIDTH),
                                  (mujoco.mjtGeom.mjGEOM_SPHERE, p, p, 0.0)):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    mujoco.mjv_initGeom(g, kind, np.full(3, ARROW_WIDTH * 1.5),
                                        p, np.eye(3).ravel(), rgba)
                else:
                    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3),
                                        np.zeros(9), rgba)
                    mujoco.mjv_connector(g, kind, w, a, b)
                scn.ngeom += 1
        return Image.fromarray(r.render())


def build_sheet(name: str, records: list[dict], px: int) -> None:
    d = obj_path(name)
    mesh_examples = {(e["placement"], e["pivot"]): e
                     for e in read_json(d / "tips" / "tips.json")["examples"]}
    label_w, head_h = int(px * 0.95), int(px * 0.22)
    sheet = Image.new("RGB", (label_w + 2 * px, head_h + px * len(records)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.052)), _font(int(px * 0.050))
    dr.text((label_w + 8, int(head_h * 0.3)),
            "the smallest set of contacts that answers every push\n"
            "red = has to press down, so it must reach over the top",
            fill=(20, 20, 20), font=f)

    for r, rec in enumerate(records):
        ex = mesh_examples[(rec["placement"], rec["pivot"])]
        T = np.asarray(ex["T_world_mesh"])
        y = head_h + r * px
        for c, az in enumerate((135.0, 315.0)):
            sheet.paste(render_contacts(name, T, rec["contacts"], px, az),
                        (label_w + c * px, y))
        n_down = sum(1 for c in rec["contacts"] if c["push"][2] < DOWNWARD)
        dr.text((10, y + int(px * 0.14)),
                f"placement {rec['placement']}, {rec['pivot']} pivot\n"
                f"tipped {rec['tip_deg']:.0f} deg\n\n"
                f"{rec['n_opposable']}/{rec['n_disturbances']} pushes answerable\n"
                f"{rec['n_contacts_minimal']} contacts needed\n"
                f"{n_down} of them press down\n"
                f"{rec['n_of_those_on_the_ground']} of them are the ground",
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))
    out = d / "disturbances" / "opposition_sheet.png"
    sheet.save(out)
    print(f"{out}  {sheet.size[0]}x{sheet.size[1]}")


def variants(name: str, pose_index: int, k: int, n_dist: int, n_cand: int,
             px: int) -> None:
    """Several different minimal sets for the same pose.

    The count is forced by the algebra and comes out the same every time. Where
    the contacts sit is not: each run draws a different set of candidate points
    and lands on a different valid answer. That freedom is the room we have left
    to spend on everything the count does not know about -- keeping the work
    surface clear, and being able to get a support in there at all.
    """
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    ex = read_json(d / "tips" / "tips.json")["examples"][pose_index]
    T = np.asarray(ex["T_world_mesh"])

    sets = []
    for i in range(k):
        res = analyse(mesh, T, n_dist, n_cand, seed=0, cand_seed=1000 + 37 * i)
        keep = minimal_set(res)
        sets.append([{"p": [float(v) for v in res["points_c"][j]],
                      "push": [float(v) for v in res["push_c"][j]]} for j in keep])
        print(f"run {i}: {int(res['opposable'].sum())}/{len(res['opposable'])} "
              f"pushes answerable, {len(keep)} contacts", flush=True)

    sheet = Image.new("RGB", (px * len(sets), px * 2 + int(px * 0.16)), "white")
    dr = ImageDraw.Draw(sheet)
    f = _font(int(px * 0.055))
    for i, cs in enumerate(sets):
        for r, az in enumerate((135.0, 315.0)):
            sheet.paste(render_contacts(name, T, cs, px, az),
                        (i * px, int(px * 0.16) + r * px))
        dr.text((i * px + 10, int(px * 0.045)), f"a different set of {len(cs)}",
                fill=(20, 20, 20), font=f)
        dr.line([(i * px, 0), (i * px, sheet.size[1])], fill=(220, 220, 220))
    out = d / "disturbances" / f"minimal_sets_pose{pose_index}.png"
    sheet.save(out)
    print(f"{out}  {sheet.size[0]}x{sheet.size[1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--disturbances", type=int, default=120)
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N poses")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--render-only", action="store_true",
                    help="redraw the sheet from the opposition.json already written")
    ap.add_argument("--variants", type=int, default=0,
                    help="show N different minimal sets for one pose")
    ap.add_argument("--pose", type=int, default=0)
    args = ap.parse_args()

    if args.variants:
        variants(args.object, args.pose, args.variants,
                 args.disturbances, args.candidates, args.size)
        return

    if args.render_only:
        rec = read_json(obj_path(args.object) / "disturbances" / "opposition.json")
        build_sheet(args.object, rec["examples"], args.size)
        return

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"][:args.limit]

    records = []
    for ex in examples:
        res = analyse(mesh, np.asarray(ex["T_world_mesh"]),
                      args.disturbances, args.candidates, args.seed)
        n = len(res["opposable"])
        n_ok = int(res["opposable"].sum())
        keep = minimal_set(res) if n_ok else np.array([], int)
        n_ground = int(res["ground"][keep].sum())

        records.append({"placement": ex["placement"], "pivot": ex["pivot"],
                        "tip_deg": ex["tip_deg"], "n_disturbances": n,
                        "n_opposable": n_ok, "n_unopposable": n - n_ok,
                        "n_contacts_minimal": int(len(keep)),
                        "n_of_those_on_the_ground": n_ground,
                        "contacts": [{"p": [float(v) for v in res["points_c"][i]],
                                      "push": [float(v) for v in res["push_c"][i]],
                                      "on_ground": bool(res["ground"][i])}
                                     for i in keep]})
        print(f"placement {ex['placement']} {ex['pivot']:5s} pivot: "
              f"{n_ok:3d}/{n:3d} disturbances can be pushed back against; "
              f"{len(keep)} contacts needed "
              f"({n_ground} of them the ground itself)", flush=True)

    write_json(d / "disturbances" / "opposition.json",
               {"object": args.object, "n_disturbances": args.disturbances,
                "n_candidates": args.candidates, "examples": records})
    print(f"\n-> {d / 'disturbances' / 'opposition.json'}")
    build_sheet(args.object, records, args.size)


if __name__ == "__main__":
    main()
