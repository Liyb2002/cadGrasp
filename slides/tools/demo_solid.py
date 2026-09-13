"""The first thing that looks like a fixture: pads on the contacts, columns to the floor.

The search returns directions and the places they are taken from, which is not yet
anything you could print. This turns one of those answers into solid geometry the
crudest way that is still honest: at each contact a small pad lying in the tangent
plane -- so its face is parallel to the surface it bears on and its normal is the
push direction the search chose -- and from that pad a column dropped straight down
to the ground.

It is deliberately the dumb version. Nothing here chooses WHERE on a face to sit
(a flat face offers a whole region and any point of it gives the same direction),
nothing merges two contacts that could share one piece, and nothing checks that the
column can be inserted or printed. Those are the interesting questions and they are
not asked here. What this does answer is the one thing a picture can: whether the
directions the search likes correspond to places a support could plausibly reach.

    python slides/tools/demo_solid.py A1-f --poses 1 3
"""
from __future__ import annotations
import coordinates as COORD

import argparse
import shutil

import mujoco
from mujoco import Renderer
import numpy as np
import trimesh
from PIL import Image

from common import mat_to_quat_wxyz, obj_path, read_json

PAD_R = 0.006          # 6 mm pad, big enough to see and to bear on
PAD_N = 16
FLOOR = 0.0

SCENE = """
<mujoco>
  <visual>
    <global offwidth="1400" offheight="1400" fovy="45"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <material name="part" rgba="0.80 0.83 0.79 1" specular="0.15" shininess="0.3"/>
    <material name="work" rgba="0.55 0.82 0.62 1" specular="0.1"/>
    <material name="prop" rgba="0.90 0.45 0.12 1" specular="0.25" shininess="0.4"/>
    <material name="floor" rgba="0.95 0.95 0.93 1" specular="0.05"/>
{assets}
  </asset>
  <worldbody>
    <light pos="0.35 -0.45 0.75" dir="-0.4 0.5 -0.85" directional="true"
           diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="2 2 0.1" material="floor"/>
    <body pos="{pos}" quat="{quat}">
      <geom type="mesh" mesh="part" material="part"/>
    </body>
{props}
  </worldbody>
</mujoco>
"""


def pad_column(p, u, r=PAD_R, n=PAD_N):
    """A pad in the tangent plane at `p`, dropped to the floor as a column.

    The pad's normal IS the push direction, so its face lies parallel to the
    surface it bears against -- that is the whole of the contact model made
    solid. The column is the laziest possible way to get the load to the ground:
    straight down, no attempt to route around anything.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    a = np.array([0.0, 0.0, 1.0])
    if abs(u @ a) > 0.9:                       # pick any axis not along u
        a = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = p + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    bot = top.copy()
    bot[:, 2] = FLOOR
    if top[:, 2].min() <= FLOOR + 1e-6:        # already on the ground, nothing to drop
        bot[:, 2] = FLOOR - 0.002
    V = np.vstack([top, bot])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]            # the wall
    for i in range(1, n - 1):
        F += [[0, i + 1, i], [n, n + i, n + i + 1]]        # the two caps
    return trimesh.Trimesh(vertices=V, faces=np.array(F), process=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--record", default="capped/capped_k1.json")
    ap.add_argument("--size", type=int, default=1100)
    args = ap.parse_args()

    d = obj_path(args.object)
    rec = {p["pose"]: p for p in read_json(d / args.record)["poses"]}
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else sorted(rec)
    tmp = d / "_demo"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy(d / "mesh.stl", tmp / "part.stl")

    shots = []
    for pose in poses:
        T = np.asarray(examples[pose]["T_world_mesh"])
        sup = rec[pose]["supports"]
        assets = ['    <mesh name="part" file="_demo/part.stl"/>']
        props = []
        for i, s in enumerate(sup):
            solid = pad_column(np.asarray(s["p"]), np.asarray(s["push"]))
            solid.export(tmp / f"prop{pose}_{i}.stl")
            assets.append(f'    <mesh name="p{pose}_{i}" file="_demo/prop{pose}_{i}.stl"/>')
            props.append(f'    <geom type="mesh" mesh="p{pose}_{i}" material="prop"/>')
        xml = SCENE.format(assets="\n".join(assets), props="\n".join(props),
                           pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
                           quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
        f = d / ".demo.xml"
        f.write_text(xml)
        try:
            model = mujoco.MjModel.from_xml_path(str(f))
        finally:
            f.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mesh = trimesh.load(d / "mesh.stl", force="mesh")
        V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
        lo, hi = V.min(axis=0), V.max(axis=0)
        cam = mujoco.MjvCamera()
        # frame the part AND its columns: the props run to the floor, so the
        # subject is taller than the part's own box
        cam.azimuth, cam.elevation = 118.0, -24.0
        cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] * 0.45]
        span = max(float(np.linalg.norm(COORD.floor(hi-lo))), hi[2] * 1.35)
        cam.distance = 1.25 * span / 2 / np.tan(np.deg2rad(22.5))
        with Renderer(model, args.size, args.size) as r:
            r.update_scene(data, camera=cam)
            shots.append(Image.fromarray(r.render()))
        print(f"pose {pose}: {len(sup)} props   "
              + "  ".join(f"push({s['push'][0]:+.2f},{s['push'][1]:+.2f},{s['push'][2]:+.2f})"
                          for s in sup), flush=True)

    w = Image.new("RGB", (args.size * len(shots), args.size), "white")
    for i, im in enumerate(shots):
        w.paste(im, (args.size * i, 0))
    out = d / "demo_fixture.png"
    w.save(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
