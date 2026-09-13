"""Build the MuJoCo scene (ground + one object) shared by sampling and rendering."""
from __future__ import annotations

import numpy as np

from common import obj_path, read_json
from pathlib import Path

# sliding / torsional / rolling.  A perfectly flat plane with MuJoCo's default
# rolling coefficient lets rounded objects roll forever, so it is raised here to
# the level of a real table.  Rolling resistance only dissipates energy; it does
# not change which placements are stable.
FRICTION = "0.8 0.02 0.01"
# condim 6 is what makes the torsional and rolling coefficients above take effect
# at all; with MuJoCo's default condim 3 they are silently ignored.
CONDIM = 6

TEMPLATE = """<mujoco model="{name}">
  <compiler meshdir="." angle="radian"/>
  <option gravity="0 -9.81 0" timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1200" offheight="1200"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.005" zfar="30"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <material name="floor" rgba="0.97 0.97 0.97 1" reflectance="0"/>
    <material name="part" rgba="0.87 0.58 0.28 1" specular="0.3" shininess="0.4"/>
{assets}
  </asset>
  <worldbody>
    <light name="key" pos="0.5 1.4 -0.6" dir="-0.35 -1 0.42" directional="true" castshadow="true"
           diffuse="0.7 0.7 0.7"/>
    <geom name="ground" type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="2 2 0.05" material="floor" friction="{friction}"
          condim="{condim}"/>
    <body name="obj" pos="0 {drop_z} 0">
      <freejoint name="root"/>
      <inertial pos="{com}" mass="{mass}" fullinertia="{inertia}"/>
      <geom name="visual" type="mesh" mesh="visual" material="part"
            group="2" contype="0" conaffinity="0"/>
{geoms}
    </body>
  </worldbody>
{keyframes}</mujoco>
"""


def build_xml(name: str, drop_z: float = 0.3, keyframes: np.ndarray | None = None,
              friction: str = FRICTION, condim: int = CONDIM) -> str:
    """Assemble the scene XML for one object.

    keyframes: optional (n, 7) array of free-joint qpos rows, exposed as
    MuJoCo keyframes so `python -m mujoco.viewer` can step through the poses.
    """
    d = obj_path(name)
    meta = read_json(d / "meta.json")
    parts = sorted((d / "collision").glob("part_*.obj"))

    assets = ['    <mesh name="visual" file="mesh.stl"/>']
    geoms = []
    for i, p in enumerate(parts):
        assets.append(f'    <mesh name="p{i:02d}" file="collision/{p.name}"/>')
        geoms.append(
            f'      <geom name="col{i:02d}" type="mesh" mesh="p{i:02d}" group="3"'
            f' friction="{friction}" condim="{condim}" rgba="0.25 0.55 0.85 0.35"/>'
        )

    I = np.asarray(meta["inertia_com"])
    fullinertia = " ".join(
        f"{v:.10g}" for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    )

    kf = ""
    if keyframes is not None and len(keyframes):
        rows = "\n".join(
            f'    <key name="pose_{i:02d}" qpos="{" ".join(f"{v:.9g}" for v in q)}"/>'
            for i, q in enumerate(keyframes)
        )
        kf = f"  <keyframe>\n{rows}\n  </keyframe>\n"

    return TEMPLATE.format(
        name=name,
        assets="\n".join(assets),
        geoms="\n".join(geoms),
        friction=friction,
        condim=condim,
        drop_z=f"{drop_z:.6g}",
        com=" ".join(f"{v:.9g}" for v in meta["com_mesh_frame"]),
        mass=f"{meta['mass_kg']:.9g}",
        inertia=fullinertia,
        keyframes=kf,
    )


def write_scene(name: str) -> Path:
    """(Re)write objects/<name>/scene.xml, with the sampled placements as keyframes.

    Called by both the sampler and the renderer so the file on disk always
    matches the current template.
    """
    d = obj_path(name)
    keyframes = None
    poses_file = d / "poses.json"
    if poses_file.exists():
        poses = read_json(poses_file)["poses"]
        keyframes = np.array([list(p["pos"]) + list(p["quat_wxyz"]) for p in poses])
    out = d / "scene.xml"
    out.write_text(build_xml(name, keyframes=keyframes))
    return out
