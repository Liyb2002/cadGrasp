# Object library and stable placements

Shared helpers for the slides and the object library. The current problem is in
[README.md](../README.md), with model decisions in
[README.md](../README.md#当前决定与讨论记录). Code lives only in `slides/` and `codes/`.
`cone_model.py` defines the shared direction cap; `mesh_export.py` exports display
meshes for MuJoCo. Optional standalone figures go to `slides/tools/figures/`.

The object library contains workpieces, each
with a handful of stable ground placements `T0`. Nothing here designs supports
yet -- it produces the inputs everything downstream is evaluated on.

Run everything in the `cadgrasp` conda environment:

```bash
conda run -n cadgrasp python slides/tools/fetch_objects.py     # download + mass properties
conda run -n cadgrasp python slides/tools/decompose.py         # convex decomposition (~5 min)
conda run -n cadgrasp python slides/tools/drop_sample.py       # drop, settle, cluster
conda run -n cadgrasp python slides/tools/render_poses.py      # renders + contact sheet
```

Only the last three depend on the previous one; each is idempotent and skips
work that is already on disk (`--force` to redo it).

## Objects

20 of the 23 STLs of the Passive Grippers (Kodnongbua et al., SIGGRAPH 2022)
test suite, from <https://github.com/milmillin/passive-gripper>. They are
watertight, in metres, 5--19 cm across, and span a wide range of concavity --
D4 encloses only 4% of its own convex hull.

D5, D6 and D7 (apple, peach, orange) are excluded. They are near-spheres with no
discrete stable placements -- 40 drops gave 36, 16 and 24 distinct resting
directions -- so they have neither a well-defined `T0` nor an edge to tip about,
which puts them outside the scope of the problem.

Meshes are stored exactly as downloaded. Every pose in the pipeline is expressed
in this untouched mesh frame, so nothing has to be un-normalised later.

Mass properties assume uniform density (1000 kg/m^3). The suite is hand-scale
while the workpieces of the paper are 5--30 kg; that only matters downstream, as
which placements are stable does not depend on scale.

## How placements are sampled

`drop_sample.py` drops each object 40 times from a uniformly random orientation,
2 cm above the ground, and simulates until it comes to rest.

Which placement an object rests in is decided by *which face touches the ground*,
i.e. by the direction of gravity expressed in the mesh frame; yaw about the world
z axis is free. Settled trials are therefore clustered on that direction (8
degrees tolerance), and the most frequent clusters are kept, with the cluster
frequency reported as the probability of that placement. The stored pose is
canonical: yaw removed, object resting on z = 0, centre of mass over the origin.
Since the cluster mean sits a fraction of a degree off the true equilibrium for
curved contacts, the object is set down once and allowed to relax before being
canonicalised, then re-simulated to confirm it holds still (`verified_stable`).

Each object is seeded off its name rather than its position in the list, so
adding or removing an object does not resample every other object.

Two settings matter and are easy to get wrong:

- **`condim = 6`.** With MuJoCo's default `condim = 3` the torsional and rolling
  friction coefficients are silently ignored, and rounded objects (D5, D6, D7)
  roll forever without ever settling.
- **Rest is judged on a time window, not on velocity.** Rounded objects creep at
  velocities that never trip a velocity threshold, and objects with many contact
  points chatter numerically at velocities that never fall below one. A pose that
  moves less than 0.5 mm and 0.5 degrees over a 1 s window counts as at rest.

Convex decomposition matters for the same reason: MuJoCo collides a mesh as its
convex hull, so without it a hook or a C-shaped part would settle on a face that
does not exist.

## Output layout

```
objects/
  index.json                 all objects: size, volume, mass, concavity
  contact_sheet.png          23 rows x 5 placements, the thing to look at
  <name>/
    mesh.stl                 as downloaded
    collision/part_XX.obj    convex pieces (CoACD)
    meta.json                mass properties, extents, concavity
    poses.json               the stable placements (below)
    scene.xml                MuJoCo scene; each placement is a keyframe
    renders/pose_XX.png      one render per placement, plus strip.png
```

`python -m mujoco.viewer --mjcf=objects/<name>/scene.xml` opens the scene; the
keyframes step through the placements.

Each entry of `poses.json` carries, besides `T_world_mesh` and its probability:

- `gravity_in_mesh_frame` -- the resting face, and the key the clustering used
- `support_polygon`, `support_area_m2`, `com_inside_support`
- `h_com_m` and `d_edges_m` -- the height of the centre of mass and its distance
  to each support-polygon edge. Those edges are the candidate pivot edges of a
  tip, and `d/h` is the tipping ratio of constraint C1: starting the tip about
  an edge needs a push of about `mg*d/h`. `d_over_h_min` is the easiest edge.

## Target poses and disturbances

`tip_examples.py` builds target poses: the workpiece rotated off a stable
placement about a straight side of its footprint (an edge pivot) or about a
single corner of it (a point pivot). It never translates, so the ground contact
it keeps is part of the contact it already had. Each tip stops either when the
centre of mass arrives over the pivot, after which gravity drives the fall
instead of resisting it, or when some other part of the workpiece reaches the
floor, after which the pivot would move. Output: `objects/<name>/tips/`.

`disturbances.py` builds the adversary set for a target pose. A rigid push on a
surface acts along the inward normal at the point it touches, so the complete set
of disturbances is one push per surface point, excluding the patch already
resting on the ground. Each push is scored by the torque it makes about the
ground contact, resolved along the direction gravity is already turning the
workpiece:

- **red** -- the push adds to the fall the workpiece is already trying to make
- **blue** -- the push opposes that fall

Magnitudes are not modelled: a push is a direction, and a support answers it if
it can push back at all. Output: `objects/<name>/disturbances/`, with the surface
painted by the two groups; the boundary between them is the set of points whose
push line passes through the pivot.

Note what red and blue do *not* mean. The score only resolves the torque along
the one axis gravity is already turning about. A blue push can still slide the
workpiece, and about a point pivot it can still topple it sideways -- about an
edge pivot it cannot, because the line of ground contact blocks that. Blue means
"does not add to this fall", not "harmless".

## Work regions

`work_regions.py` generates the patch a process has to reach, which is an input
of the problem and the thing the supports have to stay clear of.

**spray** (coating, painting, blasting): the gun stands off and sprays along one
direction, so the region is everything that both faces the gun and is in its line
of sight. One large connected patch; with the gun above, it lands on the upper
side. Parameters: gun direction, incidence limit (60 degrees).

**seam** (welding, sealing, deburring): a narrow band along a curve, here a
planar section of the workpiece. The band is measured from the section curve, not
from the plane -- on a face lying almost in the plane those differ wildly and the
distance to the plane would select the whole face. Planes whose section is
degenerate (band area far larger than curve length times width) are resampled.

Both keep only what can be reached from outside: a face counts as reachable if
any direction within 45 degrees of its normal escapes the workpiece.

Regions are stored as the parameters that generate them, not as face lists, so
the files stay small. Output: `objects/<name>/work_regions/`.

Two limits worth knowing. The band width is comparable to the facet size on the
flat parts of these meshes, so a seam crossing one large facet paints all of it.
And `n_pieces` counts connected components after the reachability filter, which
punches holes in a band and inflates the count; `largest_piece_fraction` is the
more meaningful number.

What is not modelled yet is the tool itself. The supports have to avoid the
volume the nozzle or the torch sweeps, which is much larger than the patch it
works on, and that volume -- not the patch -- is what will actually constrain the
design.
