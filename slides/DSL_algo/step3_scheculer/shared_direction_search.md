# Joint head programs and a searched common world exit

Current entry:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo \
  .venv/bin/python -m step3_scheculer.shared_direction_search B \
  --sets pose3+15 pose1+12+29 pose8+9+13+30
```

Every saved object pose remains fixed in its native task-world XYZ frame.
There are no `bases`, `offsets`, yaw or object-pose optimization variables.
A state is one contact-head set per pose plus **one shared object-exit unit
vector in world XYZ**. The support's relative ray is its negative. Direction
identity uses vector values, not per-pose menu indices or object-frame axes.

The solver intersects the current Step2 world-ray catalogues. It searches
those three-dimensional rays, with diverse angular starts and a swept-AABB
proxy for ordering. No direction, including +Z, is prescribed. A candidate
ray induces a new compatible head pool for each pose; compatible initialized
heads are retained first, then actual cached surface heads are added/replaced
to recover force feasibility. A fresh start can escape the retained heads'
roadmap-component restriction. All candidate centers, triangles, areas and
normals are real cached geometry, never interpolated fictitious forces.

The first feasible direction/head state is followed by up to eight nearby
menu-ray proposals. Each proposal uses the current accepted head sets as its
seed and repairs incompatible heads. Direction and all affected head sets
are accepted atomically only if every pose remains feasible and the swept-box
proxy decreases. Failed proposals leave the incumbent intact. This is a
finite discrete joint search, not continuous surface descent or a globally
optimal exit direction. Head count and area may change within the cached
0.5%, 1%, 2% families; default head budget is ten per pose.

A verified separating vector can prune a ray when even the optimistic cone
of **all** its eligible heads cannot support an original load. Eight original
load indices propose these exclusions; unresolved dual solves never prune.
Head additions are ranked by full original-load coverage within a bounded
64-head geometric shortlist. These proposal budgets are not infeasibility
proofs. Original LP formulations have a one-second solve limit; unresolved
proposals are excluded, never accepted or labeled physically infeasible.

Accepted contact programs undergo independent CPU checks on all 32,768
original six-dimensional force/torque demands plus the seventh shared
no-uplift equation. Force generators are rebuilt from actual contact triangles
and compared with their cached values. The support resultant is downward or
zero because summed head force **on the object** has nonnegative world Z.
Each pose's actual finite-depth head union gets a continuous ray check,
including the moving object's floor and separated endpoint. The shared exit
vectors in the per-pose plans are exactly the same world vector.

The new `shared_direction_paths.py` preserves optional source-face metadata
when shifting head geometry for relative-motion tests. This repairs the
updated withdrawal interface without modifying historical generators or
invalidating their accepted reports. Its eight path regressions include
continuous tunneling, floor collision and bent paths.

Outputs: `output/<object>/<saved-set>/step3_scheculer/dsl_shared_direction_v32/`.
V30/V31 are preserved initialization trials. V32 can warm-start a hash-valid
V31 head program but independently verifies its final original-load coverage.
Reports distinguish warm starts, search history, numerical unresolved counts,
per-pose contact files and full coverage arrays. Meshes, poses and samples are
never regenerated.

## Exact scope

This stage certifies **contact subprograms** and each subprogram's own head
exit. It does not connect the heads into a shared solid or check foreign-pose
heads against each object's exit. Reports explicitly set
`full_support_constructed=false` and
`foreign_heads_and_connections_checked=false`. Consequently `passed` here
means the stated contact-program scope, never an accepted complete fixture.
Full shared-body construction and its continuous-sweep acceptance remain the
next stage. The swept-box proxy is never reported as actual fixture volume
or evidence of improved compactness.

## Current-data verification

B `pose3+15`, `pose1+12+29` and `pose8+9+13+30` all find shared contact-program
solutions. Their selected +Z states have per-pose head counts [3,3], [3,4,3]
and [3,3,4,9], respectively. Search times including warm initialization and
refinement are 156.22, 26.94 and 58.73 seconds. The swept-box proxy retains +Z;
this is a measured search outcome, not a prescribed direction.

All three groups also have an independently CPU-verified **nonvertical**
shared direction `[-0.2588190451, 0, 0.9659258263]`, fifteen degrees from +Z.
Each pose passes all 32,768 original loads, the shared no-uplift equation and
its own continuous finite-depth head exit. These are exported as
`nonvertical/solution.json` and actual per-pose contact NPZ files. Complete
shared-body clearance is still unverified. Seven/eight other nearby direction
trials are feasible, but not all have independent CPU replays.

Nineteen regressions pass and one historical-data test is skipped. Reproduce
the final/oblique CPU and path audit with
`python -m step3_scheculer.review_shared_direction_search` in the same environment.
