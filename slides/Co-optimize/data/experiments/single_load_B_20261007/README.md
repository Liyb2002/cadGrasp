# B once-loaded pilot, 2026-10-07

Superseded by [fixed user-grasp preparation and validation](../step4_grasp_research_B_20261007/README.md).
The user now supplies the grasp; kinematics constrains support design. The earlier
finite failure below is not an impossibility result: a fixed stock-Franka example
now passes object-grasp checks at all four task endpoints. Common support holding
and insertion still require Step5 design. This directory is retained unchanged as history.

Authoritative final run:
`pose1+2+4+6/pad_checked/report.json`.
Object B, poses 1, 2, 4, 6. Deterministic budget: 100 antipodal pairs,
12 gripper rolls per pair, maximum jaw opening 80mm. Contacts must avoid
the union of all four task work surfaces, including the entire finger pads.

Result: no grasp candidate. Of 1,200 sampled hand configurations, 1,185
were rejected by exact hand/object approach or closure collision and 15 by
finger-pad/work-surface contact. No candidate reached IK in this final run.
Step5 was not entered, no original-load coverage was evaluated, and no fixture
or inter-pose robot path was accepted. This finite budget does not prove that
B lacks a feasible grasp or fixture.

Earlier folders are diagnostic revisions. They checked only grasp-point work
membership, so their eight endpoint-IK candidates are superseded by the final
whole-pad check. The earlier `final` diagnostic tried 1,120 fixture candidates:
960 translations lost grip material, 146 nominal sweeps intersected it and
14 padded sweeps consumed it. None entered the force gate. Do not report these
earlier grasps as accepted candidates.

Final runtime source snapshots and hashes, input hashes and arguments are saved
inside `pad_checked`. Four physical regression tests passed: reachable IK,
unreachable IK, whole-pad work intersection, and rejection of upside-down
open-tray retention. Robot trajectories and retention branches were not exercised
on B because the final grasp gate failed.

Reproduce into a fresh directory:

```sh
.venv/bin/python slides/Co-optimize/single_load/run.py --out /tmp/cadgrasp-B-single-load-rerun
.venv/bin/python slides/Co-optimize/single_load/test_single_load.py
```
