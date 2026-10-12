# The heads' total force

![The heads' total force](head_total_force.png)

This is a conceptual comparison using the B / pose 2 workpiece and contact geometry. Both panels show the ground plane but omit the ground-base polygon, floor connector, and floor-force arrow. The saved side frame still connects the depicted heads.

- **Panel (a):** One blue upper contact sits on the crown, with an arm that approaches from above and connects to the side frame. Its arrow shows the support pushing the workpiece downward.
- **Panel (b):** Two blue contacts remain: the crown contact and the existing bottom contact. The middle contact below the chin and its neck are omitted. The upper arrow points downward into the workpiece and the lower arrow points upward into it.

The bottom contact has a broad, shallow flared lip of radius 18 mm, tapering back to its narrower neck. This is an illustrative cup-shaped contact outline; the underlying patch and force direction are unchanged, and no suction force is implied.

In both panels, a **red applied-force arrow** presses vertically downward onto actual green work-region face 213. All arrows act **on the workpiece**. The blue arrows represent **support force on workpiece**, along the actual inward normal at each contact center, matching `F_supp` in the demand equations. All contact heads and their necks are blue. No floor-force arrow is shown. Both panels preserve the original view, expressed as `[0.8, -1, 0.12]` in Z-up coordinates. Contact IDs are recorded in the source and metadata, not printed on the illustration.

The upper contact is the actual Step2 crown patch C023. Its outward center normal has a vertical component of about 0.93, so its support force on the workpiece points downward. An illustrative overhead arm connects it to the saved side frame. The existing bottom contact is C139; its saved Step5 contact material and necks are reused. The saved C011 head and neck are omitted; the rear joint remains part of the side frame.

For an unanchored support with negligible self-weight, a necessary vertical condition is that the sum of the head forces **on the workpiece** has a nonnegative vertical component. The workpiece's own floor contact is excluded. Arrow magnitudes here are illustrative and do not establish that inequality for a solved load case, or certify moment balance, bearing, or insertion.

The [symbolic derivation](../total_force/README.md) explains how workpiece moment balance can require a downward head reaction, then derives the support's necessary vertical condition. It includes a symbolic moment diagram and a [paper proposition and proof](../total_force/paper_argument.md); the planar explanation is separate from this unsolved 3-D illustration.

A [separate calculated case](../../baseline_algo/output/B/pose_2/step3.1_score_candidate/C023_downward_force_case.md) now uses this same C023 patch and red load point with a downward process force of magnitude `0.5mg`. Its workpiece equilibrium requires a downward head resultant; an example allocation has vertical head force `−1.2315mg` and workpiece floor force `+2.7315mg`. The simplified [moment-arm figure](../total_force/B_pose_2_C023_moments.png) shows one equivalent XZ support resultant **on the workpiece**, using the same force convention as the contact arrows in the conceptual illustration above. This calculation does not certify the depicted frame as a feasible support.

Regenerate with:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/obj_supp/demand/head_total_force.py
```

`head_total_force.json` records the source hashes, actual contact normals, overhead arm, rendered parts, camera, and applied-force point and direction. Each panel has one applied-force arrow in addition to its one/two support-force arrows. The previous connectivity certificate for the earlier geometry is deliberately not reused.
