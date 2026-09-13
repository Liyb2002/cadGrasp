# The heads' total force

![The heads' total force](head_total_force.png)

This is a conceptual comparison using the B / pose 2 workpiece and contact geometry. Both panels show the ground plane but omit the ground-base polygon, floor connector, and floor-force arrow. The saved side frame still connects the depicted heads.

- **Panel (a):** The upper orange contact sits on the crown of the head, with an arm that approaches from above and connects to the side frame. The workpiece exerts an upward force on that contact; both lower blue contacts are absent.
- **Panel (b):** The existing bottom blue contact is retained, and a second blue contact is added below the chin. Its contact patch is the actual Step2 non-work-region candidate C151. A thick blue bent neck joins this added contact to the side frame. The three arrows show the workpiece pushing the upper contact up and both lower contacts down.

Both blue contact ends have broad, shallow flared lips, tapering back to their narrower necks. The lower lip has an 18 mm radius and the chin lip a 14 mm radius. These are illustrative cup-shaped contact outlines; the underlying patches and the force directions are unchanged, and no suction force is implied.

In both panels, a **red applied-force arrow** presses vertically downward onto actual green work-region face 213. It illustrates the external loading that tends to rock the workpiece and lift the upper contact. This arrow acts **on the workpiece**. The orange and blue arrows represent **workpiece force on support**, along the actual outward normal at each contact center. No floor-force arrow is shown. Both panels preserve the original view, now expressed as `[0.8, 0.12, -1]` in Y-up coordinates. Contact IDs are recorded in the source and metadata, not printed on the illustration.

The upper contact is the actual Step2 crown patch C023, replacing the previous C024 contact. Its outward center normal has a vertical component of about 0.93, making the opposite force on the workpiece clearly downward. A new illustrative overhead arm connects it to the saved side frame. The existing bottom contact is C139; its saved Step5 contact material and necks are reused. The saved C011 head and neck are omitted; the rear joint remains part of the side frame. The new C151 head uses actual Step2 contact material, but its blue neck is an illustrative connection, not a new Step5 solution.

For an unanchored support with negligible self-weight, a necessary vertical condition is that the sum of the workpiece forces **on the heads** has a nonpositive vertical component. The workpiece's own floor contact is excluded. Arrow magnitudes here are illustrative and do not establish that inequality for a solved load case, or certify moment balance, bearing, or insertion.

Regenerate with:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/obj_supp/demand/head_total_force.py
```

`head_total_force.json` records the source hashes, actual contact normals, new overhead arm, added blue neck, rendered parts, camera, and applied-force point and direction. Each panel has one applied-force arrow in addition to its one/three contact-reaction arrows. The previous connectivity certificate for the earlier geometry is deliberately not reused.
