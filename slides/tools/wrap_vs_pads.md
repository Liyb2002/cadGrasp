# wrap_vs_pads — wrapping a convex edge buys nothing

`python slides/tools/wrap_vs_pads.py  ->  slides/tools/figures/wrap_vs_pads.png`  (3297 × 3360,
`e43092b376537e1d8ad333f6973be1a1`, 1.6 s)

A cube lying flat, isometric. At one visible corner, two support designs: **(a)** a
bracket that wraps the edge, **(b)** two separate pads, one on each face, touching the
same point. Under each, the set of force directions it can push along and the set of
torques it can turn the cube about.

**They are the same set, in both spaces**, and the figure shows it rather than saying it:
the two designs' fans are drawn over each other and coincide.

| | opening | the two designs agree to |
|---|---|---|
| force | 90.0000° | `0.0e+00` |
| torque | 143.1301° | `1.2e-16` |

`n_A = (−1, 0, 0)`, `n_B = (0, +1, 0)`; `m_A = (0, −0.25, −0.50)`,
`m_B = (−0.25, 0, +0.50)`; arm 0.5590 L, `|p − c| = 0.7500 L`.

## Why, in two steps

1. **Force.** A frictionless contact pushes along the surface's inward normal, and at a
   CONVEX EDGE the normal cone is exactly the fan between the two face normals. That fan
   is precisely the non-negative combinations of `n_A` and `n_B`, which is what the two
   pads already span. The wrap adds no direction because there is no direction to add.
2. **Torque.** At a fixed contact `p` the map `u ↦ (p − c) × u` is **LINEAR**, and a
   linear map carries a cone to a cone. So the torque set is the fan between `m_A` and
   `m_B` — the same conclusion, one level up, for the same reason.

## The decisions that look free and are not

**Both designs contact the SAME POINT, and the torque half of the claim depends on it.**
Force needs only the same directions; torque needs the same positions as well. The run
measures the difference rather than warning about it: slide each pad 0.30 L along its own
face, off the edge, and the **force cone does not move** (residual `0.0e+00`) while the
**torque set tilts 18.20° out of the `m_A, m_B` plane** and the four generators go from
rank 2 to rank 3. That is METHOD §11.1 — a contact is a point *and* a direction — in one
number.

**Frictionless throughout**, which is METHOD §1's model; §5 lists friction as not
modelled, and the `mu = 0.5` in the record is Passive Grippers' (§8), not this contact
model's. **Nothing here is claimed under friction** — with it a wrap is not equivalent,
because it can carry tangential load the pads cannot.

## Known issues

- 3297 × 3360 is large and nearly square: two 3-D scenes over two fan panels, each with
  its own key. It reads, but a tighter layout exists.
- The two fans are drawn in the plane they live in, which is the plane square to the edge.
  That is honest for this corner and would not survive a non-convex edge, where the normal
  cone is not a fan at all.
