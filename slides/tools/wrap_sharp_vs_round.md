# wrap_sharp_vs_round — a fillet is a sharp corner at its centre of curvature

`python slides/tools/wrap_sharp_vs_round.py  ->  slides/tools/figures/wrap_sharp_vs_round.png`  (3324 × 3360,
`60e3607552190748b6f54d54772677b2`, 1.7 s)

The direct companion to `wrap_vs_pads` and its format exactly: two 3-D scenes over a force
fan and a torque fan, numbers under each, the claim in the footer. Same cube, same camera,
same bracket, same contact height — the only thing that changes is the edge it wraps.
**(a)** a sharp right angle, **(b)** the same edge filleted.

| | opening, sharp | opening, round | the two |
|---|---|---|---|
| force | 90.0000° | 90.0000° | identical, residual `0.0e+00`, **at every radius** |
| torque | 143.1301° | 134.2605° (drawn r) | the same construction from a point `r√2` away |

## Why, in two steps

1. **Force — nothing happens.** A fillet's outward normals sweep continuously from `n_A` to
   `n_B` and stop there. That is precisely the fan the sharp edge's normal cone already is:
   nothing added, nothing missing, and the radius never enters. Measured equal as sets to
   `0.0e+00`.
2. **Torque — the fillet is a SHARP CORNER at `q`.** A point on the fillet is `p = q + r n`
   with `q` the centre of curvature, and the push there is `u = −n`, so

   ```
   τ = (p − c) × u = (q − r u − c) × u = (q − c) × u − r (u × u) = (q − c) × u
   ```

   The `r (u × u)` term **vanishes identically** — measured `5.6e-17`. The radius does not
   enter the torque either; only `q` does. So the rounded edge's torque cone is a sharp
   corner's cone (residual `5.7e-17`), just taken from a different point.

And that is the honest part: for a 90° edge `q` sits **`r√2` inside the vertex**, so the two
cones are the same construction from two points `r√2` apart. They agree exactly along one
ray — the image of the fan's bisector, because `vertex − q` is parallel to it, residual
`0.0e+00` — and part company towards the ends, linearly in `r`.

| radius | `q` back by | `\|q − c\|` | torque opening | worst axis off the sharp set | `\|τ\|` falls |
|---|---|---|---|---|---|
| sharp | 0 | 0.7500 L | 143.1301° | — | — |
| **r = 0.0158 L — A1-f's REAL fillet** | 0.0223 L | 0.7290 L | 142.1420° | **0.55°** | **2.5 %** |
| r = 0.02 L | 0.0283 L | 0.7234 L | 141.8704° | 0.71° | 3.2 % |
| r = 0.05 L | 0.0707 L | 0.6837 L | 139.8315° | 1.86° | 7.9 % |
| **r = 0.12 L — the DRAWN fillet** | 0.1697 L | 0.5927 L | 134.2605° | **5.05°** | **18.6 %** |
| r = 0.20 L | 0.2828 L | 0.4924 L | 126.1686° | 9.83° | 30.1 % |

So "a rounded corner is the same as a sharp one" is true in the limit, and the figure says
how true rather than asserting it: at A1-f's real fillet, half a degree and two and a half
per cent.

## The decisions that look free and are not

**The drawn fillet is eight times life size, and the page says so twice.** `dimension.py`'s
`normals()` hit this first — *"the fillet is drawn far fatter than the real one … because at
true scale the fan would be a single pixel"* — and this camera makes it worse, since the near
edge is seen so obliquely that even a tenth of a side barely bends the silhouette. So the
fillet is drawn at `r = 0.12 L` and **every number is reported at BOTH radii**, drawn and
real, in the footer and in the table. Reading the drawn 5.05° and 18.6 % as the real numbers
is the one mistake this figure could cause, and the double reporting is the whole of the
defence against it.

**Not fatter still, because `q` climbs the page as `r` grows.** Past about `0.13 L` it
collides with the centre-of-mass roundel, which sits at the same page abscissa. `0.12 L` is
the widest fillet that leaves the two legible apart.

**The two torque cones are drawn about a shared bisector at the same radius, and that is
exact, not a convenience** — they really do agree along that one ray. What the panel cannot
show is that the round cone leaves the sharp one's plane, so it is *not* a sub-cone however
much the picture nests them. The `5.05° out of this plane` in the panel's corner is that
correction, and it is the number the table's "worst axis off" column carries.

**No mark for the sharp vertex in panel (b).** `q`, the middle contact and the vertex are
collinear along the fan's bisector, so anything drawn at the vertex lands under the middle
arrow. The `r√2` shift shows instead by comparing the two panels' arms: (a)'s runs out to
the corner, (b)'s stops short at `q`.

**Frictionless throughout**, METHOD §1's model. Under friction a fillet is not equivalent to
a sharp edge — the contact patch and the pressure distribution start to matter — and nothing
here is claimed under it.

## What is reused

`wrap_vs_pads.py` is **imported**, not copied, so the two figures cannot drift: the palette,
the camera and light, the cube, the contact point `P`, the bracket section, the page size,
the fan-panel geometry, and `unit / between / frame / fan / moment / outside / agree / prism
/ faces / shaded / outline / cast / veil / Arrow3D / disc / at / trim`. Only two functions
are forked, because both hard-code one body and one cone: `scene()` gains a body, a contact
set and the inward fan to `q`; `fanel()` gains a second opening. This is the only
cross-import between figure scripts in `slides/tools/`.

## Known issues

- 3324 × 3360 inherits the sibling's known issue: large and nearly square.
- The **cube's own** fillet is the weakest thing on the page. Only the quarter of it above
  the bracket is unoccluded, and the `c` roundel and the `q` disc sit on most of that. What
  actually reads as "this edge is round" is the bracket's mating inner corner, the contacts
  spread along an arc, and the pushes converging on `q`.
- The ladder is six radii of one geometry. The `r√2` is the 90° case; a general dihedral
  puts `q` at `r / sin(θ/2)` along the bisector and the figure does not draw it.
