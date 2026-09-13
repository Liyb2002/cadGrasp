# projection_right_angle.png — what it argues, and what not to break

Handoff for `slides/tools/projection_right_angle.py` and the one PNG it writes into `slides/tools/figures/`.
Every number below was printed by a run on 2026-08-14, not remembered; the script computes
all of them, including the ones drawn on the figure.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/tools/projection_right_angle.py     # < 2 s, one PNG, prints every number
```

There is no randomness anywhere in it, and two clean runs were compared byte for byte:
`projection_right_angle.png`, **2814 × 1365**, md5 **`8b25f9b4ae8b16f565007d10557ff66b`**.
The script writes nothing else. `dimension_flat.png` (`42b00980c594ab1c83ed3ead6f6a9491`)
and `normals.png` (`9a82bdb6f7d5d66197e14492a6ab2248`) were checked unchanged after every
run here.


## The argument

This is the clean textbook case and it exists to be leaned on: two contacts pushing exactly
**90° apart**, and a force asked for along their **45° bisector**. Two panels, one scale.

**Panel 1 — what it costs.** The demand is one unit up the bisector. The parallelogram that
builds it out of the two pushes is, at this angle, a **square**, and its two sides are the
components: `cos 45° = 0.7071` each. So

```
0.7071 + 0.7071 = 1.4142 of push, for 1.0000 delivered
```

The two components sum to **more than the resultant they produce**. That is the triangle
inequality, and it is the price of spreading a demand over two supports rather than putting
it on one.

**Panel 2 — what it can do.** Now cap each support at `F` and ask how hard the pair can push.
Everything they can supply is `a·u₁ + b·u₂` with `0 ≤ a, b ≤ F`, and that set is exactly a
**square standing on the origin**. So the answer for every direction at once can be read off
one picture: the reachable frontier is the square's two far edges, the far **corner sits on
the bisector at √2 F = 1.4142 F**, and the only two directions where the square touches the
circle of radius `F` are **the push directions themselves, at exactly F**. The bisector is
the pair's strongest direction; the pushes are its weakest.

**Those two facts are not in tension, and making that obvious is the point of the figure.**
Spreading one unit over both supports

- raises the **total** the supports pay, `1.0000 → 1.4142`, and
- halves the **largest single share**, `1.0000 → 0.7071`.

The bill is the sum. A per-support cap binds on the **max**. They are different functionals
of the same split, they move in opposite directions, and so the same 45° direction is at
once the most expensive one to serve and the one the pair can push hardest in. Equivalently:
each support gives 0.7071 per unit delivered, so its cap is not reached until the delivered
force is `1 / 0.7071 = 1.4142 F`. **The 1.4142 in the two panels is one number read twice.**


## The numbers this run printed

```
u1 (+0.7071,+0.7071)   u2 (-0.7071,+0.7071)   u1 . u2 = -1.015e-17

a unit along the bisector
  components                 a = 0.7071   b = 0.7071      (cos 45 deg = 0.7071)
  they sum in magnitude to   1.4142       against the 1.0000 they deliver
  the largest single share   0.7071       down from the 1.0000 one support carries alone

capped at F = 1.0000 each
  most     1.4142 F at 45.0 deg off u1 -- the bisector,  sqrt(2) = 1.4142
  least    1.0000 F at 0.0 and 90.0 deg off u1 -- the push directions themselves
  support spent per unit delivered (a + b):  1.0000 at the pushes, 1.4142 at the bisector
  the cap binds at 1 / 0.7071 = 1.4142 F
```

Across the octant between the two pushes, cost and reach in one table:

```
 phi off u1      a       b      total paid    capped reach
    0.0 deg    1.0000  0.0000     1.0000        1.0000 F      a push itself
   22.5        0.9239  0.3827     1.3066        1.0824 F
   45.0        0.7071  0.7071     1.4142        1.4142 F      the bisector
   67.5        0.3827  0.9239     1.3066        1.0824 F
   90.0        0.0000  1.0000     1.0000        1.0000 F      the other push
```

Cost and reach reach their maximum at the same place, and their minimum at the same place.
That is the whole reconciliation in five rows: **the evenness that makes a direction
expensive is the same evenness that makes it strong.**

This agrees with `dimension.md`'s own two-push formula, `a = sin(t−φ)/sin t`,
`b = sin(φ)/sin t`, which at `t = 90°` degenerates to `a = cos φ`, `b = sin φ` — the dot
products. Checked at φ = 0, 22.5, 45, 67.5, 90: the formula, the linear solve and the dot
products agree to every digit printed, and the reach column reproduces that file's 90° row
(`1.4142 F` at the middle, `1.0000 F` weakest, at `φ = 0`, a push itself).


## The check that makes this the reference panel

**At 90°, and at no other angle, the orthogonal projection is the decomposition.** The dot
product `uᵢ · f` asks each push separately how much of `f` lies along it and never checks
that the two answers add back up to `f`. Solving `[u₁ u₂] c = f` asks what the contacts must
actually supply. When the pushes are orthonormal these coincide, and the figure draws the
coincidence: **the perpendicular dropped from the tip of the resultant lands exactly on the
end of the component arrow**, which is why the dashed lines in panel 1 are simultaneously
the square's far sides and the two dropped perpendiculars. Both right-angle marks sit at
that foot, because that is the one place in the picture where the right angle is a claim
rather than just the arrangement of the pushes.

Measured, on the bisector:

```
solve  a = 0.7071067811865475   b = 0.7071067811865475
dot    a = 0.7071067811865475   b = 0.7071067811865476
```

and over **901 target directions** across the octant the largest disagreement between the
two is **2.220e-16** — floating-point noise, nothing else.

**And it fails immediately off 90°**, which is why the intuition everyone has from this
panel is the thing that breaks in the other projection figures:

```
 pushes apart   solve            dot              gap       the dot answer builds
   60 deg       0.5774 0.5774    0.8660 0.8660    0.2887    1.5000 of what was asked
   90 deg       0.7071 0.7071    0.7071 0.7071    0.0000    1.0000  -- they agree
  120 deg       1.0000 1.0000    0.5000 0.5000    0.5000    0.5000 of what was asked
```

At 60° the intuitive answer **over-delivers by half**; at 120° it supplies **half** of what
was asked. Neither error is small, and neither is visible from inside the right-angle case.
That is the sense in which this panel is a special case and not a general lesson.


## Decisions that look free and are not

**The two panels share one pair of limits**, `x ∈ [−1.34, 1.34]`, `y ∈ [−0.66, 1.56]`, with
`aspect="equal"`. That is the argument, not the styling: panel 1's resultant is 1.0000 long
and panel 2's is 1.4142 long, and at one scale the reader *sees* the difference instead of
comparing two numbers. It is also why panel 1 has dead space above it — the space is the
1.4142 that panel 2 uses and panel 1 does not. The drawing-convention note is parked in that
dead space rather than being given furniture of its own.

**Magnitude is encoded twice, in length and in width**, `lw = 1.00 + 3.30·m` points:

```
0.7071 -> 3.33 pt      1.0000 -> 4.30 pt      1.4142 -> 5.67 pt
```

Linear and monotone, so reading widths is reading magnitudes. A factor of 2 in magnitude
comes out as a factor of 1.7 in width, which is enough to be unmistakable without the thin
arrows going to hairline. **The arrowhead is scaled with it too**, `14.0 + 8.5·m`: a fixed
head on a 5.67 pt shaft ends the arrow in a point narrower than itself and reads as a stub.

**`FancyArrowPatch` directly, not `annotate`.** `annotate` sets the patch's `mutation_scale`
from the *text* size, so the head would track the font and not the force. Everything else in
the house style is kept — Agg, dpi 210, `PAPER` ground, `INK`/`MUTED` text, **`ORANGE` for
what a contact supplies and `BLUE` for what is demanded of it**, which is the same reading as
`dimension.py`.

**Panel 2's set is filled and panel 1's square is dashed**, and the distinction carries
meaning: a **fill is a set of forces**, dashes are a **construction**. The frontier — the two
far edges — is drawn in `INK` at 2.0 pt because it is the answer, while the circle of radius
`F` is dashed and muted because it is only a foil. The circle is what "capped at F" would
mean if a cap meant `F` in every direction; the square bulges past it everywhere except at
the two push directions, where it touches. Remove the circle and "strongest" and "weakest"
become assertions in the caption instead of things in the picture.

**The bisector stands vertical** (`HALF = 45°`, pushes at 45° and 135°). The figure is then
mirror-symmetric about the page's vertical, so the reader sees that the two components are
equal rather than reading two identical numbers.

**Labels sit `LAB = 0.20` to the right of an arrow tip, not hard against it.** At both
resultant tips a 45° construction edge leaves the same point, and a two-line label hung
directly off the tip has its second line struck through by that edge. This was the first
pass and it looked wrong.


## Known open issues

- **`HALF = 45.0` is a free parameter for the geometry and not for the text.** The angle
  marks, the panel title and the "only at 90°" caption are derived from it, but the `√2 F`
  label and the whole of the argument are the 45° case written longhand. Moving `HALF`
  produces a correct drawing with two wrong strings on it.
- **The output path is absolute and hard-coded** to `/Users/yuanboli/.../figures/`, the same
  as `dimension.py`. The script cannot be run from a clone elsewhere without editing.
- **`F` is drawn at 1.0000 and so is indistinguishable from a unit demand.** Panel 2's cap
  and panel 1's delivered force are both 1.0000, which is what makes the two panels
  comparable, but it means the figure cannot be read to say whether `1.4142` is `√2 F` or
  `√2` — the labels carry the `F` and the arrows cannot.
- **Nothing here is a support *arrangement*.** There is no body, no contact point and no
  floor: position never enters, exactly as in `dimension.py`'s model, so the panel is about
  two directions and a cap and nothing else. Anyone wanting the same statement about a real
  part has to bring the arrangement with them.
- **The reachable square is the 2-D case.** In space, two capped pushes give a parallelogram
  in a plane, and the third generator that `dimension.py` cares about is not in this picture
  at all.
