# `sweep_demo` — does the support's swept volume hit the workpiece?

**Current procedure (2026-09-06): [problem_statement.md](../problem_statement.md#当前决定与讨论记录).** The robot holds the
workpiece at `T*`; one or more independent rigid supports are inserted sequentially,
each reaching its designed contacts with **perfect final fit**, before the robot
releases. Different pieces may use different insertion directions. Check each path
against the workpiece and already inserted pieces; no gravity-settle stage closes gaps.
This page illustrates one piece's corridor only: its measured **1.79 mm gap** means it
is not yet a realisation of that fitting load-bearing design.

A support is a rigid solid standing on the floor, slid into place along the floor in a
horizontal straight line `a`. What it sweeps on the way,

```
Sweep(supp, a) = { x − t·a : x ∈ supp, t ≥ 0 }
```

— the support and the whole corridor behind it along `a`, out to the edge of the scene —
must not enter the workpiece at its target pose. Two panels, one workpiece, one support
placement, two directions: one corridor cuts through the part (**blocked**), one is clear.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/trajectory/sweep_demo.py        # 5.7 s, of which big_tip.sweep("B") is 1.2
```

| file | md5 | px |
|---|---|---|
| `sweep_demo.png` | `c53264ca78b0688b18c7d0147f397c3e` | 1454 × 792 |

Deterministic: five consecutive runs, identical md5. `slides/setup/poses/tip_B.png` is rewritten by
`big_tip.sweep` on the way past and checked byte-identical before and after
(`5b3b5fd46687a60cc8afe27b60327316`). Nothing left under `objects/B/`.

Three revisions against the first cut (owner, all 2026-08-30): the support is a real little
fixture rather than a bare wedge; the corridor is the support's **own shape** extruded —
"sweep出的应该是支撑的形状，而不只是一个方块" — not one box round it; and the fixture is now a
**claw closed round the bunny's tail** rather than two pads under its haunch, because
"这个支撑只卡住B的尾部的，然后这样什么角度插入就很重要了" — a pad presses on a face and can be
withdrawn nearly anywhere, a claw round a lobe has an axis, and the corridor is what enforces
it. The change is worth 42 → **60 of 72** blocked bearings.

## The workpiece

Object **B at tip 1** of the slides, taken from the shared setup:
`big_tip.refine` (1000 → 4000 faces), `big_tip.sweep("B")[0]` for `T_star`, the work
region (`take`), the contact and the row's camera; rendered through `tip_sequence.render`
with the work region painted by `big_tip.paint_parts`. Part 155 × 153 × 119 mm, 735.2 cm³,
standing on one point at (11.9, −37.3, 0) mm, centre of mass at (13.6, −66.9, 61.9) mm.
The camera is the row's own direction (elev 12°, azim 19.5°) with the distance refit as
`tip_sequence.fit` refits it — on the part, the claw **and both cases' arrows**, so nothing of
the page is clipped.

## The support — a claw of sixteen convex pieces

Six **jaws** round the tail, a **hub** past the tail's tip carrying six **spokes** out to
them, a **column** up from the floor, a **brace** up its back and a base **slab**. Every
piece embeds ≥ 2 mm into its neighbour and the drawn solid is their exact union
(`manifold3d`), one watertight body of **33.00 cm³**, 63.8 × 62.5 × 64.8 mm across its
world box. Colour: teal, `capped_a1f.SHADE[4]` (`#33897d`) — a raw generator.

**The bore is not a shape anybody drew.** It is the workpiece's own silhouette along the
axis the claw slides in on, offset by `CLEAR` = 2 mm, and the six jaws sit on six
supporting lines of that polygon. It is *forced*, not chosen:

> a socket comes off along an axis only if its bore contains the silhouette of everything
> it holds — `r_bore(v) ≥ max_{w ≤ v} r_part(w)` — so the smallest removable bore is the
> constant one cut to the widest section, and there is no tighter claw that comes off at all.

Every millimetre of the fit is therefore the tail's and not the designer's. The measured
bore is **36.0 × 33.6 mm** across, centred (−58.4, 40.5) mm in the plane across the axis.

| | |
|---|---|
| axis | the tail's own — mesh `+x` under `T_star`, horizontal bearing **328.01°**; the claw goes on along **148.01°** |
| jaws | from **116.5** to **138.5 mm** along that axis; the tail's tip is at **130.5**, so the hub caps it with 4 mm to spare |
| fit | the held skin (5602 area-uniform samples beyond the mouth) stands **1.79 – 10.54 mm** off the claw: 1.79 at the seating shoulder, 10.54 round the tip, which is the taper |
| clash | claw ∩ part = 0 cm³, exact boolean; also every part vertex outside every piece's hull and every claw vertex outside the part |
| work region | 8.9 % of the faces, and the claw clears it by **15 mm** at its nearest — not far, and a fact about this pose rather than a margin anybody chose |

## The swept volume — exact, and the support's own shape

Translation distributes over a union, so the sweep of the claw is the union of its pieces'
sweeps; each piece is convex, and the sweep of a convex solid under a translation is the
convex hull of the solid and its translate (METHOD §4.6):

```
Sweep(supp, a) = ⋃ᵢ conv( Pᵢ ∪ (Pᵢ − L·a) ),      L = 1 m (six part widths, off the frame)
```

with the union an exact boolean (`manifold3d`, one watertight body; the far end is asserted
off the panel in both cases). The corridor's silhouette therefore carries six jaw tubes, the
spokes, the column and the slab — it reads as *this* support's corridor, not a box. Its
intersection with the part is an exact boolean too; the part is then drawn as its two exact
pieces — `part − corridor` in the page's own colours (each face painted work / spare by the
original face it lies on, via `closest_point`) and `part ∩ corridor` in red — so the red is
the boolean and nothing overlaps (no z-fighting). The two pieces add up to the part to ~4e-9
relative (manifold's floating point; the assert allows 1e-7).

One render-only nuance, `SKIN_EPS`: `conv(P ∪ (P − L·a))` keeps `P`'s own leading faces, so
the corridor's boundary coincides with the support's over the whole leading side, and two
coincident faces z-fight (a comb of stripes on the fixture, seen 2026-08-30). The
translucent solid handed to the renderer is therefore inflated **0.25 mm** radially per
piece — under a pixel here; every boolean and every printed number uses the exact corridor.

## The two directions, and the window

`a` is the direction the support **moves** in; the corridor lies behind it along `−a`.

| panel | `a` (compass) | corridor toward | part inside | red skin (facing the eye) |
|---|---|---|---|---|
| **blocked** | **280°** | 100° — in through the flank and out past the haunch | **42.44 cm³** (5.8 % of the part) | 45.6 cm² (41.0) |
| **clear** | **148°** — the tail's own axis | 328° — off along the tail, over open floor | **0 cm³** | 0 |

Every 5° of the compass is priced (cm³ of the part inside the exact corridor):

```
  0:64.8    5:50.3   10:38.0   15:27.4   20:19.9   25:14.7   30:10.7   35: 7.6   40: 5.6
 45: 4.3   50: 3.5   55: 2.8   60: 2.3   65: 1.8   70: 1.5   75: 1.1   80: 0.8   85: 0.6
 90: 0.4   95: 0.3  100: 0.2  105: 0.1  110: 0.1  115–190: 0.0  195: 0.1  200: 0.1
205: 0.3  210: 0.5  215: 0.7  220: 1.1  225: 1.5  230: 1.9  235: 2.5  240: 3.1  245: 4.0
250: 5.4  255: 7.6  260:11.0  265:15.6  270:21.4  275:29.9  280:42.4  285:58.1  290:76.2
295:95.9  300:118.2 305:141.8 310:162.2 315:174.8 320:181.3 325:177.6 330:166.9 335:157.3
340:141.8 345:121.2 350:100.7 355:81.7
```

**Blocked on 60 of 72 bearings** — and the sixteen that print `0.0` are not the twelve that
are clear: four of them are slivers of a few hundredths of a cm³ that round to zero at one
decimal. So the window's two edges are walked in at **1°** rather than read off this table:

> **clear from 123° to 181° — 58° of the 360, centred 152°.** The tail's own axis is 148.

`280°` was picked for the blocked panel from a per-bearing scan of *visible* red skin among
the bearings under 70 cm³: 41.0 of its 45.6 cm² face the eye, the most in that band, and the
corridor visibly enters at the flank and leaves beyond the haunch. The bearings that cut
more (up to 181 cm³ at 320°) bury most of what they cut on the far side.

### Why the window is 58° and not 13°

The clearance alone would give an escape cone of `atan(CLEAR/DEPTH)` = **6.3°** off the
axis. The rest is the tail's own shape:

> the tail **tapers** — 18.9 mm of radius at the jaws' mouth down to 12.7 at its tip over
> 14.0 mm, a half angle of **23.8°** — and a bore that has to swallow the widest section is
> loose by that much everywhere further in.

23.8 + 6.3 = **30.1°**, against a measured half window of **29°**. Both numbers print on
every run. The consequence is that the window is a fact about the workpiece and not a knob:
shorten the jaws and it widens (**88°** at `DEPTH` 13 mm, because the claw grips only the
thin end), lengthen them and the bore has to swallow the haunch and it widens again (**72°**
at 26 mm, bore 48 × 62). 18 mm is the floor of that curve — measured, four depths.

## The drawing

- Two 700 px panels, one camera, no text on the panels; one muted line under each
  (`blocked · a at 280° · 42.4 cm³ inside the workpiece` / `clear · a at 148° · 0 cm³, and
  so is 58° of the 360`), the shared `cover.py` page colour and ink, and an 18 px gap.
- The corridor: one translucent watertight mesh, **alpha 0.35**, a **greyed teal** `#8fb5b0`
  (see *Known issues* for why not `SHADE[2]`).
- The intersection: opaque red `#B02A26` (METHOD §3.0's "what is owed"), with **emission
  1.0** on the material — without it the red read as brown through the corridor.
- The arrow: an ink mesh arrow (`skin.py`'s cylinder-and-cone proportions), 40 mm long,
  lying on the floor beside the corridor on the eye's side, just clear of the claw's own
  silhouette across `a`, pointing along `a`. The side is chosen off the row's azimuth alone
  so both arrows exist before the camera is fitted and are fitted into the frame. Asserted
  outside the corridor and the part.
- Renderer: **mujoco**, through `tip_sequence.render`, with the four materials (`supp`,
  `sweep`, `owed`, `ink`) patched into `T.SCENE` for the call and restored — `skin.shot`'s
  trick, including render-until-two-agree (0 extra renders on every recorded run). The
  translucent geom goes last in the parts list. Matplotlib was not needed.

## Known issues

- **The claw hides what it holds.** The eye stands 66° off the claw's axis, on the far side
  of the fixture from the tail, so the tail is inside the jaws and mostly out of sight; the
  jaw gaps were widened to 5 mm (`JAW_GAP`) and the back opened from a solid plate to a hub
  and six spokes precisely to let some of it show. It still reads as a gripper on the
  bunny's rear rather than as a gripper on a *tail*.
- **This corridor example has not been tested for equilibrium or printability.**
  A finished design must satisfy the workpiece force/torque equations and the assembly's
  floor-hull check, as stated in `setup/equations/equations.md`; this example supplies
  no such verdict.
- **The bore is read off a sample.** `SAMPLES` = 200 000 area-uniform surface points at a
  fixed seed, because a 4000-face part's *vertices* are not its outline. The silhouette hull
  is then buffered by `CLEAR` and simplified to `SIMPLIFY` = 0.4 mm: a rounded offset of a
  50-vertex hull is 500 points, and `conv(P ∪ (P − L·a))` on that comes back degenerate and
  `manifold3d` refuses the union ("Not all meshes are volumes", seen 2026-08-30).
- **The red's emission is a rendering choice**, not a material of the page family; nothing
  else on the slides glows.
- **The corridor's colour is not a `SHADE` entry.** `SHADE[2]` at alpha 0.35 turned the red
  under it salmon and a plain grey vanished into the 0.80 grey floor; the teal pulled
  halfway to grey was the compromise (three renders compared, 2026-08-30).
- **The claw does not touch**: 1.79 mm at its nearest. Stated, not fixed — this page is about
  the corridor, not the contact.
- **The claw's clearance from the work region is 15 mm**, which is close for a support that
  is meant to stay off `W` (METHOD §2). It is not enforced anywhere here; it is printed.
- **`big_tip.sweep("B")` runs both of B's tips** (and re-renders `tip_B.png`) to produce the
  one row this page uses; 1.2 of the 5.7 s.
- The blocked-case bunny is manifold's re-triangulation of the part minus the corridor (4206
  faces against 4000); shading differences against the clear panel's untouched part are
  confined to the cut's neighbourhood and were not visible side by side.
