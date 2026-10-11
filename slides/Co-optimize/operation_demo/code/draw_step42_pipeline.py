"""Render the current whole-set Step4.2 operation selection as one PNG.

Flow follows stable_gradient / loss_first_gradient / progressive_gradient,
and the subsequent feasible XYZ material descent. No search is executed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "step4_2_pipeline.png"
INK = "#17212d"
MUTED = "#526171"
BLUE = "#1768a4"
TEAL = "#147c80"
ORANGE = "#b95821"
GREEN = "#347748"

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "stix"})
fig = plt.figure(figsize=(16, 19), dpi=170, facecolor="white")
checks = []
card_checks = []


def text(x, y, value, size=16, bold=False, color=INK, align="center", rotation=0):
    artist = fig.text(x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
                      ha=align, va="center", color=color, rotation=rotation,
                      linespacing=1.45, zorder=5)
    checks.append(artist)
    return artist


def box(x, y, width, height, color="#abb9c8", fill="#f8fafc"):
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.002,rounding_size=0.009",
                          transform=fig.transFigure, facecolor=fill,
                          edgecolor=color, linewidth=1.2, zorder=2)
    fig.add_artist(patch)
    return (x-width/2, y-height/2, x+width/2, y+height/2)


def inside(bounds, artist):
    card_checks.append((bounds, artist))


def arrow(points, color=MUTED, width=1.35):
    for start, end in zip(points[:-2], points[1:-1]):
        fig.add_artist(plt.Line2D([start[0], end[0]], [start[1], end[1]],
                                 transform=fig.transFigure, color=color,
                                 linewidth=width, zorder=1))
    fig.add_artist(FancyArrowPatch(points[-2], points[-1], transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=15,
                                  linewidth=width, color=color, zorder=1,
                                  shrinkA=0, shrinkB=1))


def diamond(x, y, width, height, title, size=17):
    fig.add_artist(Polygon([(x,y+height/2), (x+width/2,y),
                            (x,y-height/2), (x-width/2,y)],
                           closed=True, transform=fig.transFigure,
                           facecolor="white", edgecolor=MUTED,
                           linewidth=1.25, zorder=2))
    text(x, y, title, size, bold=True)


text(.05, .969, "Step4.2   /   Whole-set operation selection", 29, True, align="left")
text(.05, .939, "Satisfy every pose, then minimize the shared support's material volume.",
     19, align="left", color=MUTED)

main_x, main_w = .34, .57
b = box(main_x, .877, main_w, .082)
inside(b, text(main_x, .898, "Start from saved Step4.1", 21, True))
inside(b, text(main_x, .872, "All poses registered for rotating reuse;", 16))
inside(b, text(main_x, .85, "common / similar exit directions.", 16))
text(.804, .892, "One rigid fixture", 20, True, color=BLUE)
text(.804, .865, "Rotate the empty fixture\nbetween tasks; one state\ncan serve many poses.", 16, color=MUTED)
arrow([(main_x,.836), (main_x,.812)])

b = box(main_x, .775, main_w, .075)
inside(b, text(main_x, .802, "One ALL-pose objective", 20, True))
inside(b, text(main_x, .773,
               r"$\widehat D_{\mathrm{all}}=\frac{1}{2K}\sum_k\frac{1}{s_k^2}"
               r"\sum_\ell\omega_{k\ell}D_k$", 23))
inside(b, text(main_x, .748, "Same nodes / weights for a local block and its branches.", 14))
arrow([(main_x,.7375), (main_x,.712)])

container = box(main_x, .646, main_w, .132, BLUE, "#f5f9fc")
inside(container, text(main_x, .697, "Continuous repair: gains AND losses", 18, True, color=BLUE))
left = box(.195, .635, .257, .096, BLUE, "white")
inside(left, text(.195, .667, "DIRECTION", 18, True, color=BLUE))
inside(left, text(.195, .643, "Sphere-tangent gradient", 15))
inside(left, text(.195, .621, "Independent / joint / coherent", 13.5))
inside(left, text(.195, .601, "Several angular step sizes", 14))
right = box(.485, .635, .257, .096, TEAL, "white")
inside(right, text(.485, .667, "TRANSLATION", 18, True, color=TEAL))
inside(right, text(.485, .643, "World XYZ gradient", 15))
inside(right, text(.485, .621, "Only explicitly seated poses", 14))
inside(right, text(.485, .601, "Several steps; floor projection", 13.5))
arrow([(main_x,.58), (main_x,.558)])

b = box(main_x, .517, main_w, .082)
inside(b, text(main_x, .544, "Choose a tool AND its step size", 19, True))
inside(b, text(main_x, .519, "Rank within gradient step groups by whole-set loss.", 15))
inside(b, text(main_x, .495, "Try moderate → smaller → larger; accept early.", 15))
inside(b, text(main_x, .481, "If gradients stall: bounded contact-event samples.", 13.5, color=MUTED))
arrow([(main_x,.476), (main_x,.451)])

b = box(main_x, .409, main_w, .084)
inside(b, text(main_x, .435, "Check shortlisted changed layouts", 19, True))
inside(b, text(main_x, .409, "Every original 32,768 force–torque demand / pose.", 15))
inside(b, text(main_x, .385, "Commit net whole-loss progress; a full PASS ranks first.", 14))
arrow([(main_x,.367), (main_x,.348)])
diamond(main_x, .304, .29, .088, "ALL demands\npass?", 17)

arrow([(main_x,.26), (main_x,.235)], GREEN)
text(.381, .248, "YES", 13, True, color=GREEN)
b = box(main_x, .173, main_w, .124, GREEN, "#f6faf6")
inside(b, text(main_x, .214, "Keep PASS; reduce material", 20, True, color=GREEN))
inside(b, text(main_x, .187, "Try restoring rotating reuse where possible.", 16))
inside(b, text(main_x, .161, "Direction + XYZ material descent, several steps.", 15))
inside(b, text(main_x, .134, "Accept only smaller material with ALL loads still passing.", 13.5))
arrow([(main_x,.111), (main_x,.091)], GREEN)
b = box(main_x, .066, main_w, .05, GREEN, "white")
inside(b, text(main_x, .077, "Export the fixed-layout mesh; report real material volume.", 14))
inside(b, text(main_x, .055, "Reuse saved force PASS; keep a smaller feasible incumbent.", 13.5))

arrow([(.485,.304), (.655,.304)])
text(.568, .32, "NO", 13, True)
diamond(.805, .304, .3, .096, "Local progress\n& budget left?", 16)
# Continue locally only while an accepted step and local-round budget remain.
arrow([(.955,.304), (.986,.304), (.986,.726), (.485,.726), (.485,.712)], BLUE)
text(.971, .579, "YES  •  continue local descent", 13, color=BLUE, rotation=90)
arrow([(.805,.352), (.805,.431)], ORANGE)
text(.805, .392, "NO  •  local repair unresolved", 13, True, color=ORANGE).set_bbox(
    dict(facecolor="white", edgecolor="none", pad=3))

b = box(.805, .559, .3, .256, ORANGE, "#fff9f4")
inside(b, text(.805, .663, "JUXTAPOSE", 21, True, color=ORANGE))
inside(b, text(.805, .635, "Discrete jump / new seat", 16, True))
inside(b, text(.805, .605, "Choose a failing pose OR blocker;", 14))
inside(b, text(.805, .582, "try hosts with similar exit directions.", 13.5))
inside(b, text(.805, .555, "Seat / direction seeds; paired jumps", 13.5))
inside(b, text(.805, .533, "can release jointly blocked contacts.", 13.5))
inside(b, text(.805, .507, "≤96 cheap screens → ≤3 branches.", 14))
inside(b, text(.805, .482, "Refine each with Direction + XYZ.", 14, True))
inside(b, text(.805, .455, "Same parent measure; keep the best gain.", 13))
arrow([(.655,.647), (.625,.647)], ORANGE)
text(.8, .179, "Jump accepted → resume repair.\nNo improving branch → stop this budget;\nunresolved sets may continue from\ntheir own current layout.", 14, color=MUTED)
text(.8, .077, "Fast candidates: cached contact / material\ndeltas and valid projection-basis reuse.\nSearch candidates do not rebuild Boolean.", 13.5, color=MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist in checks:
    bounds = artist.get_window_extent(renderer)
    if bounds.x0 < .005*fig.bbox.width or bounds.x1 > .995*fig.bbox.width or bounds.y0 < 0 or bounds.y1 > fig.bbox.height:
        raise RuntimeError("Text outside image: " + artist.get_text())
for (x0,y0,x1,y1), artist in card_checks:
    bounds = artist.get_window_extent(renderer)
    if bounds.x0 < x0*fig.bbox.width or bounds.x1 > x1*fig.bbox.width or bounds.y0 < y0*fig.bbox.height or bounds.y1 > y1*fig.bbox.height:
        raise RuntimeError("Text outside its card: " + artist.get_text())
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", metadata={
    "Title": "Step4.2 whole-set operation-selection pipeline",
    "Description": "Multiscale whole-demand Direction / XYZ descent, bounded event sampling, discrete Juxtapose and feasible material reduction.",
})
plt.close(fig)
print(OUT)
