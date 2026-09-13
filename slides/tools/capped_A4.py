"""capped_cuboid.py's model, unchanged, with A4's contact numbers pulled apart.

A4 is a 150 x 150 mm frame only 30 mm thick, so two supports pressing on the two
faces of one rib stand 30 mm apart in space and land on top of each other on
screen. Measured on the figures this file replaces: the closest pair of contact
numbers sits 5.6 px apart on pose 2 and 9.5 px on pose 0, in a 46 px font. The
digit underneath is simply gone -- pose 0 step +5 reads "1 3 4 5", so a reader
counts four supports where there are five. On cuboid_baseline, whose faces are
150 mm apart, the tightest pair is 34 px and every digit is legible, which is
why the shared tool never needed this.

Numbering exists so the supports can be COUNTED when the one fixed camera hides
some of them. A number hidden under another number fails that on its own terms,
so the labels are relaxed apart to a minimum separation before they are drawn.

Nothing else moves. This is a fork rather than an edit to `capped_cuboid.py`
because two other objects are being rendered from that file at the same moment.
It patches exactly one function -- `screen`, which main() uses only to place the
text -- so the model, the exhaustive search, the panels and the JSON record come
out bit for bit as the shared tool produces them; only the pixels under the
digits differ. Verified by diffing capped_k1.json across the two runs.

    python slides/tools/capped_A4.py                 # all ten poses of A4
    python slides/tools/capped_A4.py --poses 0 2
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

import capped_cuboid as cc

# far enough apart to read, close enough to stay on the arrowhead the label
# belongs to. cuboid_baseline's tightest legible pair is 34 px at this size and
# the font is 46 px, so 42 px separates the digits without a leader line.
MIN_SEP = 0.050            # of the render width
PAD = 0.035                # keep a nudged label off the edge of the frame


def spread(xy: np.ndarray, px: int, sep: float) -> np.ndarray:
    """Push labels apart until no two are closer than `sep`, moving both equally.

    Symmetric relaxation rather than "move the later one": which support gets
    displaced would otherwise depend on the order the search happened to pick,
    and the same contact would sit in a different place in two neighbouring
    cells of the same figure. Splitting the correction keeps every label the
    same small distance from its own arrowhead.
    """
    p = np.asarray(xy, float).copy()
    if len(p) < 2:
        return p
    jitter = np.random.default_rng(0)
    for _ in range(400):
        moved = False
        for a, b in itertools.combinations(range(len(p)), 2):
            v = p[b] - p[a]
            dist = float(np.linalg.norm(v))
            if dist >= sep:
                continue
            if dist < 1e-6:                       # exactly coincident: any axis will do
                v = jitter.normal(size=2)
                dist = float(np.linalg.norm(v))
            step = 0.5 * (sep - dist) * v / dist
            p[a] -= step
            p[b] += step
            moved = True
        if not moved:
            break
    return np.clip(p, PAD * px, (1 - PAD) * px)


# bound now, not looked up later: main() is made to call the function below by
# rebinding `cc.screen`, so reaching back through `cc.screen` would call this
# same function again forever
_project = cc.screen


def screen(mesh, T, px, pts):
    """`capped_cuboid.screen`, then de-collided."""
    return spread(_project(mesh, T, px, pts), px, MIN_SEP * px)


def main() -> None:
    cc.screen = screen                            # main() resolves it at call time
    if "--object" not in sys.argv:
        sys.argv += ["--object", "A4"]
    cc.main()


if __name__ == "__main__":
    main()
