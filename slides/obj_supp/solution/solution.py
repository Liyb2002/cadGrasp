r"""Expand [I_3; [r_i]_cross] F_i = demand, for one and several forces.

    python slides/obj_supp/solution/solution.py

All force vectors show their Cartesian components. Each force has a 6 x 3
geometry block: identity above, the cross-product matrix of r_i = p_i - c below.
The lower equation adds separate matrix-times-force terms, A_0 F_0 + ... + A_n F_n.
These are equilibrium equations to solve; contact admissibility is a separate
constraint, documented in solution.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT = Path(__file__).with_name("solution.png")
PAPER, INK, MUTED = "#ffffff", "#1b1b1a", "#6b6b66"
GEOMETRY, FORCE, DEMAND = "#367c72", "#ae6b1c", "#b02a26"
RULE = "#d9ddd6"


def bracket(ax, left, right, bottom, top):
    cap = 0.006
    ax.plot([left + cap, left, left, left + cap],
            [top, top, bottom, bottom], color=INK, lw=1.8)
    ax.plot([right - cap, right, right, right - cap],
            [top, top, bottom, bottom], color=INK, lw=1.8)


def geometry_rows(index):
    """Exactly [I_3; [r_i]_cross], with [r_i]_cross F_i = r_i x F_i."""
    x, y, z = (rf"r_{{{index}{axis}}}" for axis in "xyz")
    return (("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1"),
            ("0", "-" + z, y), (z, "0", "-" + x), ("-" + y, x, "0"))


def force_components(index):
    return [rf"F_{{{index}{axis}}}" for axis in "xyz"]


def main():
    fig = plt.figure(figsize=(18, 13), dpi=180, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, value, fs=26, color=INK, ha="center"):
        ax.text(x, y, value, ha=ha, va="center", fontsize=fs, color=color)

    def matrix(left, right, top, bottom, rows, *, color, fs=24, split=None):
        bracket(ax, left, right, bottom, top)
        count = len(rows[0])
        xs = np.linspace(left, right, 2 * count + 1)[1::2]
        ys = np.linspace(top, bottom, 2 * len(rows) + 1)[1::2]
        for y, row in zip(ys, rows):
            for x, entry in zip(xs, row):
                text(x, y, "$" + entry + "$", fs, color)
        if split is not None:
            y = (ys[split - 1] + ys[split]) / 2
            ax.plot([left + .007, right - .007], [y, y], color=RULE,
                    lw=1.1, linestyle=(0, (4, 4)))

    demand = [[rf"F_{{D{axis}}}"] for axis in "xyz"]
    demand += [[rf"\tau_{{D{axis}}}"] for axis in "xyz"]

    text(.5, .960, "Solution", 34)
    text(.5, .918, "Geometry matrix × force components = force–moment demand", 21, MUTED)

    text(.04, .854, "One force", 21, ha="left")
    text(.32, .854, r"$6\times3$ geometry matrix", 18, GEOMETRY)
    text(.61, .854, "force", 18, FORCE)
    text(.862, .854, "demand", 18, DEMAND)
    matrix(.195, .455, .818, .557, geometry_rows(1), color=GEOMETRY, fs=27, split=3)
    text(.133, .753, r"$I_3$", 25, GEOMETRY)
    text(.133, .622, r"$[r_1]_\times$", 25, GEOMETRY)
    text(.508, .688, r"$\times$", 31)
    matrix(.561, .659, .753, .622, [[v] for v in force_components(1)],
           color=FORCE, fs=26)
    text(.748, .688, r"$=$", 33)
    matrix(.811, .913, .818, .557, demand, color=DEMAND, fs=26, split=3)

    ax.plot([.04, .96], [.511, .511], color=RULE, lw=1)
    text(.04, .474, "Several forces", 21, ha="left")
    text(.175, .426, "Force 0 contribution", 18, MUTED)
    text(.600, .426, "Force n contribution", 18, MUTED)
    text(.897, .426, "same demand", 18, DEMAND)

    # Repeat the upper matrix-times-force form for each contribution, then add.
    # Each force retains its own three-row vector and dedicated horizontal space.
    for index, shift in ((0, 0.0), ("n", .425)):
        matrix(.030 + shift, .220 + shift, .382, .140,
               geometry_rows(index), color=GEOMETRY, fs=23, split=3)
        text(.238 + shift, .261, r"$\times$", 25)
        matrix(.257 + shift, .324 + shift, .3215, .2005,
               [[v] for v in force_components(index)], color=FORCE, fs=23)
    text(.350, .261, r"$+$", 28)
    text(.386, .261, r"$\cdots$", 28)
    text(.422, .261, r"$+$", 28)
    text(.798, .261, r"$=$", 33)
    matrix(.844, .950, .382, .140, demand, color=DEMAND, fs=24, split=3)

    text(.5, .076,
         r"$r_i=p_i-c$   ·   $[r_i]_\times F_i=r_i\times F_i$   ·   "
         r"$i=0$: floor contact", 18, MUTED)
    text(.5, .037, "Each force contributes to both force balance and moment balance.",
         18, MUTED)
    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
