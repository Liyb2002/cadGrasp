r"""Open obj_supp with the two workpiece equilibrium equations.

    python slides/obj_supp/two_equations.py

The equations and visual notation follow setup/equations/three_equations.py.
Both rows use the same contact pressure field, including the workpiece's own
floor contact. Moments are about the workpiece centre of mass. The required
process load is fixed at K = 0.5; see ../problem_statement.md#当前决定与讨论记录.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT = Path(__file__).with_name("two_equations.png")
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
EQ, LAB, SUB = 30, 21, 20


def main() -> None:
    fig = plt.figure(figsize=(16.0, 7.0), dpi=200, facecolor=PAPER)

    def text(y, value, *, x=0.5, ha="center", fs=EQ, color=INK):
        fig.text(x, y, value, ha=ha, va="center", fontsize=fs, color=color)

    label_x, equation_x = 0.030, 0.150
    columns = (0.030, 0.165, 0.520, 0.720)
    text(0.945, r"Find a pressure field  $F_{\rm supp}$  — it presses, it cannot pull",
         fs=36)
    for y, label, cells in (
        (0.840, "the supports", (r"$F_{\rm supp}$  what it presses with",
                                 r"$r_{\rm supp}$  its arm", r"$\hat{y}$  up")),
        (0.775, "the process", (r"$F_{\rm push}$  the push, one vector",
                                r"$r_{\rm push}$  its arm", r"$q$  where it lands")),
    ):
        text(y, label, x=columns[0], ha="left", fs=SUB)
        for x, cell in zip(columns[1:], cells):
            text(y, cell, x=x, ha="left", fs=SUB, color=MUTED)

    text(0.710, r"$mg$  the workpiece's weight, so gravity is $-mg\,\hat{y}$   ·   "
         r"$c$  the centre of mass, which both arms are measured from",
         x=label_x, ha="left", fs=SUB, color=MUTED)
    text(0.645, r"$F_{\rm supp}$ points along the skin's own inward normal   ·   "
         r"the floor under the workpiece counts as a support",
         x=label_x, ha="left", fs=SUB, color=MUTED)

    for y, label, equation in (
        (0.490, "force",
         r"$\int_{\rm supp\_obj}\; F_{\rm supp} \; dA \;=\;"
         r" -\left( -mg\,\hat{y} \;+\; F_{\rm push} \right)$"),
        (0.300, "torque",
         r"$\int_{\rm supp\_obj}\; r_{\rm supp} \times F_{\rm supp}"
         r" \; dA \;=\; -\left( r_{\rm push} \times F_{\rm push} \right)$"),
    ):
        text(y, label, x=label_x, ha="left", fs=LAB, color=MUTED)
        text(y, equation, x=equation_x, ha="left")

    text(0.140, r"the same pressure field  $F_{\rm supp}$  must satisfy both rows",
         fs=26)
    text(0.055, r"for every push   ·   $|F_{\rm push}| = 0.5\,mg$",
         fs=SUB, color=MUTED)

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    pixels = plt.imread(OUT)
    ink = (pixels[:, :, :3] < 0.96).any(axis=2)
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    matplotlib.image.imsave(
        OUT,
        pixels[max(rows.min() - 30, 0):rows.max() + 31,
               max(cols.min() - 30, 0):cols.max() + 31],
    )
    print(OUT)


if __name__ == "__main__":
    main()
