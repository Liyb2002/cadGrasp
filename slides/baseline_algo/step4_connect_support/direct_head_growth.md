# Deterministic volume-aware growth

For fixed Step3 heads, exit directions, fixture placements and ground targets, grow the support first and measure its occupied box afterwards. No candidate box is enlarged or used as a target feasibility gate.

Start at the first physical head. Candidate rods connect unattached heads or foot targets to network endpoints or nearest projections on existing rods. Rank candidates lexicographically by the increase in the aggregate workstation XYZ bounding-box volume, then rod length, target index and endpoint coordinates. The box includes all object poses and mandatory head/foot material from the outset. Test whole rods against withdrawal exclusions and all installed floor constraints. Emit the first legal candidate and update occupied bounds.

If no direct candidate is legal, lazily route around obstacles on a fixed navigation grid, shortcut with whole-rod legality checks, and choose among available target routes by the same volume-increment and length criteria. A finite navigation window bounds computation; it is a search limit, not a minimized or expanded output box. The current implementation retains previously validated placement and ground-target locations as inputs. It does not optimize those variables.

After all terminals are connected, independently replay the exported solid against complete 500 mm sweeps, contacts, working faces and ground demand coverage. Compute the final box by min/max over all objects and all installed support vertices. This is the exact axis-aligned box of the emitted result, not a global optimum over possible supports.

Construction figures show contacts, sweep exclusions, actual partial growth, final support, then the measured occupied box. Complete structural rod cores remain at least 5 mm in diameter; original contact edges and documented short transitions remain exceptions. Step3 force verdicts remain unchanged.
