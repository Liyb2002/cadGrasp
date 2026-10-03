# Greedy growth close to a fixed usage envelope

Entry: `run_envelope_growth.py`, constructor `EnvelopeGrow`. Grow one support from the original contact starts. Keep Step3 heads, exit directions and saved placements fixed. Rebuild all nine active groups and measure all ten Step5 groups; preserve historical `pose1+3` Step4.

## Construction rule

Compute a fixed workstation AABB over all object poses, ground demand points and installed contact starts. Add one sole radius (3 mm) as a fixed allowance for finite-thickness branches. This reference is guidance, not a certified construction domain. It never expands during growth.

Each iteration proposes a component join and a ground branch. A join connects an existing start to a start or projection on a different component. For ground growth, find the most exposed uncovered demand-hull vertex across all poses and the direction from its nearest point on the current ground hull. Propose optional soles extending that hull beyond the exposed supporting line, preferably with one sole radius of slack. If slack is unavailable, reach the demand line or advance as far as legal candidates permit. Foot positions are generated outputs from demand neighborhoods and floor sections of the fixed envelope; there are no required foot targets.

Choose the proposal with least additional enclosing-box volume relative to the fixed reference, considering every installed copy of its entire rod and sole. On equal cost prefer a component join, then shorter branches and fixed geometric ties. Free joins can be accepted without evaluating ground proposals. Ground attachment sites use the same box-volume ordering. Prefer straight rods; invoke bounded lazy obstacle routing only when the straight proposals are blocked. Routing guidance also stays tied to the fixed reference. Accept one action, update components and actual ground hulls, and repeat. Stop when the support is connected and every demand hull is covered. Conservatively remove redundant ground leaves.

There is one growing state, no beam, weighted multi-objective score or mandatory head-first phase. Numerical volume ties are rounded at 1e-14 m3. A 160-action cap detects failure. The greedy construction does not guarantee a globally minimal or uniformly improved final box.

## Acceptance and output

Every candidate rod and sole must obey all installed floors and full withdrawal exclusions. Perform one construction acceptance for contacts, work surfaces, connectivity, ground coverage and continuous 500 mm withdrawal sweeps. No exported-model recheck or independent replay. Preserve complete 5 mm rod/foot cores, with documented original-contact and short-transition exceptions. Step3 force verdicts remain unchanged.

Measure the final workstation XYZ box over all object poses and installed copies of the constructed support. Report occupied volume, XY area, material volume, construction time and acceptance time separately. Publish only English `overview.png` and `construction_steps.png`; panel 4 displays one object pose. Blue coverage outlines are constraints, and the blue final-box outline shows the fixed reference. The copied reference keeps the historical first two illustration panels.

Comparison archives: `data/history/before_envelope_growth/` contains the joint beam experiment; `data/history/before_joint_growth/` contains the preceding fast sequential method. Neither archive supplies foot targets or support geometry to this construction. Saved placements remain fixed inputs.
