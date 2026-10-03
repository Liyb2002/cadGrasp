# Fast head growth with adaptive ground coverage

Entry: `run_coverage_growth.py`; constructor: `CoverageGrow`.

Keep Step3 contacts, exit directions and fixture placements fixed. Ground demands specify a region the contact hull must contain; foot locations are construction outputs. This is a fast deterministic heuristic with a compact-volume preference, not global optimization.

1. Join contact starts with legal straight rods, sharing existing branches. Prefer smaller occupied-box increments and shorter rods. When blocked, accept the first feasible detour rather than comparing all detours.
2. Detours use a lazy 8 mm navigation graph. Check complete rods only on proposed paths, reject colliding edges, and search again. Cache attachments and edge checks. Also reject primitives that bury original contact vertices; volume tolerance alone is insufficient for this surface condition. Penalize excursions outside the current occupied box. A bounded 4 mm recovery is available if coarse navigation fails; no fine grid is built by default.
3. Add ground branches only where coverage is missing. Use a small proposal set from eight directional demand extremes, their neighborhoods, current-envelope ground sections and all-pose floor halfspaces. Do not construct or Boolean-test the entire pool. Retain candidates providing at least 75% of the best coverage gain, rank compact volume and length, and check exact solids only when they are about to be used. Try four attachment choices; stop at the first legal connection or detour.
4. Maintain ground hulls using accepted part vertices and assemble all original primitives in one flat union only for snapshots and final output, preserving complete cores at coincident junctions. Remove redundant leaf branches using coverage and conservative dependency checks; retain all branches if final connectivity fails after pruning. Validate the constructed support once against original contacts, work surfaces, every floor, all ground demands and full 500 mm withdrawal sweeps.
5. Measure actual workstation XYZ min/max over all objects and installed supports after construction.

Ground containment uses all vertices of the original demand convex hull. Final acceptance checks the constructed material, so candidate approximations cannot relax correctness. The 50 mm navigation margin and finite routing budgets are computational limits, not an optimized box or general infeasibility proof. Rod cores remain at least 5 mm; original contact edges and recorded short transitions remain exceptions. Step3 force verdicts are unchanged.

Reports separate setup, contact starts, head-network growth, ground growth, one construction validation. Exported-model rechecks and independent replay are disabled by user instruction. Figure generation is presentation cost, not construction time. Speed and occupied volume must be reported separately; some groups may use larger boxes.
