# Candidate-local force-descent trial

B/pose2+3+4+7, one chain, 3 rounds × 32 candidates, seed 42. Both raw and descent endpoints checked on all original loads using contact locks. No support Boolean construction. Search runtime: 40.52 seconds (constructor excluded). Mean force-descent time: 245 ms; 199 ms prepare/critical-load selection and about 46 ms inner optimization.

88 candidates improve their real rank after descent; 8 worsen and are rolled back. Analytic derivative directional relative error maximum: 2.77e-8. Rank is minimum feasible pose fraction, then total fractions; individual pose monotonicity is not claimed. Selected round counts: [(1, [6, 32630, 1875, 7800]), (2, [967, 32715, 2852, 31641]), (3, [3204, 32766, 31489, 32751])].

Final: **[3204, 32766, 31489, 32751]**, each pose has 32768 original loads. **Unresolved**. This relaxed force descent is not exact smooth optimization of the discontinuous contact set, nor a convergence guarantee. Whole support patches, contact cores and 1% clearance have not been certified. Historical published results remain unchanged.

Data: report.json, chain_trajectory.json (every raw/descent comparison and timing), contact_locks.npz, directions.npz, run.log. Implementation: helper_func/optimization/force_descent.py and force_candidate_chain.py. 80 active tests pass.
