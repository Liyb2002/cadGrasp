# Selected B-tested hybrid

The fixed version selected for five-pose testing on other objects is `hybrid_fast.py` (`FastHybridSearch`). Its independently initialized B batch completed and passed provenance/construction checks: **28/30**, including **19/20 normal** and **9/10 illegal** groups. Two cases were initially feasible; 26 were recovered. This matches the historical clearance search's force/exit count, rather than demonstrating a higher success rate. The historical run had some stored-direction warm starts and is not a clean timing control.

The algorithm is the concurrent search in `hybrid_final_README.md`, with a separately recorded compiled signed-distance field for gradient guidance. The field uses libigl 2.6.3, unchanged grid/time quadrature and a separate cache key. Floating-point field values can differ from the earlier Trimesh backend. Exact original geometry, all original loads and the 1% clearance construction remain the acceptance test; the field is not an exit certificate.

B starts only from each group's original Step4.1 native-up directions. Each case has 1,200 sampling proposals, at most eight two-step gradient branches, and a 60-minute wall budget. The two unresolved outcomes exhausted the proposal/branch search, not the wall budget: `pose1+4+7+12+21+27` and `illegal/pose4+7+12+21+23+27`. They are not impossibility proofs.

The successful-case wall-time median was 134.5 s, and the slowest success was 2,971.7 s. These include constructor costs and were measured while other experiments competed for CPU. Faster field queries do not establish faster end-to-end search. Efficiency on difficult cases remains an open issue.

Acceptance is every original load and complete legal exits with 1% clearance outside compatible contact cores. Connectivity, support-ground footprint/coverage, strength and robot motion are deferred. The force model retains the original seven-coordinate no-uplift condition and original workpiece-floor point/friction model (friction coefficient 64, not a measured material value); force acceptance is relative to that model, not an experimental fixture guarantee.

Results: `data/experiments/physics_guided_fast_batch/B/batch.json`. Other-object test entry: `run_generalization5_fast.py`; its original fixed manifest, parameters and this exact algorithm are frozen throughout that batch. Additional B pilots are separate experiments and never add their successes to this 28/30.
