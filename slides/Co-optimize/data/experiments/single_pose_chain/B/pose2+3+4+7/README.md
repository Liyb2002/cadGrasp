# Single-pose sampling chain trial

All four poses start at native up. Each iteration samples only one pose using a random tangent cardinal direction and a 5–30 degree angle, then jointly refines all four poses for up to two descent steps. Continue from this chain state; poses are selected without replacement within each four-pose sampling cycle. Seed: 42.

12 sampling attempts, 22 accepted descent iterations, 35 exact evaluation calls, 311.88 seconds (search, excluding constructor). Best constructed counts in pose 2/3/4/7 order: **32768 / 32765 / 32768 / 32768**. **Unresolved: three original saved loads remain infeasible.** Last chain counts: 32768 / 32762 / 32768 / 32768. Step 9 failed exact clearance/contact/partition construction and retained the preceding chain state.

Report: [data/report.json](data/report.json). Sampling and per-cycle direction states: [data/chain_trajectory.json](data/chain_trajectory.json). Descent acceptance records: [data/optimization_trace.json](data/optimization_trace.json). Candidate support: data/remaining_support.obj. This run records cycle endpoints; individual gradient trial directions were added to the logger after this process started and are available in subsequent runs, not this run.

Original Step3, Step4.1 and published results were preserved. Acceptance checks use original saved loads, full exits and 1% per-side clearance; connectivity remains deferred. This is a bounded one-seed trial, not a success or a comparison proving a better optimizer.
