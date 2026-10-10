# Fast candidate timing (uninstrumented)

B/pose2+3+4+7, one chain, 32 candidates/round, 3 rounds, seed 42. Candidates use shared 64-probe frozen sweep/physics guidance; only the top finalist receives exact construction/all-load validation.

| Stage | Measured time |
| --- | --- |
| One direction_choice | 0.086–0.089 ms |
| One gradient_descent | 3.68–5.77 ms |
| Shared preparation per round | 36.9–43.1 ms |
| 32 candidates plus preparation per round | 0.164–0.229 s |
| Exact finalist validation/comparison per round | 5.65 / 16.26 / 7.28 s |
| All three rounds plus initial exact construction and finish | 33.99 s |

Instrumented classification measurement from the matching fast_candidate_benchmark run: 4 × 32768 original demands checked in 69.8–237.9 ms per state, including contact generator creation. Boolean sweep geometry dominates the full validation cost. The saved six-dimensional demands and floor reactions are unchanged; no new downward-force-only substitute is used.

Final counts in pose 2/3/4/7 order: [492, 32768, 4101, 32683]. Unresolved after the bounded three-round trial; faster candidate evaluation does not prove feasibility or convergence. All three submitted finalists improved the real working-load force deficit. Candidate surrogate rankings remain approximations and exact acceptance is retained.

Artifacts: data/report.json, data/candidate_timing.json, data/chain_trajectory.json, data/run.log. Published outputs and upstream stages were preserved. 76 active tests pass, including analytic surrogate-gradient and single-pose-candidate regressions.
