# Fast 32-candidate benchmark

One chain; 3 rounds / 96 cheap candidates; exact construction only for the top finalist of each round. Instrumented with cProfile; includes timing overhead. Search and finish: 36.19 seconds, constructor: 1.08 seconds. No complete force/exit acceptance: final pose 2/3/4/7 counts are 492 / 32768 / 4101 / 32683.

- Shared guidance: 37.9–42.7 ms per round.
- direction_choice: about 0.14 ms per candidate.
- gradient_descent: 6.53–10.20 ms per candidate.
- All 32 candidates including guidance: 0.256–0.374 seconds per round.
- Full original-load classification across all four poses (4 × 32768 six-dimensional demands, including saved resultant forces and floor reactions): 69.8–237.9 ms per exact state, including generator creation.
- Exact finalist geometry/classification/comparison: 5.87 / 17.20 / 7.75 seconds.

Profile: 34.15 of 36.26 seconds inside exact construction; swept solids account for 28.69 seconds, including 17.80 seconds in Boolean sweep unions. Classification is 0.56 seconds summed over four exact states. Inner candidate evaluation is 0.92 seconds total.

Artifacts: data/report.json, data/candidate_timing.json, data/classification_timing.json, data/runtime.prof, data/run.log. This is a bounded one-seed timing trial; approximate surrogate ranking and faster evaluation do not prove convergence or better fixture quality. The uninstrumented repeat is in ../fast_candidate_timing's corresponding object/set path.
