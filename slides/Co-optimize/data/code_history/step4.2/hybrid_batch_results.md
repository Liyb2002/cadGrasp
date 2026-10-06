# B: serial hybrid comparison

All 30 groups were attempted from their own Step4.1 native-up directions. **28/30 passed**: normal 19/20, illegal 9/10. Of these, two were initially feasible and 26 were recovered. All successful report inputs, code and artifacts passed provenance checks.

Two groups exceeded a wall-time budget introduced during the exploratory run: `pose1+4+7+12+21+27` and `illegal/pose4+7+12+21+23+27`. The cap was introduced after they had already run longer than 60 minutes; actual elapsed time is preserved in `wall_budget.json`. These are budget-exhausted cases, not proofs of impossibility and not numerical-force failures. The original proposal budget was 1,200 and each refinement allowed 12 gradient steps.

This is an independent fresh-initialization experiment; no historical successful directions were used. Force/exit acceptance excludes connectivity, actual support ground coverage, strength and robot motion. Old clearance search also had 28/30 force/exit successes, including 19/20 normal groups, but some old records used stored successful directions as warm starts; its proposal counts are not a fresh-search timing baseline.

| Pose set | Force/exit | Initial counts | Final counts | Seconds |
|---|---|---|---|---:|
| pose3+15 | pass | None | [32768, 32768] | 46.8 |
| pose19+28 | pass | [32768, 23885] | [32768, 32768] | 18.1 |
| pose8+21 | pass | None | [32768, 32768] | 30.6 |
| pose2+20 | pass | None | [32768, 32768] | 30.2 |
| pose1+12+29 | pass | [459, 24310, 32157] | [32768, 32768, 32768] | 46.5 |
| pose4+5+7 | pass | [7348, 89, 468] | [32768, 32768, 32768] | 48.5 |
| pose8+10+19 | pass | [331, 1904, 0] | [32768, 32768, 32768] | 20.4 |
| pose18+23+24 | pass | [147, 0, 211] | [32768, 32768, 32768] | 22.8 |
| pose8+9+13+30 | pass | [0, 0, 0, 0] | [32768, 32768, 32768, 32768] | 390.2 |
| pose2+3+4+7 | pass | [0, 30313, 200, 0] | [32768, 32768, 32768, 32768] | 32.9 |
| pose1+11+14+27 | pass | [0, 0, 0, 0] | [32768, 32768, 32768, 32768] | 55.4 |
| pose5+6+23+29 | pass | [0, 0, 0, 0] | [32768, 32768, 32768, 32768] | 28.0 |
| pose1+4+7+9+24 | pass | [0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768] | 360.6 |
| pose5+6+11+23+29 | pass | [0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768] | 90.5 |
| pose12+16+19+21+27 | pass | [0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768] | 446.0 |
| pose6+10+13+17+30 | pass | [0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768] | 172.3 |
| pose1+2+4+5+6+7 | pass | [0, 0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768, 32768] | 209.7 |
| pose4+5+8+9+19+23 | pass | [0, 0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768, 32768] | 1364.6 |
| pose1+6+11+13+14+17 | pass | [0, 0, 0, 0, 0, 0] | [32768, 32768, 32768, 32768, 32768, 32768] | 1143.0 |
| pose1+4+7+12+21+27 | wall_budget_exhausted | — | — | 4120.0 |
| illegal/pose2+29 | pass | [32741, 32768] | [32768, 32768] | 23.0 |
| illegal/pose11+19 | pass | [32768, 32768] | [32768, 32768] | 15.0 |
| illegal/pose1+2+29 | pass | [641, 32740, 32767] | [32768, 32768, 32768] | 31.2 |
| illegal/pose5+13+27 | pass | [32768, 32768, 32768] | [32768, 32768, 32768] | 13.6 |
| illegal/pose1+2+6+15 | pass | [0, 0, 0, 0] | [32768, 32768, 32768, 32768] | 357.6 |
| illegal/pose7+11+13+19 | pass | [31811, 29109, 6206, 32176] | [32768, 32768, 32768, 32768] | 46.2 |
| illegal/pose1+2+6+7+29 | pass | [0, 2170, 115, 449, 0] | [32768, 32768, 32768, 32768, 32768] | 38.0 |
| illegal/pose5+6+13+15+27 | pass | [32768, 28919, 32768, 30006, 32768] | [32768, 32768, 32768, 32768, 32768] | 569.5 |
| illegal/pose4+7+12+21+23+27 | wall_budget_exhausted | — | — | 3876.1 |
| illegal/pose5+12+20+24+27+28 | pass | [1975, 1769, 32768, 30971, 19824, 840] | [32768, 32768, 32768, 32768, 32768, 32768] | 56.2 |
