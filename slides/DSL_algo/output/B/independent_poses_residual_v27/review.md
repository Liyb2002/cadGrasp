# Residual witness fallback results

Greedy: 613/630; fallback rescued: 3/17; combined: 616/630.

| Object | Pose | Passed | Heads | Added heads | Original coverage | Final coverage | Seeds tried | Search seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A4 | pose_4 | no | 7 | 1 | 32080 | 32646 | 30 | 13.49 |
| A4 | pose_8 | no | 6 | 0 | 28682 | 28682 | 30 | 4.15 |
| A4 | pose_15 | no | 6 | 0 | 32656 | 32656 | 30 | 4.06 |
| A4 | pose_19 | no | 6 | 0 | 30958 | 30958 | 30 | 3.79 |
| A4 | pose_20 | no | 6 | 0 | 30980 | 30980 | 30 | 5.78 |
| A4 | pose_26 | no | 7 | 1 | 32104 | 32698 | 30 | 107.14 |
| A5 | pose_29 | yes | 9 | 3 | 0 | 32768 | 1 | 35.16 |
| C1 | pose_15 | no | 6 | 0 | 31134 | 31134 | 30 | 14.31 |
| C5 | pose_26 | yes | 9 | 3 | 32360 | 32768 | 28 | 7.42 |
| C7 | pose_4 | no | 6 | 0 | 29080 | 29080 | 30 | 1.18 |
| C7 | pose_12 | no | 6 | 0 | 25285 | 25285 | 30 | 3.39 |
| C7 | pose_14 | no | 6 | 0 | 30114 | 30114 | 30 | 17.30 |
| C7 | pose_15 | yes | 8 | 2 | 32499 | 32768 | 9 | 2.15 |
| C7 | pose_24 | no | 6 | 0 | 25910 | 25910 | 30 | 2.35 |
| C7 | pose_30 | no | 6 | 0 | 32649 | 32649 | 30 | 2.05 |
| D4 | pose_9 | no | 6 | 0 | 32751 | 32751 | 30 | 8.81 |
| D4 | pose_27 | no | 6 | 0 | 32758 | 32758 | 30 | 9.98 |

All accepted fallback solutions use all original loads and retain their entire greedy seed. No complete support body was constructed. Failed bounded searches are not infeasibility proofs.

Final v27 batch wall time: **117.731 s** (six workers), including setup, search and final CPU checks; excludes the pre-existing greedy run and interrupted development trials. Sum of per-pose elapsed times: 256.276 s.

| Rescued pose | Total heads | Added heads | Total pose seconds |
|---|---:|---:|---:|
| A5/pose_29 | 9 | 3 | 35.457 |
| C5/pose_26 | 9 | 3 | 7.842 |
| C7/pose_15 | 8 | 2 | 2.600 |

Compared with v24, the same three poses are rescued. A5/pose_29 uses nine heads versus eight before. This experiment improves failure screening, but demonstrates no rescue-rate or head-count improvement. All thirty seeds of each of the fourteen remaining poses were excluded by a verified separator for their optimistic eligible-head cone, possibly after a partial addition. This does not exclude different seeds, replacement of existing heads or a richer candidate catalogue.
