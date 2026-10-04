# B independent initialization comparison

All 20 saved poses, unchanged original 32,768 loads per pose. New initialization: up to six heads, top10 random greedy selection, 30-trajectory budget divided among 0.5%, 1%, 2% area families. CUDA batches LP certificates; original CPU LP checks the selected final solution. Local contact/exit evidence only; full support is not constructed here.

| Pose | Old search | Old coverage | New search | Heads | Selected area/head | Trajectories run | Search seconds |
|---|---|---:|---|---:|---:|---:|---:|
| [pose_1](pose_1/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 3 | 198.7 |
| [pose_2](pose_2/heads.png) | not found | 31533/32768 | PASS | 2 | 1% | 3 | 214.7 |
| [pose_3](pose_3/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 3 | 200.7 |
| [pose_4](pose_4/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 3 | 231.6 |
| [pose_5](pose_5/heads.png) | PASS | 32768/32768 | PASS | 3 | 1% | 3 | 205.4 |
| [pose_6](pose_6/heads.png) | PASS | 32768/32768 | PASS | 3 | 1% | 3 | 184.9 |
| [pose_7](pose_7/heads.png) | PASS | 32768/32768 | PASS | 4 | 2% | 4 | 249.6 |
| [pose_8](pose_8/heads.png) | PASS | 32768/32768 | PASS | 2 | 1% | 3 | 230.7 |
| [pose_9](pose_9/heads.png) | PASS | 32768/32768 | PASS | 5 | 1% | 4 | 301.3 |
| [pose_10](pose_10/heads.png) | PASS | 32768/32768 | PASS | 3 | 1% | 5 | 252.1 |
| [pose_11](pose_11/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 4 | 248.8 |
| [pose_12](pose_12/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 3 | 228.4 |
| [pose_13](pose_13/heads.png) | PASS | 32768/32768 | PASS | 5 | 1% | 3 | 224.3 |
| [pose_14](pose_14/heads.png) | PASS | 32768/32768 | PASS | 5 | 1% | 3 | 207.3 |
| [pose_15](pose_15/heads.png) | PASS | 32768/32768 | PASS | 2 | 1% | 3 | 231.6 |
| [pose_16](pose_16/heads.png) | PASS | 32768/32768 | PASS | 2 | 1% | 3 | 215.1 |
| [pose_17](pose_17/heads.png) | PASS | 32768/32768 | PASS | 2 | 1% | 3 | 235.0 |
| [pose_18](pose_18/heads.png) | PASS | 32768/32768 | PASS | 5 | 1% | 7 | 242.3 |
| [pose_19](pose_19/heads.png) | PASS | 32768/32768 | PASS | 4 | 1% | 4 | 221.2 |
| [pose_20](pose_20/heads.png) | not found | 32551/32768 | PASS | 2 | 1% | 12 | 234.3 |

Each area family stops after its first successful trajectory. Thirty trajectories is a budget, not a requirement to run all thirty after finding initialization. Search failure is not an infeasibility proof.
