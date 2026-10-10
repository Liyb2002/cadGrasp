# 32-candidate single-chain trial — stopped

Stopped at user request because serial full construction and all-load candidate evaluation is too expensive. Two rounds completed; third round has 18 completed candidates. No final acceptance/report or final support was produced.

Each candidate changes one pose from the same incumbent, followed by one joint gradient_descent attempt. Each completed round commits at most one improving candidate. Round 1 working maximum deficit: 0.142206 → 0.114960. Round 2: 0.114960 → 0.080212. These are working-load deficits, not full-fixture acceptance.

Records: data/chain_trajectory.json (completed rounds), data/candidate_round_in_progress.json (partial third round), data/optimization_trace.json, data/run.log, data/stopped.json. Published results and upstream inputs were preserved.
