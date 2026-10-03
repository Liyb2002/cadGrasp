# Fixed-group, conditional-value pilot

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python slides/value_network/data_producer/collect_states.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python slides/value_network/data_producer/finalize_pilot.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python slides/value_network/data_producer/audit_pilot.py
```

The group list is the ten entries in the existing Step5 `all_groups.json`. Every group receives 20 distinct states, including the empty state, varied prefixes of successful completions, and some compatible prefixes with an extra candidate. This pilot uses the current `independent_poses` task inputs, as did the accepted value experiment. Historical fixture placements are not the input. `pose1+3` and `pose1+3copied` have identical current task identities, but get separate sampled states; they are not two independent geometries.

For every state, all unselected candidates in all member poses receive a record. Step2-invalid candidates are marked `step2_rejected`. Empty common direction or connection-component intersections are marked `no_path`, with no mechanical solve. Remaining actions reuse all eligible cached certified local completions. If none exists, they receive up to eight conditional continuation proposals. Each successful completion preserves every selected head and the scored action. No pose may exceed six selected heads. Intermediate prefixes do not need to pass mechanical or uplift checks. Compact seeds can also be bootstrapped by forming maximal common-direction/common-component supersets, verifying them, and deleting redundant heads; supersets larger than six are only search proposals, never training completions. Pose2 received thirteen three/four-head seeds from this search. When a pose has no successful baseline seed, the generator first augments the stored incomplete baseline groups and tries random compatible growth; an empty certified bank is allowed and does not abort collection.

Local terminal acceptance uses the existing baseline joint force/moment/no-uplift verifier: 96 original loads can reject a proposal, but success requires all 32,768 original loads. Numerical failures are recorded separately, not treated as proof of infeasibility. Successful completions and checks are shared across states. Four independent worker processes start from common certified banks, keep separate caches while running, and merge their certificates after all groups finish. `budget_unresolved` has a null value; it is not an impossibility label.

For a completion, the target is the number of heads still required AFTER the action, plus the final exit dispersion (lambda=1). Dispersion is the mean squared angle/pi over all unordered pose pairs, measured in the object frame. Two-pose direction selection is exact over the recorded finite direction menu and completion bank. For more than two poses, eight deterministic multistart coordinate searches choose directions and corresponding certified completions. Scores are the best found feasible costs, hence upper bounds on the true minimum; they are not global-optimal labels.

Files:

- `data/B/<group>/pilot20/states.json`: 20 fixed states and member poses.
- `state_000.json` through `state_019.json`: all action values, statuses, decompositions and complete contact witnesses.
- `training_records.jsonl`: one record per state/action, including the selected-head state.
- `summary.json`: counts and measured state compute times.
- `data/B/pilot20_shared/`: common preparation caches, certified completion banks, full-load terminal results, numerical retry evidence, configuration and final batch summary.
- `data/B/pilot20.log`: batch progress log; `pilot20_shared/<group>.log` contains per-state worker progress.
- `pilot20_shared/groups/<group>/`: isolated worker caches and configurations.
- `bootstrap_pose.py`: optional compact-seed discovery; pose2 evidence lives in `pilot20_shared/bootstrap_pose_2/`.

Existing state outputs are reused on restart, while a changed code/input/search configuration is rejected. Runtime sums in group summaries retain completed-state compute times; invocation wall time measures that invocation only. No Step4 support solid is generated. Baseline inputs and outputs are read only.

After collection, `finalize_pilot.py` recomputes labels using the merged certified bank for all groups. This requires no new mechanical solves. Previously unknown actions become finite when a compatible verified completion is available; existing values only decrease. Old values are retained in `value_before_global_bank` when changed. The audit checks the JSONL export against the state files.
