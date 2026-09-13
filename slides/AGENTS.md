# Baseline output rules

Applies to `baseline_algo/` and its generated outputs.

# Output layout

- All generated baseline results, logs, summaries, candidates, meshes and videos must live under `output/{object_name}/{pose}/{step_directory}/`. Preserve the established stage directory names (such as `step4_floor_contact` and `step5_connect_support`).
- Batch totals go to stdout. Store per-case batch logs and ledgers inside the last requested stage. Do not create top-level run, summary, proposal or source-snapshot directories.
- Store separate Step5 proposals as flat `proposal_*` files (with `proposal.json` as the report) inside that case's Step5 directory. Do not overwrite baseline geometry with a proposal.
- Do not automatically archive stage folders, restore deleted outputs or regenerate deleted experiments. Existing case locks and input references may stay at the pose root.
