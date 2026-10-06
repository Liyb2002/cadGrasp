# Fixed five-pose tests on other objects

First select a version on B, then freeze it and evaluate other objects. The current candidate is [`hybrid_clearance_fast.py`](hybrid_clearance_fast.py), the concurrent sampling/short-gradient hybrid with a compiled distance field and conservative repair of numerically collapsed padded sweep prisms. Its fresh B batch finished at **28/30**, meeting the predeclared gate. The fixed other-object batch has also finished: **20 cases ended, 6 accepted**, with six wall-budget outcomes, six unresolved outcomes and two bounded-search non-passes. Accepted objects are C3, C5, C6, D1, D3 and D4. See [the output index](../data/experiments/previous_output_README.md).

`generalization5_manifest.json` freezes the first saved five-pose set of each of the 20 other active objects. No favorable group is substituted after seeing outcomes. Original meshes, transforms, work regions, loads and the original force model remain unchanged. Parameters stay fixed during the batch; subsequent B-only gradient pilots do not change it.

The active batch is `data/experiments/generalization5_ready/batch.json`. Settings: 12 configured iterations, 1,200 sampling proposals, bounded two-step/eight-branch parallel gradients, and a 30-minute per-case evaluation wall budget. Observed waiting for B selection is excluded from that budget. Prerequisite construction is included; gate-observation delay and process-clock precision limit timing accuracy. These are observational timings, not controlled performance comparisons.

`helper_func/prepare_five_pose_group_ready.py` reuses checked prerequisites or constructs Step3.1 registration, Step3.2 common non-work surface and Step3.3 support/rings without images. If the original shared-vertex surface offset fails, a recorded fallback uses 5 mm prisms along each original face normal. It preserves the original inner triangles, subtracts original object/work solids, and retains the original all-load and work checks. This fallback is a prerequisite construction change; it does not move the object, resample loads or imply a force pass. Existing valid reports are reused with provenance checks.

Native-up directions initialize every object. Full exit travel is at least 500 mm and exceeds the combined support/object bounding-box diagonal by 20 mm. No B exit directions or successful warm starts are transferred. Collapsed *padded* exit prisms are conservatively enlarged by a recorded tiny numerical amount and retained, rather than omitted. Original nominal sweeps remain unchanged.

Acceptance requires all 32,768 saved loads in each of the five poses and complete legal exits with per-side clearance equal to 1% of that object's maximum extent. Only actual motion-compatible bearing cores are exempt from padding. Connectivity, support ground coverage, strength and robot motion remain deferred. This is force/exit acceptance, not full fixture acceptance. No exported-model replay is performed. Certified prerequisite force failures, numerical uncertainty and search/time budget exhaustion are reported separately.

```sh
.venv/bin/python slides/Co-optimize/step4.2/run_generalization5_ready.py \
  --out /absolute/path/to/slides/Co-optimize/data/experiments/generalization5_ready \
  --workers 4 --iterations 12 --max-proposals 1200 \
  --b-gate /absolute/path/to/slides/Co-optimize/data/experiments/physics_guided_clearance_fast_batch/B/batch.json
```

Historical batches remain separate. The earlier `generalization5_fast` batch started after its version passed 28/30 on B, then stopped with eight completed cases: four evaluation timeouts and four unresolved surface-offset prerequisites, zero accepted cases. The remaining twelve were not completed tests. Other superseded batches stopped before any optimization results. These observations motivated numerical sweep repair and the explicit surface-prism fallback; they do not establish that the objects are infeasible. Profiling and warm-continuation pilots are excluded from the 20-object count.
