# Neural absolute-volume experiment

This is the first trained network for the revised absolute-world-direction objective. The older head-count/object-frame checkpoints remain unchanged. Training data lives in `../../data/B/absolute_volume/`; weights and diagnostics are `best.pt` and `training.json` here.

The 20 actual accepted Step4/Step5 solutions supply 64 state samples per group (1,280 samples, 1,192 unique within-group selected sets, and 21,302 positive state/action labels). A separate random seed supplies 320 evaluation state samples of the same groups, including repeated empty states. For an unselected head belonging to a witnessed completion, the value label is `log1p(V_supported / V_objects - 1)` from the real unchanged Step5 result. No unsuccessful or unsearched head gets a fabricated volume label. There is no head-count penalty.

The model takes explicit selected-head identity bits and pose-membership bits. Its state/action value branch predicts final occupied-volume cost. Additional branches learn membership in the demonstrated successful completion, a deterministic tie order and each pose's XY seating. Non-demonstrated heads are negative examples only for imitation membership, not physical infeasibility. The terminal value prior is conditioned on the pose group; state/action residuals can change it as the selected set changes. This is a fixed-pose model, not a geometry encoder for unseen objects.

Each group currently has one completed solution, so its demonstrated actions have equal observed continuation costs. These labels cannot establish which alternative heads would yield a smaller support. Within the measured value-regression uncertainty, learned demonstration order breaks ties. This version tests whether a network can reproduce the successful new-objective solutions without retrieving their saved plans; it does not establish optimization or generalization beyond those solutions.

Runtime loads the network checkpoint, candidate geometry and original pose/load inputs. It never loads the new successful-completion JSONL or retrieves its contact/layout choices. The historical pair's task loader also opens frozen baseline reference records to reconstruct the original physical inputs; the loader's reference contacts and placement are discarded. The network predicts contacts and XY seating (rounded to a global 5 mm grid), checks common insertion paths and all original 32,768 loads per pose, then constructs a new support. World +Z withdrawal, full original sweeps, complete rod/sole cores, ground coverage and the one construction acceptance remain mandatory. Unknown or illegal neural predictions cause a recorded failure; there is no teacher-plan or greedy fallback. Step5 measures the accepted support. Two English Step4 images are published per successful group.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/train/absolute_volume/fit.py --epochs 6000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/train/absolute_volume/batch.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python slides/value_network/train/absolute_volume/report.py
```

[Actual constructed results](REPORT.md). These 20 groups are all included in training. New partial-state evaluation and fresh construction are not evidence of unseen-group generalization.

The cloned baseline also exposes the neural batch through `baseline_algo/run_neural_absolute_direction.py`. Its older `run_absolute_direction.py` remains the geometric-search comparison entry.

`batch.py` keeps verified completed neural outputs. For construction failures, it tries original-sweep carving and then the original uncarved constructor on the same neural contact/layout predictions. Both require the same complete cores and one full construction acceptance. Carving is rejected if it would cut a complete rod/sole core; the plain constructor must still pass full original withdrawal checks. This constructor retry never retrieves a teacher plan or changes the neural policy.
