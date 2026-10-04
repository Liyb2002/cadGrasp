# Absolute direction / occupied volume experiment

Optimize contacts jointly across poses in the fixed workstation frame. No head-count penalty and no object-frame direction comparison. This implementation currently searches world +Z object withdrawal with XY translations; it is a bounded geometric search, not a newly trained neural network.

`experiment.py` reuses the cloned baseline's full-load mechanics, path components, continuous contact withdrawal, EnvelopeGrow construction and unchanged Step5 evaluator. Direction features and the measured terminal training target are defined in `value.py`. `render.py` publishes only `overview.png` and `construction_steps.png` per accepted Step4, with one object pose in construction panel 4.

The stationary support must stay above the floor. The object moves +Z; the equivalent relative support motion for collision checking is −Z. The original moving-head floor convention remains the default for legacy callers. Full-body collision, work surfaces, mandatory contact cells, full 5 mm rod/foot cores and ground coverage are retained. Exact original mandatory cells lost during Boolean merging are reinserted before the one construction acceptance; no tolerance is relaxed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/absolute_direction/pipeline.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python slides/value_network/absolute_direction/render.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s slides/value_network/absolute_direction -p 'test_*.py' -v
```

Default batch includes all 10 baseline sets and all 10 additional transfer sets. The latter currently lack an already constructed previous Step4/Step5 comparator; do not invent a percentage improvement for them. Actual results and limitations: [REPORT.md](REPORT.md).

The previous `N_remaining + D_object` datasets/checkpoints are historical and have not been relabeled or retrained. Search failures leave the new value target unknown. Original baseline files and outputs are preserved.

The public entry is `pipeline.py` (also `baseline_algo/run_absolute_direction.py`). It retains verified completed supports, retries failed sets using coupled placement/contact search, then original-sweep contact-transition carving or short collar repairs. Every acceptance uses the original hard thresholds. A carved candidate must preserve every complete rod/start-ball/sole core. Alternative helper scripts and logs record the exploration history; they are not trained predictors. `data/successful_completion_targets.jsonl` stores only actual successful completion costs, with source hashes.

Material connectivity counts positive material components using the existing baseline rule. Internal cavity shells and zero-volume boundary fragments are not additional supports; cavities are retained in the exported solid.
