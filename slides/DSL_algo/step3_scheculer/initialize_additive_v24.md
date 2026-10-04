# Greedy initialization with additive fallback

Entry: `python -m step3_scheculer.initialize_additive_v24 --jobs 6`.
Run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo`
and the project `.venv/bin/python`. `--failed-only` restricts the batch to
existing failed greedy reports; `--objects` and `--poses` restrict cases.

The fast stage is the existing six-head, thirty-trajectory cached greedy.
If a valid report already exists, reuse it. Otherwise run that stage first.
Return immediately if it succeeds, including solutions with fewer than six
heads. Ordinary cases do not run fallback or demand-guided LPs.

If all greedy trajectories fail, continue from their final contact sets,
prioritizing larger original coverage. Try every seed if necessary and stop
at the first accepted solution. Existing contacts are never deleted,
repositioned, replaced, enlarged, or merged. Fallback may add cached heads
of any of the three precomputed area families. Default maximum head count
is twelve (`--max-heads`); this is a computational budget, not a feasibility
theorem. Candidate centers must remain distinct.

For each seed:

1. Recompute full original-load coverage, retaining common withdrawal rays
   and roadmap ports as sets rather than selecting one ray in advance.
2. Evaluate every geometrically eligible single-head addition using all
   32,768 original demands. Add the head with the largest strictly positive
   coverage gain.
3. If no tested single head improves coverage, take up to eight fixed demand
   indices from the currently failed portion of the original sample list.
   These are proposal probes, not a new sample set or acceptance subset.
4. For compatible candidate groups, solve those demands with the current
   heads and all group candidates available. Sparse primal reaction witnesses
   identify candidate owners whose rays actually support the failed demand.
   Propose compatible distinct-center pairs among those owners. A direction
   used for proposing a pair does not lock the accepted state's direction.
5. Evaluate up to 128 proposed pairs against all original demands; add the
   pair with the largest strictly positive coverage gain. Pair generation
   is deliberately bounded and is not exhaustive two-head search.
6. Repeat until full coverage, the head budget, or no improving tested
   addition. Retain all previously satisfied loads. On failure try the next
   original greedy seed.

CUDA batches nonnegative primal and valid dual certificates; LP witness
solves remain on CPU. Clamped nonnegative primal coefficients are accepted
only after reconstructing the actual original equations at the original
residual tolerance, with a conservative GPU margin. Final coverage is checked
independently by the original CPU classifier, rebuilding normals and moments
from actual mesh/contact triangles. Retain the shared no-uplift constraint,
work exclusions, floor clearance, strict Step2 hashes, continuously checked
head rays, and common roadmap ports.

Output stage: `output/<object>/independent_poses_additive_v24/<pose>/step3_scheculer/`.
Fallback outputs include final contacts, full coverage, seed histories,
numerical recovery evidence, and provenance. A fast-stage reuse report refers
to the original greedy report in `provenance.inputs`; that original report
owns its accepted contacts and force evidence. No old greedy or diagnostic
generator/output is overwritten. This stage certifies single-pose contacts
and cached local exits, not a constructed fixture, strength, or Step4 volume.

Failure is **bounded additive search did not find a solution**, not proof of
infeasibility. Original contacts can restrict exits permanently, a useful
combination may require more than two new heads, witness-based proposals may
miss another pair, numerical proposals can remain unresolved, and the head
budget may be insufficient. Enlarging these budgets does not establish a
continuous-geometry infeasibility proof.
