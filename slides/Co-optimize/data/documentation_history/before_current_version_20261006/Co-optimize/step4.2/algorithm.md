# Step4.1 起点上的联合局部优化（当前默认）

默认 `--search local-descent`。直接读取本组 Step4.1 `data/report.json` 的退出方向，地面法向只作为合法半球约束；不随机 propose，不从 native up 重启。每轮在当前方向的切平面中联合优化全部 pose，4 度局部信赖域，最多 12 个 SLSQP 内步。

梯度目标是连续力学松弛：从实际剩余材料的反力锥缺口选择每 pose 最多三个关键原始载荷，并给已通过的 pose 保留一个监测载荷；选择最多 128 个潜在有益接触。每次目标求值重新求解无上限非负反力、七维平衡残差与连续接触获取代价。第七维保留整体不上抬约束。距离场/接触锁仅用于引导，不是材料或净空证书。

每轮局部优化后，沿所得方向回溯六个步长（1 到 1/32）。每个试探都从不可变 Step3.3 材料重新构造，检查完整连续退出、每侧 1% 净空、材料分割及全部原始 32,768 条载荷。保护已经全通过的 pose；按最差 pose 和总通过比例接受改善；比例相同可接受同一扩大关键载荷集上的真实反力锥缺口下降。接受后重新建立局部模型。失败则回退并报告停滞。

Step4.1 几何未决不会被解释成物理无解：先尝试精确重建；若失败，保留其保存的实际接触受力模型作起点引导。后续只能接受几何检查通过的重建。几何解析成功且保存载荷不退步时允许完成起点恢复。没有解析几何时，报告 geometry_constructed/clearance_certified=False，即使受力通过也不计为 force_exit_passed。

连通、实际支撑接地与强度仍未验收；所有报告 full_fixture_accepted=False。没有全局或局部最优保证，局部停滞不等于无解。历史结果保持不变，新实验位于 data/experiments。

运行失败组：

```sh
.venv/bin/python slides/Co-optimize/step4.2/run_batch.py --failed-only --workers 2 --iterations 8 --out slides/Co-optimize/data/experiments/local_descent_new_batch/B
```

B 的 7 个失败组试跑已完成：4 组部分改善，0 组全通过；所有已通过的 pose 保持通过。6 组停在真实回溯、1 组停在松弛优化。实测记录见 [实验结果](../data/experiments/local_descent_real_B_20261006/README.md)。这不是局部或全局无解证明。

实现：[local_descent.py](../helper_func/optimization/local_descent.py)、[force_descent.py](../helper_func/optimization/force_descent.py)。下方算法为历史对照。

---

# Candidate-local force-envelope gradient_descent

Default `--search force-descent` maintains one incumbent chain. Each round proposes 32 single-pose direction_choice candidates from the same incumbent. Both each raw proposal and its gradient_descent endpoint are checked using incremental contact locks and ALL original saved loads. No surrogate-only shortlist. Rank by minimum pose feasible fraction, then summed feasible fractions; keep the better raw/descent state per candidate. The incumbent also competes, so a round cannot worsen this ranking. This ranking is not a guarantee that every individual load or pose improves.

Previous descent minimized a frozen weighted contact-geometry cost and shared an incumbent-centered sweep linearization across candidates. Its analytic derivatives were correct for that surrogate, but it was not minimizing physical infeasibility. Actual hard contact locks switch discretely: while the active contact set stays unchanged, the unbounded wrench cone and real loss stay unchanged. A classical smooth gradient of true binary-contact feasibility therefore cannot guide motion through the switches. Scaling unbounded cone rays by soft availability also cannot solve this: positive ray scaling leaves the cone unchanged.

Current descent builds a tangent chart and full-exit sweep linearization at EACH candidate's own direction_choice endpoint. Select one critical original load per pose and up to 128 helpful contact generators, plus generators carrying the projected loads. At every objective call, re-solve nonnegative unbounded contact/floor reactions with L1 equilibrium residual slack and a smooth contact-acquisition penalty. The normalized soft worst-load objective differentiates through the optimized value via reaction-based envelope sensitivities. Reactions are not frozen. No physical force capacities are introduced. These costs and slacks are optimization guidance, not final feasibility.

SLSQP runs at most 12 steps in a 12-degree local trust region. Check analytic derivatives with a central-difference directional test; clip trust-radius numerical overshoot and backtrack invalid/nonimproving endpoints. This is bounded local minimization of a relaxed, locally linearized force objective, not proof of an exact binary-contact local minimum. True all-load scoring decides whether descent helped. A worsened true score restores the raw proposal.

Reports are contact-model reports only: geometry_constructed=False, clearance_certified=False, full_fixture_accepted=False. Finite seed points, reverse-ray obstruction and all original loads are retained; whole contact patches/cores and final support clearance remain separate. Experiment records include direction_choice, gradient_descent, raw/descent counts, rollback, relaxed loss, derivative error, timings and selected round states.

Implementation: [force_descent.py](../helper_func/optimization/force_descent.py), [force_candidate_chain.py](../helper_func/optimization/force_candidate_chain.py). `--candidates 32 --max-proposals 96` is three rounds. Historical search modes remain explicit comparisons. Measurements and limitations: data/experiments/descent_review and data/experiments/force_descent_chain.

# Earlier contact-lock and surrogate variants (historical)

# Incremental contact-lock search

Default `--search contact-chain` maintains one chain and proposes 32 cheap single-pose direction_choice candidates per round, each followed by joint frozen-model gradient_descent. Search does not build swept solids or Boolean-cut support.

Each immutable Step3.3 seed contact point has a lock bit per pose. A pose locks the point if its exit violates local normal compatibility or a continuous reverse ray from the point intersects the object before the checked exit length. Rays start an object-extent-scaled epsilon outward to avoid self-intersection. Changed directions update only the corresponding rows; unchanged pose locks are reused. A point is active only when its lock count is zero. Killing/releasing one pose's lock cannot release a point still locked by another pose.

Rank the 32 candidate descent endpoints by the shared 64-probe physics/sweep surrogate. Check the top finalist's updated contact set against all original saved six-dimensional loads using unchanged floor reactions and batched force-cone LP certificates. Commit only a strict real working-load deficit improvement preserving already feasible poses, or full contact-model feasibility. Otherwise keep the incumbent. Candidate ranking is approximate; top-only checking can stagnate even if another candidate would help.

This is a finite seed-contact-point optimization model, not complete patch geometry or contact-core/1% clearance certification. No support mesh is exported by contact-chain. Any final support construction remains a separate later task; do not claim a full fixture from contact-model force feasibility. Reports explicitly distinguish contact_model_force_passed, geometry_constructed, clearance_certified and full_fixture_accepted.

Timing: `data/candidate_timing.json`; per-pose locks, lock counts and active points: `data/contact_locks.npz`; killed/released contacts: `data/chain_trajectory.json`. Main implementation: [contact_lock_chain.py](../helper_func/optimization/contact_lock_chain.py). Historical exact modes below are explicit comparison modes, not the current default.

# Fast candidate evaluation

Default `--search fast-candidate` maintains one sampling chain and proposes 32 single-pose direction_choice candidates per round. Shared preparation selects up to 64 contact probes valued by the incumbent's actual force-cone deficits and linearizes their full-exit distance field once. Every candidate independently optimizes this frozen surrogate with joint constrained SLSQP (up to 20 iterations), without solid construction or all-load classification. The analytic surrogate derivative is covered by finite-difference tests.

Rank candidate endpoints by surrogate loss, then exactly validate only the top `--exact-finalists` candidates (default 1). Real construction still checks original loads, exit clearance, floors, contact cores, partition and endpoint separation. Commit only a strict real working-load cone-deficit improvement which preserves fully feasible poses, or full feasibility. Otherwise keep the incumbent. The frozen field model is guidance, never collision or force acceptance; its ranking is approximate. A rejected top candidate may hide a useful lower-ranked candidate when the finalist budget is one.

Timing separates shared guidance, direction_choice, gradient_descent, and exact finalist construction/comparison in data/candidate_timing.json. `--max-proposals 96` means three rounds of 32 candidates. `--iterations` belongs to historical exact-descent modes; fast-candidate uses its fixed 20-iteration cheap inner optimization. `candidate-chain` retains the previous expensive exact-per-candidate comparison; published shared results remain historical.

# One chain with 32 direction_choice candidates

Default search is `candidate-chain`: one persistent incumbent, initially every pose's native up. Each round proposes 32 candidates from that SAME incumbent. Each candidate changes exactly one pose direction, choosing a signed tangent axis and a 5–30 degree angle inside its legal hemisphere. Pose exposure is balanced in shuffled cycles across candidates.

Each candidate independently performs joint `gradient_descent` (default one step, matching the paper's short candidate evaluation). Candidates are temporary branches, not 32 persistent chains. Compare candidate endpoints against the incumbent using the SAME accumulated working set of original saved loads and maximum real cone-projection deficit. Choose the candidate with greatest strict deficit reduction, preserving any pose that was fully feasible at the incumbent. Full all-load feasibility wins immediately. If no candidate improves, retain the incumbent. Candidate descent can explore geometry progress internally; geometry-only progress cannot replace the incumbent.

The working set is a guidance/comparison approximation, not a proof that every load's deficit decreases. Every exact candidate still checks all saved loads, continuous full exits, native-floor hemispheres, 1% per-side clearance, endpoint separation and partition consistency. Connectivity, installed ground coverage and strength are deferred. No exported-model replay.

Terminology: `sampling` refers to the persistent chain; `direction_choice` proposes a single-pose direction; `gradient_descent` adjusts all pose directions. A round is proposal, candidate descent, comparison and incumbent selection. `--candidates` defaults to 32, `--iterations` controls descent steps per candidate, `--max-proposals` counts total candidate attempts (96 means three complete rounds), and `--seed` controls randomness. Historical `single-pose` and `shared` modes remain explicit comparisons, not the default.

All numerical artifacts live under a fresh result directory's `data/`. `chain_trajectory.json` records candidates and selected round endpoints; `candidate_round_in_progress.json` checkpoints the current round. `optimization_trace.json` records actual candidate descent trials. `directions.npz`, `continuation_directions.npz` and final material refer to the selected incumbent, never an unselected temporary candidate. Prior published results are preserved.

Implementation: [candidate_chain.py](../helper_func/optimization/candidate_chain.py). This borrows the propose–local-optimize–evaluate pattern from D4Descent §3.3; changing an existing direction is not a structural grammar rewrite. Shared support couples poses, so only one candidate is committed per round rather than combining independently evaluated proposals.
