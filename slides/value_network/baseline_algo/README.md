# 新目标入口：绝对方向 / Step5 占用空间

运行 `run_absolute_direction.py`，结果在 `../absolute_direction/output/B/`。当前是共同工位 +Z 方向下的联合接触／摆放和真实 Step4/Step5 试验，没有头数惩罚；[分数定义](../method/method.md) 和 [实际结果](../absolute_direction/REPORT.md)。下方 `run_step3.py` 和旧网络继续保留为历史版本，不能解释为新版 value 网络已经训练。

# Value-network baseline copy

This is a full copy of `slides/baseline_algo/`. The original remains the comparison baseline. Object meshes and poses still come from the shared `objects/` directory.

The active Step3 uses the complete selected-head mask and task membership to predict all candidate values in one neural forward pass. It chooses the smallest predicted `Q(S,h) = N_remaining + D_final`, after exact Step2/no-repeat/common-path masks and a learned recorded-completion coverage gate. The coverage gate does not prove infeasibility. Mechanical equilibrium and overall no-uplift are final acceptance checks; incomplete intermediate states remain searchable. There is no forced shared old head or three-head minimum.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python slides/value_network/baseline_algo/run_step3.py
```

The copied `step3_scheculer/scheduler.py` and `run_independent.py` command-line entries also run this network version. Older helper functions and historical programs remain available for reuse. The supported neural scope is object B and the ten recorded pose groups, not arbitrary objects.

New results are in `output/B/value_network_step3/`: summary `report.json`, each group's `schedule.json`, action trace, and `final_contacts_pose_N.npz`. The other copied output directories contain historical baseline results, not neural results. Step4 has not been rerun or built for these new contacts.

The final checkpoint is `../train/current/best.pt`. Actual inference does not read a completion bank or label table; it uses copied candidate geometry, neural outputs, and baseline mechanical acceptance. Previously verified path and mechanical evidence is hash-checked/reused for identical fixed inputs. Ten of ten groups pass, with head counts and exit-direction costs matching the best recorded completions. This is a fixed-task fit after collecting corrective search states, not a generalization result or proof of global optimality.
