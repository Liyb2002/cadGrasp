# Current Step4: fixed-envelope greedy growth

Entry: `step4_connect_support/run_envelope_growth.py`. Algorithm: `step4_connect_support/envelope_growth.md`. Grow one support, preferring low added box volume beyond the fixed initial object/demand/contact-start envelope, then short branches. Joins and ground coverage repair share the action choice; no prescribed feet or required head-first phase. Step3 and saved placements are unchanged.

Current model: `data/growing_support/shape.obj` and `report.json`. Public images: `overview.png` and `construction_steps.png`. Blue outlines show required coverage; panel 4 displays one object pose. Final XYZ volume and XY area are measured after growth.

Use one construction acceptance for contacts, work surfaces, all installed floors, connectivity, original ground demands and continuous 500 mm withdrawal sweeps. No export recheck or independent replay. Complete 5 mm rod/foot cores retain documented original-contact/short-transition exceptions. No global optimum or strength claim. Previous joint and fast models are archived under `data/history/before_envelope_growth/` and `data/history/before_joint_growth/`.
