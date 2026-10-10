# Fast contact-area sensitivity pilot — B, 2026-10-07

Started from saved Step4.1 directions. For each original contact triangle, four equal-area quadrature cells represent contact availability. Continuous reverse-ray obstruction and initial normal compatibility determine per-pose locks. Available cells are the complement of the union of all pose locks; each perturbation reports released and newly locked area, then reprojects the fixed worst original wrench onto the updated unbounded contact cone. Area is not a contact-force capacity. Force/moment scaling and seventh no-uplift coordinate remain unchanged.

No full solid reconstruction or all-load classification occurs during gradient differences. Full initial reconstruction occurred once per case; only locally improving candidates would receive complete geometry/all-load acceptance. No such candidate emerged in these two pilots. These are finite quadrature proxies, not exact contact area derivatives or fixture certificates.

| Group | Gradient times (0.25° / 1°) | Total runtime | Outcome |
|---|---|---|---|
| pose1+2+3+4+7+27 | 0.034 / 0.036 s | 8.69 s | Both gradients zero; unchanged pose2 32638/32768 |
| pose1+2+4+5+6+11 | 0.034 / 0.030 s | 8.14 s | Nonzero gradient; all backtracking candidates have unchanged worst-wrench proxy deficit; no update |

The second case released some area, but those rays did not reduce the selected wrench deficit. No efficacy claim follows from these pilots. Timing reflects four quadrature points per triangle and the local machine. Exact support connectivity/installed floor support remain deferred. Prior exact-geometry difference pilots were stopped by the user and remain separate interrupted experiments.

105 unit tests passed, including union-lock release/new-lock accounting, losing a necessary reaction, and verifying gradient computation without a full geometry method.
