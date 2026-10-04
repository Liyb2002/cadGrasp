# pose3+6: Step5 occupied space

Compare aggregate bounding boxes of all objects and of all objects plus installed supports. Use saved workstation axes and placements; no recentering, placement optimization or motion sweeps.

| Geometry | XYZ dimensions (mm) | Box volume (cm3) | XY area (cm2) |
|---|---:|---:|---:|
| Objects | 151.416 x 134.746 x 190.681 | 3890.415 | 204.027 |
| Objects and supports | 288.236 x 189.947 x 190.681 | 10439.741 | 547.497 |

Extra XY area: 168.35%. Box volume measures occupied space, not material volume. This evaluation does not rerun force or geometry acceptance.

[Bounding-box comparison](bbox.png) - [Metrics and provenance](report.json)
