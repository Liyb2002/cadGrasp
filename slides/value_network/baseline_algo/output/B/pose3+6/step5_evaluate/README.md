# pose3+6: Step5 occupied space

Compare aggregate bounding boxes of all objects and of all objects plus installed supports. Use saved workstation axes and placements; no recentering, placement optimization or motion sweeps.

| Geometry | XYZ dimensions (mm) | Box volume (cm3) | XY area (cm2) |
|---|---:|---:|---:|
| Objects | 151.416 x 134.746 x 190.681 | 3890.415 | 204.027 |
| Objects and supports | 174.246 x 216.955 x 195.903 | 7405.822 | 378.035 |

Extra XY area: 85.29%. Box volume measures occupied space, not material volume. This evaluation does not rerun force or geometry acceptance.

[Bounding-box comparison](bbox.png) - [Metrics and provenance](report.json)
