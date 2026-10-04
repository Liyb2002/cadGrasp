# Absolute-direction DSL comparison

Primary: aggregate XYZ occupied bounding-box volume of all saved object and installed support poses. Reaction angles use area-weighted surface normals in the physical fixture frame. Head count is not optimized. All original 32768 loads per task pass; one construction acceptance, no exported-body geometry replay.

| Set | Old cm³ | New cm³ | Reduction | Reaction angle | Exit angle |
|---|---:|---:|---:|---:|---:|
| pose1+2+3+4+5 | 122495.88 | 24796.45 | 79.76% | 83.2 → 44.5° | 77.4 → 0.0° |
| pose1+2+8+17 | 107455.32 | 26753.11 | 75.10% | 73.6 → 28.2° | 54.0 → 0.0° |
| pose1+3 | 5792.76 | 5792.76 | 0.00% | 87.2 → 87.2° | 110.9 → 110.9° |
| pose1+3copied | 5792.76 | 5792.76 | 0.00% | 87.2 → 87.2° | 110.9 → 110.9° |
| pose2+10+15 | 48821.86 | 8287.99 | 83.02% | 97.1 → 35.5° | 74.1 → 0.0° |
| pose2+12+15 | 33970.83 | 8761.45 | 74.21% | 102.9 → 25.2° | 68.7 → 0.0° |
| pose2+9+13+15+17 | 126578.82 | 41537.05 | 67.18% | 83.8 → 24.2° | 71.6 → 0.0° |
| pose3+6 | 5272.21 | 5272.21 | 0.00% | 0.0 → 0.0° | 2.5 → 2.5° |
| pose5+7 | 6449.67 | 5836.70 | 9.50% | 125.8 → 89.0° | 177.9 → 0.0° |
| pose6+8+10+19 | 30605.14 | 18790.81 | 38.60% | 43.2 → 33.5° | 55.7 → 0.0° |
