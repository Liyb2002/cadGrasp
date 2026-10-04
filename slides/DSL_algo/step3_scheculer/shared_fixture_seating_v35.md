# Complete shared-world-exit fixtures on current B poses

The nine distinct requested combinations now have complete connected Step4 supports and Step5 PNGs. Native object meshes, poses, original loads and cached contact patches remain fixed. Fixture bases and offsets are seating variables, as in the existing Step4 pipeline; setting them all to identity/zero was an extra restriction and is not the intended method. The earlier shared_volume_comparison_v33 0/9 result applies only to that rejected restricted experiment and is superseded by these free-seating results.

Construction uses the copied EnvelopeGrow with RecoveryGrow contact-start/routing proposals, exact full-root/work-surface and continuous full-object-sweep acceptance, original floor-demand coverage, contact preservation, one connected closed solid and complete 5 mm cores. Exactly one in-memory construction acceptance is used. Force authority remains the full 32768-load native Step3 proof including the shared no-uplift equation. Every pose's object-exit unit vector is identical in native world XYZ; changing the fixture placement does not change the object pose or its force generators.

Historical Step4 transforms provide unaccepted proposal seeds. Current meshes and roots are checked again; incompatible seeds are repaired with the existing translation-separation LP and exact sweep Booleans, followed by bounded separated seating proposals. Historical heads/models/certificates are never reused as current acceptance.

Entry: `PYTHONPATH=slides/DSL_algo OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m step3_scheculer.build_shared_fixture_v35 --sets pose1+3 pose3+6 --jobs 2`. Seven vertical-exit groups retain identical accepted v34 witnesses; two horizontal-exit groups use v35, which supplies the metadata expected by the legacy convex-head withdrawal adapter without altering collision tolerances or geometry. The shared_direction_batch_v33 source contact programs are hash checked.

Step4 stage folders contain the exact OBJ, geometry certificate, per-pose contacts/state, two PNGs, construction reports and all rejected proposals. Step5 is measurement only: aggregate AABB of all original object poses plus every installed pose of the completed support. Material volume is reported separately. Historical public Step4 files remain untouched.

Selected solutions: pose1+3 and pose3+6 share +Y in world XYZ; seven other sets share +Z. All nine pass complete construction. With the identical bounded free-seating protocol for the +Z comparison, the +Y results reduce actual Step5 XYZ occupied-box volume by 8.4984% and 17.5400% respectively; seven +Z groups use the same baseline solution and have zero difference. This is a comparison of feasible bounded-search results, not a global optimum or material-volume claim.

[All Step4/Step5 full-resolution PNGs and models](../output/B/pose6+8+10+19/step5_evaluate/shared_fixture_seating_v35/gallery.md).

Construction plus initial images for the final successful runs total 89.9813 s (5.7220–15.3146 s per group); retries and subsequent presentation refreshes are separate. The independent +Z comparison takes 10.4447 s including full original-load CPU acceptance for its two new head programs. Fourteen direction/path/Step5 regression tests pass. Final provenance audits check recorded source and image hashes; there is no exported-geometry replay.
