# Absolute-volume witnessed-continuation data

20 accepted Step4/Step5 completions supply 1,280 state samples and 21,302 positive state/action cost labels. There are 1192 unique within-group selected-head states; repeated empty/complete prefixes are retained. Another seed samples 320 evaluation states from the same pose groups, including repeated empty states; this is not unseen-group evaluation.

`states.npz` contains explicit selected-head masks, pose membership, witnessed-completion membership, actual terminal log-volume costs, tie order and XY seating targets. `manifest.json` lists candidate identities and source hashes. No unsearched action receives a failed-volume label. The completion-membership classifier is imitation evidence, not a feasibility classifier.

[Training and fresh construction](../../../train/absolute_volume/README.md).
