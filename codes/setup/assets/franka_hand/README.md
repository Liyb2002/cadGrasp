# Franka Hand model

Unmodified model and mesh files from Google DeepMind MuJoCo Menagerie,
`franka_emika_panda`, revision `8161bba264d7fa7c99ca301e91e7fb44737676ad`:

https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda

Upstream Apache-2.0 license is included as `LICENSE`. The hand has two finger
joints with 0–40 mm travel each. This is the Panda Hand model, not an FR3 arm.
`grasp.py` builds the test scene in memory and supplies a force-limited closing
controller. No downloaded model file is modified by a run.
