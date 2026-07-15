"""SmolVLA task examples: grasping, poking, insertion, and (the caveated) force.

Each "task" is just a different natural-language instruction handed to the SAME
model. That is the whole point of a VLA: you change behaviour with words, not
with a new network. This script builds an observation, runs SmolVLA, and prints
the predicted action chunk so you can see the interface work end to end.

Run:
    conda run -n smolvla python smolvla_examples/run_task.py --task grasp
    conda run -n smolvla python smolvla_examples/run_task.py --task poke
    conda run -n smolvla python smolvla_examples/run_task.py --task insert
    conda run -n smolvla python smolvla_examples/run_task.py --task force
    conda run -n smolvla python smolvla_examples/run_task.py --instruction "stack the two cubes"
    conda run -n smolvla python smolvla_examples/run_task.py --task grasp --image /path/to/photo.jpg

Reality check: smolvla_base is NOT fine-tuned, so the numbers are a correctly
shaped prediction, not a competent skill. See README.md.
"""

import argparse

import numpy as np

from smolvla_runner import build_frame, describe_action_space, load_image_views, load_smolvla, predict_chunk, select_device

# Each preset = an instruction + a note on what it demonstrates for cadGrasp.
TASKS = {
    "grasp": {
        "instruction": "Grasp the red cube and lift it off the table.",
        "note": "The canonical VLA skill. Watch the gripper dim (last action dim) close as the chunk progresses.",
    },
    "poke": {
        "instruction": "Poke the blue block with the gripper tip.",
        "note": "A reach-and-touch: mostly arm-approach motion, little/no gripper actuation.",
    },
    "insert": {
        "instruction": "Insert the peg into the hole.",
        "note": "A contact-rich alignment task. Position-only VLAs struggle here without force feedback.",
    },
    "force": {
        "instruction": "Push down on the object and press firmly until contact.",
        "note": (
            "CAVEAT: SmolVLA outputs POSITION targets, not forces, and takes no force/torque input.\n"
            "         'Force' is not something this model does. To get real force behaviour you must either:\n"
            "           (a) run these position targets through an impedance/admittance controller in a\n"
            "               torque-capable sim (e.g. MuJoCo), which converts position error -> joint torque, or\n"
            "           (b) fine-tune a checkpoint whose STATE vector includes force/torque sensor readings.\n"
            "         This preset still runs so you can see the position chunk you'd feed into (a)."
        ),
    },
}


def summarize_chunk(chunk: np.ndarray) -> None:
    """Print a compact view of the (n_steps, action_dim) chunk."""
    n, d = chunk.shape
    print(f"action chunk: {n} steps x {d} dims (joint-position targets)")
    print("  step 0 :", np.array2string(chunk[0], precision=3, suppress_small=True))
    print(f"  step {n // 2:<2}:", np.array2string(chunk[n // 2], precision=3, suppress_small=True))
    print(f"  step {n - 1:<2}:", np.array2string(chunk[-1], precision=3, suppress_small=True))
    print("  per-dim range over chunk:")
    for i in range(d):
        lo, hi = chunk[:, i].min(), chunk[:, i].max()
        tag = "  <- last dim (gripper on SO-100/101)" if i == d - 1 else ""
        print(f"    dim {i}: [{lo:+.3f}, {hi:+.3f}] delta={hi - lo:.3f}{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--task", choices=list(TASKS), help="preset task instruction")
    g.add_argument("--instruction", help="free-form instruction (overrides --task)")
    ap.add_argument("--image", help="optional real image used for all camera views")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    args = ap.parse_args()

    if args.instruction:
        instruction, note = args.instruction, "(free-form instruction)"
    else:
        preset = TASKS[args.task or "grasp"]
        instruction, note = preset["instruction"], preset["note"]

    device = select_device(args.device)
    print(f"device: {device}")
    print("loading lerobot/smolvla_base ...")
    policy, preprocess, postprocess = load_smolvla(device)
    print(describe_action_space(policy))
    print("\n" + "=" * 78)
    print(f"TASK       : {args.task or 'custom'}")
    print(f"INSTRUCTION: {instruction!r}")
    print(f"NOTE       : {note}")
    print("=" * 78)

    images = None
    if args.image:
        img_keys = [k for k, f in policy.config.input_features.items() if f.type.name == "VISUAL"]
        images = load_image_views(args.image, img_keys, policy.config.input_features[img_keys[0]].shape[1])
        print(f"(using real image {args.image} for {len(img_keys)} view(s))")

    frame = build_frame(policy, task=instruction, images=images)
    chunk = predict_chunk(policy, preprocess, postprocess, frame)
    print()
    summarize_chunk(chunk)
    print(
        "\nWhat you're seeing: a real SmolVLA prediction with the correct interface.\n"
        "It is NOT a trained skill (base model, no robot). Fine-tune on ~50 episodes\n"
        "of teleop data to make these motions actually solve the task -- see README.md."
    )


if __name__ == "__main__":
    main()
