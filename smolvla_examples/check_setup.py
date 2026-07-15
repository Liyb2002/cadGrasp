"""Sanity check: is the smolvla env wired up correctly?

Run:  conda run -n smolvla python smolvla_examples/check_setup.py

Verifies imports, MPS, loads smolvla_base, and does one real forward pass.
The first run downloads ~1.8 GB of weights (cached afterwards).
"""

import torch

from smolvla_runner import build_frame, describe_action_space, load_smolvla, predict_chunk, select_device


def main():
    print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
    device = select_device()
    print("using device:", device)

    print("loading lerobot/smolvla_base (first time downloads weights)...")
    policy, preprocess, postprocess = load_smolvla(device)
    print("loaded OK\n")
    print(describe_action_space(policy), "\n")

    frame = build_frame(policy, task="Grasp the red cube.")
    chunk = predict_chunk(policy, preprocess, postprocess, frame)
    print("forward pass OK -> action chunk shape:", chunk.shape)
    print("first predicted action (step 0):", chunk[0].round(4))
    print("\nSetup is working. Try:  python run_task.py --task grasp")


if __name__ == "__main__":
    main()
