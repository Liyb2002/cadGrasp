"""Shared SmolVLA inference helpers (LeRobot 0.4.4, Apple-Silicon / MPS).

SmolVLA (`lerobot/smolvla_base`, 450M params) is a vision-language-action model:
    inputs  = one or more camera views + the robot's joint state + a language instruction
    output  = a *chunk* of future robot actions (here: 50 steps x 6-D joint targets)

This module hides the LeRobot plumbing so the task scripts stay short. The key
idea: we never hard-code the observation keys. We read them from the loaded
checkpoint's `config.input_features`, so the frame we build always matches what
*this* checkpoint expects (3x 256x256 cameras + a 6-D state for smolvla_base).

IMPORTANT HONESTY NOTE
----------------------
`smolvla_base` is a *base* model. It has NOT been fine-tuned on any grasp / poke /
insert task, and there is no robot or simulator attached here. So the action
numbers you get are *not* a competent skill -- they are a real, correctly-shaped
model prediction that shows how the interface works end to end. To get useful
motions you must fine-tune on ~50 teleop episodes of your task (see README).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_ID = "lerobot/smolvla_base"


def select_device(requested: str = "auto") -> torch.device:
    """Pick a torch device. On this Mac that means MPS, with a CPU fallback."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_smolvla(device: torch.device, model_id: str = MODEL_ID):
    """Load the policy + its pre/post-processing pipelines onto `device`.

    Returns (policy, preprocess, postprocess). The processors are loaded FROM the
    checkpoint, so they carry the correct tokenizer and the normalization stats
    the model was trained with. We override only the target device (the saved
    default is 'cuda', which doesn't exist here).
    """
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.reset()  # clear the internal action-chunk queue

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess, postprocess


def _draw_scene(label: str, color: tuple[int, int, int], size: int) -> torch.Tensor:
    """Make a tiny synthetic 'table + object' RGB frame so the demo needs no assets.

    The content is deliberately simple. Because smolvla_base isn't fine-tuned,
    the pixels don't change the *quality* of the output -- this just shows the
    exact format a real camera frame would take: float32, CHW, values in [0, 1].
    Swap this for a real image via `load_image_views(...)` when you have one.
    """
    img = Image.new("RGB", (size, size), (205, 200, 190))  # table-ish background
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(size * 0.62), size, size], fill=(150, 140, 120))  # table edge
    c = size // 2
    r = size // 6
    d.ellipse([c - r, c - r, c + r, c + r], fill=color)  # the "object"
    d.text((6, 6), label, fill=(20, 20, 20))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC in [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # -> CHW


def load_image_views(path: str, keys, size: int) -> dict[str, torch.Tensor]:
    """Load one real image and use it for every camera view the model expects."""
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return {k: t.clone() for k in keys}


def build_frame(policy, task: str, images: dict | None = None, state=None) -> dict:
    """Assemble a single (un-batched) raw observation frame for the preprocessor.

    Keys and shapes come straight from `policy.config.input_features`, so this
    adapts automatically to whatever cameras / state dim the checkpoint declares.
    The preprocessor will add the batch dim, tokenize `task`, normalize, and move
    everything to the device.
    """
    cfg = policy.config
    frame: dict = {}
    palette = [(200, 40, 40), (40, 90, 200), (40, 170, 80)]  # distinct per camera

    img_keys = [k for k, f in cfg.input_features.items() if f.type.name == "VISUAL"]
    for i, key in enumerate(img_keys):
        _, h, w = cfg.input_features[key].shape  # (C, H, W)
        if images is not None and key in images:
            frame[key] = images[key].to(torch.float32)
        else:
            frame[key] = _draw_scene(key.split(".")[-1], palette[i % len(palette)], h)

    for key, feat in cfg.input_features.items():
        if feat.type.name == "STATE":
            dim = feat.shape[0]
            if state is not None:
                frame[key] = torch.as_tensor(state, dtype=torch.float32)
            else:
                frame[key] = torch.zeros(dim, dtype=torch.float32)  # neutral pose

    frame["task"] = task
    return frame


def predict_chunk(policy, preprocess, postprocess, frame: dict) -> np.ndarray:
    """Run one forward pass and return the full action chunk as (n_steps, action_dim)."""
    policy.reset()
    obs = preprocess(frame)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(obs)  # (1, chunk_size, action_dim)
    chunk = postprocess(chunk)
    return chunk.squeeze(0).float().cpu().numpy()


def describe_action_space(policy) -> str:
    cfg = policy.config
    a = cfg.output_features["action"].shape[0]
    cams = [k for k, f in cfg.input_features.items() if f.type.name == "VISUAL"]
    return (
        f"inputs: {len(cams)} camera view(s) {cams} + {cfg.input_features['observation.state'].shape[0]}-D state\n"
        f"output: {cfg.chunk_size}-step chunk of {a}-D actions (joint-position targets)"
    )
