"""
data.py

ReplayVLA data pipeline on top of OpenVLA's RLDS loader (TF reads/decodes/augments, PyTorch consumes):

    ReplayRLDSDataset         RLDSDataset that gathers anchor / memory / short / current frames per step
                              (via memory_bank.rlds_transforms.replay_chunk_obs)
    ReplayRLDSBatchTransform  RLDSBatchTransform that also emits the memory tensors
    ReplayCollator            PaddedCollatorForActionPrediction that also stacks/pads the memory tensors

To keep CPU and memory cost proportional to the real history, each example carries only its REAL memory frames;
the collator pads memory to the largest count in the batch (not always to max_memory). Empty slots stay at the end,
as the gist encoder expects.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
import torch
from PIL import Image

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset

from .rlds_transforms import replay_chunk_obs


@dataclass
class ReplayRLDSBatchTransform(RLDSBatchTransform):
    n_short: int = 4
    max_memory: int = 64
    n_anchor: int = 1

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        images = rlds_batch["observation"]["image_primary"]          # [W, H, W_img, 3] uint8, W = A + M + S + 1
        valid = rlds_batch["observation"]["pad_mask"]                # [W] bool
        frame_idx = rlds_batch["observation"]["replay_frame_index"]  # [W] int
        A, M, S = self.n_anchor, self.max_memory, self.n_short - 1
        assert S >= 1, "n_short must be >= 2 (current frame + at least one short-term frame)"
        assert images.shape[0] == A + M + S + 1, f"window {images.shape[0]} != {A} + {M} + {S} + 1"

        # Text, labels and the current frame: exactly OpenVLA's transform, applied to the current frame only
        current_only = dict(rlds_batch, observation=dict(rlds_batch["observation"], image_primary=images[-1:]))
        out = super().__call__(current_only)

        to_pixels = lambda img: self.image_transform(Image.fromarray(img))  # noqa: E731
        mem_valid = np.asarray(valid[A : A + M], dtype=bool)
        n_mem = int(mem_valid.sum())
        assert mem_valid[:n_mem].all(), "memory slots must be filled oldest-first, empty slots at the end"

        out["anchor_pixel_values"] = torch.stack([to_pixels(img) for img in images[:A]])         # [A, C, H, W]
        out["anchor_valid"] = torch.as_tensor(valid[:A], dtype=torch.bool)
        out["short_pixel_values"] = torch.stack([to_pixels(img) for img in images[A + M : A + M + S]])
        out["short_valid"] = torch.as_tensor(valid[A + M : A + M + S], dtype=torch.bool)
        out["memory_pixel_values"] = [to_pixels(img) for img in images[A : A + n_mem]]  # real frames only
        out["memory_timesteps"] = torch.as_tensor(frame_idx[A : A + n_mem], dtype=torch.long)
        return out


class ReplayRLDSDataset(RLDSDataset):
    def __init__(
        self,
        *args: Any,
        memory_stride: int = 8,
        n_short: int = 4,
        max_memory: int = 64,
        n_anchor: int = 1,
        shuffle_buffer_size: int = 20_000,  # each element holds ~n_anchor + max_memory + n_short encoded JPEGs
        **kwargs: Any,
    ) -> None:
        self.memory_stride, self.n_short, self.max_memory, self.n_anchor = memory_stride, n_short, max_memory, n_anchor
        super().__init__(*args, shuffle_buffer_size=shuffle_buffer_size, **kwargs)

    def make_dataset(self, rlds_config):
        rlds_config["traj_transform_kwargs"]["chunk_fn"] = partial(
            replay_chunk_obs, stride=self.memory_stride, n_short=self.n_short, max_memory=self.max_memory,
            n_anchor=self.n_anchor,
        )
        return super().make_dataset(rlds_config)


@dataclass
class ReplayCollator(PaddedCollatorForActionPrediction):
    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        output = super().__call__(instances)
        B = len(instances)

        output["anchor_pixel_values"] = torch.stack([x["anchor_pixel_values"] for x in instances])   # [B, A, C, H, W]
        output["anchor_valid"] = torch.stack([x["anchor_valid"] for x in instances])
        output["short_pixel_values"] = torch.stack([x["short_pixel_values"] for x in instances])
        output["short_valid"] = torch.stack([x["short_valid"] for x in instances])

        # Memory: pad to the largest number of real frames in the batch (>= 1 so shapes stay non-empty)
        counts = [len(x["memory_pixel_values"]) for x in instances]
        M = max(1, max(counts))
        frame_shape = output["anchor_pixel_values"].shape[2:]
        memory = torch.zeros(B, M, *frame_shape, dtype=output["anchor_pixel_values"].dtype)
        timesteps = torch.zeros(B, M, dtype=torch.long)
        valid = torch.zeros(B, M, dtype=torch.bool)
        for b, (x, n) in enumerate(zip(instances, counts)):
            if n > 0:
                memory[b, :n] = torch.stack(x["memory_pixel_values"])
                timesteps[b, :n] = x["memory_timesteps"]
                valid[b, :n] = True
        output["memory_pixel_values"], output["memory_timesteps"], output["memory_valid"] = memory, timesteps, valid
        return output
