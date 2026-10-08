"""
inference.py

Closed-loop ReplayVLA inference (e.g. LIBERO eval). `ReplayMemoryBuffer` keeps an episode's frames and builds exactly
the memory inputs the model saw in training: frame selection comes from `replay_frame_indices`, the same function the
training data pipeline mirrors.

Cost: anchor + short-term frames are re-encoded every step (as in training, they go through the vision backbone with
the current frame); memory frames are encoded ONCE, when they first become memory, and their patch features are cached.

Step counting: by default every policy step advances t (the eval loop's step counter after the simulator's warm-up).
Training data (`*_no_noops`) dropped demo steps whose action was ~0 with an unchanged gripper; set `noop_threshold`
to drop the matching steps from the history at eval too (a sensitivity knob; predicted actions are binned, so an
exact-zero test would almost never fire). See planning/PLAN.md, "Open issues".
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from .frame_indices import replay_frame_indices
from .modeling import ReplayVLAForActionPrediction


class ReplayMemoryBuffer:
    def __init__(
        self,
        model: ReplayVLAForActionPrediction,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        noop_threshold: Optional[float] = None,
    ) -> None:
        cfg = model.config
        self.model, self.device, self.dtype = model, device, dtype
        self.stride, self.n_short, self.max_memory = cfg.memory_stride, cfg.n_short, cfg.max_memory_frames
        self.noop_threshold = noop_threshold
        self.reset()

    def reset(self) -> None:
        """Call at the start of every episode."""
        self.t = -1
        self.pixels: Dict[int, torch.Tensor] = {}     # frame index -> [C, H, W] (anchor + grid frames still "short")
        self.features: Dict[int, torch.Tensor] = {}   # frame index -> [P, vision_dim] (grid frames that became memory)
        self.prev_action: Optional[np.ndarray] = None

    def add_frame(self, pixel_values: torch.Tensor) -> int:
        """Register the current observation (processed pixels [C, H, W] or [1, C, H, W]); returns its step t."""
        self.t += 1
        if self.t == 0 or self.t % self.stride == 0:   # the anchor, or a grid frame (needed later as short/memory)
            self.pixels[self.t] = pixel_values.reshape(-1, *pixel_values.shape[-2:]).to(self.device, self.dtype)
        return self.t

    def record_action(self, action: np.ndarray) -> None:
        """Optionally drop the current step from the history if its action is a near no-op (see module docstring)."""
        if self.noop_threshold is not None and self.t > 0:
            gripper_same = self.prev_action is None or action[-1] == self.prev_action[-1]
            if np.linalg.norm(action[:-1]) < self.noop_threshold and gripper_same:
                self.pixels.pop(self.t, None)
                self.t -= 1
        self.prev_action = np.asarray(action)

    @torch.no_grad()
    def _memory_features(self, frame: int) -> torch.Tensor:
        if frame not in self.features:
            self.features[frame] = self.model._featurize(self.pixels[frame][None])[0]
        return self.features[frame]

    def model_inputs(self) -> Dict[str, torch.Tensor]:
        """Memory kwargs for ReplayVLAForActionPrediction.forward / predict_action at the current step t."""
        assert self.t >= 0, "add_frame() first"
        idx = replay_frame_indices(self.t, self.stride, self.n_short, self.max_memory)
        zeros = torch.zeros_like(self.pixels[0])

        short = torch.stack([self.pixels[f] if v else zeros for f, v in zip(idx.short, idx.short_valid)])
        real_memory = [f for f, v in zip(idx.memory, idx.memory_valid) if v]
        M = max(1, len(real_memory))                    # same padding rule as ReplayCollator
        feats = torch.zeros(M, self.model.num_patches, self.model.vision_backbone.embed_dim, device=self.device, dtype=self.dtype)
        timesteps = torch.zeros(M, dtype=torch.long, device=self.device)
        valid = torch.zeros(M, dtype=torch.bool, device=self.device)
        for slot, frame in enumerate(real_memory):
            feats[slot] = self._memory_features(frame).to(self.dtype)
            timesteps[slot], valid[slot] = frame, True

        # Grid frames that are now memory no longer need their pixels (only their cached features)
        for frame in [f for f in self.pixels if f != 0 and f in self.features]:
            del self.pixels[frame]

        return dict(
            anchor_pixel_values=self.pixels[0][None],
            short_pixel_values=short[None],
            short_valid=torch.tensor([idx.short_valid], device=self.device),
            memory_features=feats[None],
            memory_valid=valid[None],
            memory_timesteps=timesteps[None],
        )


# === LIBERO / robot eval glue (mirrors experiments/robot/openvla_utils.py) ===
def get_replayvla(cfg) -> ReplayVLAForActionPrediction:
    """Load a merged ReplayVLA checkpoint (e.g. a training run dir) for eval, with its dataset statistics."""
    from experiments.robot.openvla_utils import DEVICE

    from .modeling import register_replayvla
    from .training import load_replayvla

    register_replayvla()
    vla = load_replayvla(cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)
    vla.replay_buffer = ReplayMemoryBuffer(vla, DEVICE, noop_threshold=getattr(cfg, "noop_threshold", None))
    return vla


def get_replayvla_action(vla, processor, obs, task_label, unnorm_key, center_crop=False) -> np.ndarray:
    """Same image/prompt handling as OpenVLA's get_vla_action, plus the episode's memory."""
    from experiments.robot.openvla_utils import DEVICE, crop_and_resize

    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:  # trained with random-crop augmentation -> center crop at test time (same as OpenVLA)
        import tensorflow as tf

        img = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(image)), tf.float32)
        img = tf.clip_by_value(crop_and_resize(img, 0.9, 1), 0, 1)
        image = Image.fromarray(tf.image.convert_image_dtype(img, tf.uint8, saturate=True).numpy()).convert("RGB")

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    buffer: ReplayMemoryBuffer = vla.replay_buffer
    buffer.add_frame(inputs["pixel_values"][0])
    action = vla.predict_action(**inputs, **buffer.model_inputs(), unnorm_key=unnorm_key, do_sample=False)
    buffer.record_action(action)
    return action
