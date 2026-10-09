"""
inference.py

Closed-loop ReplayVLA inference (e.g. LIBERO eval). `ReplayMemoryBuffer` keeps an episode's memory and builds exactly
the memory inputs the model saw in training: frame selection comes from `replay_frame_indices`, the same function the
training data pipeline mirrors.

Cost per step, following MemoryWAM's cached inference:
    - anchor + short-term frames go through the vision backbone with the current frame (as in training)
    - when a grid frame becomes a memory frame, it is encoded ONCE (ViT -> GistEncoder.add_frame against the gist
      encoder's key/value cache), and its pixels are dropped. Only its gists are kept; older gists are never recomputed.
    - the gist encoder's cache holds the anchors, every past frame's gists, and the patches of only the newest
      `gist_n_recent` memory frames (see memory_bank/gist_encoder.py)

Like MemoryWAM, gists are never evicted: the LLM sees every memory frame's gists, however long the episode. These are
exactly the gists training computes over the whole bank. (Training's `max_memory_frames` only sizes batches; LIBERO-10
episodes need at most 61 memory frames at stride 8, so training never drops one either. A longer episode at eval would
give the LLM more gist tokens than it saw in training.)

Step counting: by default every policy step advances t (the eval loop's step counter after the simulator's warm-up).
Training data (`*_no_noops`) dropped demo steps whose action was ~0 with an unchanged gripper; set `noop_threshold`
to drop the matching steps from the history at eval too (a sensitivity knob; predicted actions are binned, so an
exact-zero test would almost never fire). See planning/PLAN.md, "Open issues".
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .frame_indices import memory_frames_before, replay_frame_indices
from .gist_encoder import GistCache
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
        self.n_anchor = getattr(cfg, "n_anchor", 1)
        self.noop_threshold = noop_threshold
        self.reset()

    def reset(self) -> None:
        """Call at the start of every episode."""
        self.t = -1
        self.pixels: Dict[int, torch.Tensor] = {}     # frame index -> [C, H, W] (anchors + grid frames not yet memory)
        self.cache: Optional[GistCache] = None        # gist encoder key/value cache (built at the first memory frame)
        self.memory_frames: List[int] = []            # frames already in memory, oldest first
        self.gists: List[torch.Tensor] = []           # their gists [G, gist_dim], computed once each
        self.prev_action: Optional[np.ndarray] = None

    def add_frame(self, pixel_values: torch.Tensor) -> int:
        """Register the current observation (processed pixels [C, H, W] or [1, C, H, W]); returns its step t."""
        self.t += 1
        if self.t % self.stride == 0:                  # a grid frame: anchor, or needed later as short/memory
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
    def _add_memory_frame(self, frame: int) -> None:
        """Encode a frame that just entered memory: ViT once, then only its own gists against the cache."""
        encoder = self.model.gist_encoder
        if self.cache is None:   # all anchors exist by the time the first frame enters memory
            anchors = [k * self.stride for k in range(self.n_anchor)]
            features = self.model._featurize(torch.stack([self.pixels[f] for f in anchors]))[None]  # [1, A, P, Dv]
            self.cache = encoder.init_cache(features, torch.tensor([anchors], device=self.device))
        features = self.model._featurize(self.pixels.pop(frame)[None])                  # [1, P, vision_dim]
        timestep = torch.tensor([frame], dtype=torch.long, device=self.device)
        self.gists.append(encoder.add_frame(self.cache, features, timestep)[0])        # [G, gist_dim]
        self.memory_frames.append(frame)

    def model_inputs(self) -> Dict[str, torch.Tensor]:
        """Memory kwargs for ReplayVLAForActionPrediction.forward / predict_action at the current step t."""
        assert self.t >= 0, "add_frame() first"
        idx = replay_frame_indices(self.t, self.stride, self.n_short, self.max_memory, self.n_anchor)
        entered = memory_frames_before(self.t, self.stride, self.n_short, self.n_anchor)   # uncapped, oldest first
        assert entered[: len(self.memory_frames)] == self.memory_frames, "memory frames must arrive in order"
        for frame in entered[len(self.memory_frames) :]:
            self._add_memory_frame(frame)

        zeros = torch.zeros_like(self.pixels[0])
        anchors = torch.stack([self.pixels[f] if v else zeros for f, v in zip(idx.anchors, idx.anchor_valid)])
        short = torch.stack([self.pixels[f] if v else zeros for f, v in zip(idx.short, idx.short_valid)])

        frames = self.memory_frames                                                     # every gist, never evicted
        if self.gists:
            memory_gists = torch.stack(self.gists)                                      # [M, G, gist_dim]
        else:                                                                           # same padding rule as ReplayCollator
            enc = self.model.gist_encoder
            memory_gists = torch.zeros(1, enc.n_gist, enc.d, device=self.device, dtype=self.dtype)
        return dict(
            anchor_pixel_values=anchors[None],
            anchor_valid=torch.tensor([idx.anchor_valid], device=self.device),
            short_pixel_values=short[None],
            short_valid=torch.tensor([idx.short_valid], device=self.device),
            memory_gists=memory_gists[None],
            memory_valid=torch.tensor([[True] * len(frames) or [False]], device=self.device),
            memory_timesteps=torch.tensor([frames or [0]], dtype=torch.long, device=self.device),
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
