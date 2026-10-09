"""
frame_indices.py

Single source of truth for *which* frames ReplayVLA sees at policy step t. The training data pipeline
(memory_bank/rlds_transforms.py, in TF) mirrors this function, and the eval buffer (memory_bank/inference.py) calls it
directly, so training-time and inference-time memory are identical.

Frames live on a fixed grid anchored at the start of the episode: 0, s, 2s, 3s, … (s = stride). Using a fixed grid
(rather than t - s, t - 2s, …) means a frame's role never depends on when you look at it, so eval can cache features.

At step t (0-indexed frame of the episode; the current frame is frame t):
    anchors = the first `n_anchor` grid frames 0, s, …, (n_anchor-1)·s, full tokens (MemoryWAM's "sink window" of
              initial frames); frame 0 is always valid, a later anchor once it is in the past (k·s < t)
    current = frame t                                                  (always)
    short   = the (n_short - 1) newest non-anchor grid frames g < t, ALIGNED BY AGE:
              the last slot holds the newest (right next to the current frame), the first slot the oldest;
              missing ages (early in the episode) are invalid
    memory  = all older non-anchor grid frames, oldest first, keeping only the newest `max_memory`;
              empty slots at the END (gist encoder convention)
Invalid slots point at frame 0 so gathers stay in range; their validity flag is False.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ReplayFrameIndices:
    anchors: List[int]          # len n_anchor; 0, s, 2s, …
    anchor_valid: List[bool]
    current: int
    short: List[int]            # len n_short - 1; slot order oldest age → newest age
    short_valid: List[bool]
    memory: List[int]           # len max_memory; oldest first, empty slots at the end; also the frames' timesteps
    memory_valid: List[bool]


def grid_frames_before(t: int, stride: int, n_anchor: int = 1) -> List[int]:
    """Non-anchor grid frames strictly before t: n_anchor·s, (n_anchor+1)·s, … < t."""
    return list(range(n_anchor * stride, t, stride))


def memory_frames_before(t: int, stride: int = 8, n_short: int = 4, n_anchor: int = 1) -> List[int]:
    """Every frame that has entered memory by step t (oldest first, uncapped): grid frames older than the short window."""
    grid = grid_frames_before(t, stride, n_anchor)
    return grid[: len(grid) - min(n_short - 1, len(grid))]


def replay_frame_indices(
    t: int, stride: int = 8, n_short: int = 4, max_memory: int = 64, n_anchor: int = 1
) -> ReplayFrameIndices:
    assert t >= 0 and stride >= 1 and n_short >= 1 and max_memory >= 1 and n_anchor >= 1
    n_past = n_short - 1
    grid = grid_frames_before(t, stride, n_anchor)

    anchor_valid = [k == 0 or k * stride < t for k in range(n_anchor)]
    anchors = [k * stride if v else 0 for k, v in zip(range(n_anchor), anchor_valid)]

    # Short-term past: newest `n_past` grid frames, right-aligned so the newest sits in the last slot
    past = grid[-n_past:] if n_past > 0 else []
    pad = n_past - len(past)
    short = [0] * pad + past
    short_valid = [False] * pad + [True] * len(past)

    # Memory: everything older than the short-term window, newest `max_memory` kept, oldest first, padded at the end
    older = grid[: len(grid) - len(past)][-max_memory:]
    memory = older + [0] * (max_memory - len(older))
    memory_valid = [True] * len(older) + [False] * (max_memory - len(older))

    return ReplayFrameIndices(
        anchors=anchors, anchor_valid=anchor_valid, current=t, short=short, short_valid=short_valid, memory=memory, memory_valid=memory_valid
    )
