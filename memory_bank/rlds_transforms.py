"""
rlds_transforms.py

TF trajectory transform that gathers ReplayVLA's frames for every step of a trajectory. A drop-in replacement for
`chunk_act_obs` (plugged in via `apply_trajectory_transforms(..., chunk_fn=...)`), mirroring
`memory_bank.frame_indices.replay_frame_indices` exactly (tested in tests/test_replay_data.py).

Per step t, observations get a new axis (index 1) of length W = n_anchor + max_memory + (n_short - 1) + 1, in the order

    [anchors (0, s, …) | memory slots (oldest first, empty at the end) | short slots (age-aligned) | current (frame t)]

Images are still JPEG-encoded at this point (decoding happens later, per frame), so the gather is cheap.
`observation["pad_mask"]` [T, W] marks real (True) vs empty (False) slots; empty slots point at frame 0.
Actions become [T, 1, A]: only the current step's action, as with chunk_act_obs(window_size=1).
"""

from typing import Dict

import tensorflow as tf


def replay_window_indices(traj_len: tf.Tensor, stride: int, n_short: int, max_memory: int, n_anchor: int = 1):
    """Returns (indices [T, W] int32, valid [T, W] bool) for every step t of a trajectory of length traj_len."""
    n_past = n_short - 1
    t = tf.range(traj_len)                                                       # [T]
    n_grid = tf.maximum((t - 1) // stride, 0)                                    # largest j with j·s < t (0 if none)

    # Anchors: j = 0 … n_anchor-1; frame 0 always, j·s once it is in the past (j <= n_grid)
    a = tf.range(n_anchor)[None, :]                                              # [1, A]
    anchor_valid = a <= n_grid[:, None]                                          # [T, A]
    anchor_idx = tf.where(anchor_valid, a * stride, 0)

    # Short-term past: grid multipliers j = n_grid - n_past + 1 … n_grid (age-aligned; j < n_anchor means missing)
    k = tf.range(n_past)[None, :]                                                # [1, n_past]
    j_short = n_grid[:, None] - (n_past - 1) + k                                 # [T, n_past]
    short_valid = j_short >= n_anchor
    short_idx = tf.where(short_valid, j_short * stride, 0)

    # Memory: grid frames older than the short window (j = n_anchor … n_grid - n_past), newest `max_memory` kept
    n_old = tf.maximum(n_grid - n_past - n_anchor + 1, 0)                        # [T]
    n_keep = tf.minimum(n_old, max_memory)
    j_first = n_grid - n_past - n_keep + 1                                       # [T]
    m = tf.range(max_memory)[None, :]                                            # [1, M]
    memory_valid = m < n_keep[:, None]                                           # [T, M]
    memory_idx = tf.where(memory_valid, (j_first[:, None] + m) * stride, 0)

    ones = tf.ones_like(t[:, None], dtype=tf.bool)
    indices = tf.concat([anchor_idx, memory_idx, short_idx, t[:, None]], axis=1)
    valid = tf.concat([anchor_valid, memory_valid, short_valid, ones], axis=1)
    return indices, valid


def replay_chunk_obs(traj: Dict, stride: int = 8, n_short: int = 4, max_memory: int = 64, n_anchor: int = 1) -> Dict:
    traj_len = tf.shape(traj["action"])[0]
    indices, valid = replay_window_indices(traj_len, stride, n_short, max_memory, n_anchor)

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj["observation"])
    traj["observation"]["pad_mask"] = valid
    traj["observation"]["replay_frame_index"] = indices                         # which episode frame each slot holds
    traj["action"] = tf.gather(traj["action"], tf.range(traj_len)[:, None])     # [T, 1, A]
    return traj
