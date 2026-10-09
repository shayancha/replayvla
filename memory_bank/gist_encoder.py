"""
gist_encoder.py

ReplayVLA's gist encoder: compresses each memory frame (256 ViT patch features) into a few learned "gist" tokens,
in the context of everything that came before it, following MemoryWAM's hybrid memory (arXiv 2606.20562).

When memory frame i was the newest frame, its gists may draw on the "entire historical context":
    - the anchor frames' patches (frame 0, plus later anchors 0 < k·s once they exist; MemoryWAM's sink frames)
    - frame i's own patches, and the patches of the `n_recent` frames before it (its recent window)
    - the gists of every frame <= i (the memory bank)
They never see later frames or empty slots, so memory only flows forward in time. This makes the whole bank trainable
in one parallel pass (no BPTT) and identical at training and inference.

Each layer is factorized (cheap, see planning/lessons/0007):
    (1) frame-local attention over [patches_i ; gists_i] (one sequence per frame; each anchor is its own sequence)
    (2) history attention: every frame's gists (queries) attend to anchor + recent patches + earlier gists (keys)
    (3) a per-token MLP
all pre-norm with residuals. Spec & tests: planning/milestones/m1a-gist-encoder.md, tests/test_gist_encoder.py.

Conventions: memory slots are ordered oldest-first with empty slots at the end; empty slots output exactly zero.

Two equivalent ways to run it:
    - `forward`: the whole memory bank in one parallel pass (training).
    - `init_cache` + `add_frame`: one new memory frame at a time against a per-layer key/value cache (closed-loop eval,
      as in MemoryWAM's inference). Frame i's gists depend only on frames <= i, so earlier gists never change and are
      never recomputed. The cache keeps, per layer, the anchors' keys/values, the gists' keys/values of every past
      frame (the long-term memory; never evicted, no cap, as in MemoryWAM) and the patches' keys/values of only the
      newest `n_recent` frames (older patches are evicted: their information survives in the gists).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# === Attention ===
class GistAttention(nn.Module):
    """Multi-head attention where queries and keys/values may come from different tensors (self- or cross-attention)."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        assert d % heads == 0, f"d ({d}) must be divisible by heads ({heads})"
        self.heads, self.head_dim = heads, d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, allowed: Optional[torch.Tensor]) -> torch.Tensor:
        """x_q: [N, Lq, d], x_kv: [N, Lk, d], allowed: bool [N, Lq, Lk] (True = may attend; None = all) -> [N, Lq, d]."""
        return self.attend(x_q, *self.project_kv(x_kv), allowed)

    def project_kv(self, x_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_kv: [N, Lk, d] -> keys, values [N, H, Lk, dh] (what the incremental cache stores)."""
        N, Lk, _ = x_kv.shape
        k = self.k_proj(x_kv).view(N, Lk, self.heads, self.head_dim).transpose(1, 2)   # [N, H, Lk, dh]
        v = self.v_proj(x_kv).view(N, Lk, self.heads, self.head_dim).transpose(1, 2)   # [N, H, Lk, dh]
        return k, v

    def attend(self, x_q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: Optional[torch.Tensor]) -> torch.Tensor:
        N, Lq, d = x_q.shape
        q = self.q_proj(x_q).view(N, Lq, self.heads, self.head_dim).transpose(1, 2)    # [N, H, Lq, dh]
        mask = None if allowed is None else allowed[:, None]                           # mask broadcast over heads
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(N, Lq, d))


class GistMLP(nn.Module):
    def __init__(self, d: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


# === One Factorized Layer ===
class GistEncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(d)
        self.local_attn = GistAttention(d, heads)
        self.history_q_norm = nn.LayerNorm(d)
        self.history_kv_norm = nn.LayerNorm(d)
        self.history_attn = GistAttention(d, heads)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = GistMLP(d, mlp_ratio)

    # Building blocks, shared by the parallel forward and the incremental (cached) path
    def local_block(self, x: torch.Tensor, allowed: Optional[torch.Tensor]) -> torch.Tensor:
        normed = self.local_norm(x)
        return x + self.local_attn(normed, normed, allowed)

    def history_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keys/values of post-local states [N, L, d] as history-attention keys."""
        return self.history_attn.project_kv(self.history_kv_norm(x))

    def mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.mlp_norm(x))

    def forward(
        self,
        anchor: torch.Tensor,          # [B, A, P, d]
        patches: torch.Tensor,         # [B, M, P, d]
        gists: torch.Tensor,           # [B, M, G, d]
        local_allowed: torch.Tensor,   # [B·M, P+G, P+G] bool
        history_allowed: torch.Tensor, # [B, M·G, A·P + M·P + M·G] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, P, d = patches.shape
        A, G = anchor.shape[1], gists.shape[2]

        # (1) Frame-local attention: fold (example, frame) into the batch; each frame = [its patches ; its gists]
        frames = torch.cat([patches, gists], dim=2).reshape(B * M, P + G, d)                # [B·M, P+G, d]
        frames = self.local_block(frames, local_allowed).reshape(B, M, P + G, d)
        patches, gists = frames[:, :, :P], frames[:, :, P:]

        # Each anchor is one more frame (no gists); same weights. Empty anchors are hidden from the history keys
        anchor_allowed = torch.ones(B * A, P, P, dtype=torch.bool, device=anchor.device)
        anchor = self.local_block(anchor.reshape(B * A, P, d), anchor_allowed).reshape(B, A, P, d)

        # (2) History attention: gists of every frame (queries) attend to anchor, recent patches, and earlier gists
        queries = gists.reshape(B, M * G, d)                                                # [B, M·G, d]
        keys = torch.cat([anchor.reshape(B, A * P, d), patches.reshape(B, M * P, d), queries], dim=1)  # [B, A·P + M·P + M·G, d]
        queries = queries + self.history_attn(self.history_q_norm(queries), self.history_kv_norm(keys), history_allowed)
        gists = queries.reshape(B, M, G, d)

        # (3) Per-token MLP (shared across anchor, patches, gists)
        anchor = self.mlp_block(anchor)
        patches = self.mlp_block(patches)
        gists = self.mlp_block(gists)

        return anchor, patches, gists


# === Incremental (Eval) Cache ===
KV = Tuple[torch.Tensor, torch.Tensor]  # keys, values [B, H, L, dh]


@dataclass
class GistCache:
    """Per-layer history-attention keys/values for one batch of episodes (built by GistEncoder.init_cache)."""

    anchor: List[KV]                                                          # per layer: the anchor's patches
    recent_patches: List[List[KV]] = field(default_factory=list)              # per layer: newest n_recent frames only
    gists: List[Optional[KV]] = field(default_factory=list)                   # per layer: every frame so far
    n_frames: int = 0


# === Gist Encoder ===
class GistEncoder(nn.Module):
    def __init__(
        self,
        vision_dim: int = 2176,
        d: int = 1024,
        n_gist: int = 8,
        depth: int = 4,
        heads: int = 16,
        n_recent: int = 3,
        max_timestep: int = 1024,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.d, self.n_gist, self.n_recent, self.max_timestep = d, n_gist, n_recent, max_timestep
        self.gradient_checkpointing = False

        self.in_proj = nn.Linear(vision_dim, d)
        self.gist_queries = nn.Parameter(torch.randn(n_gist, d) * 0.02)
        self.time_emb = nn.Embedding(max_timestep, d)
        nn.init.normal_(self.time_emb.weight, std=0.02)
        self.layers = nn.ModuleList([GistEncoderLayer(d, heads, mlp_ratio) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d)

    def build_masks(
        self, frame_valid: torch.Tensor, P: int, anchor_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build both attention masks from metadata (frame index, token type, validity). True = may attend."""
        B, M = frame_valid.shape
        A = anchor_valid.shape[1]
        G, device = self.n_gist, frame_valid.device

        # Frame-local: keys must be real; every token may always see itself (never a fully-masked row => no NaN)
        n_local = P + G
        local_valid = frame_valid.reshape(B * M, 1).expand(B * M, n_local)                 # [B·M, P+G]
        eye = torch.eye(n_local, dtype=torch.bool, device=device)
        local_allowed = local_valid[:, None, :] | eye[None]                                # [B·M, P+G, P+G]

        # History: query = gist of frame i; keys = [anchor | patches of frame j | gists of frame j]
        frame_of_query = torch.arange(M, device=device).repeat_interleave(G)               # [M·G]
        frame_of_patch = torch.arange(M, device=device).repeat_interleave(P)               # [M·P]
        frame_of_gist = frame_of_query                                                     # [M·G]
        i = frame_of_query[:, None]
        sees_anchor = torch.ones(M * G, A * P, dtype=torch.bool, device=device)
        sees_patch = (frame_of_patch[None, :] <= i) & (frame_of_patch[None, :] >= i - self.n_recent)
        sees_gist = frame_of_gist[None, :] <= i
        rule = torch.cat([sees_anchor, sees_patch, sees_gist], dim=1)                      # [M·G, A·P + M·P + M·G]
        key_valid = torch.cat(
            [
                anchor_valid.repeat_interleave(P, dim=1),                                   # anchor 0 is always real
                frame_valid.repeat_interleave(P, dim=1),
                frame_valid.repeat_interleave(G, dim=1),
            ],
            dim=1,
        )                                                                                  # [B, A·P + M·P + M·G]
        history_allowed = rule[None] & key_valid[:, None, :]  # anchor 0 always visible => no fully-masked row
        return local_allowed, history_allowed

    def forward(
        self,
        patches: torch.Tensor,         # [B, M, P, vision_dim]
        frame_valid: torch.Tensor,     # [B, M] bool
        timesteps: torch.Tensor,       # [B, M] long, episode step of each memory frame (< max_timestep)
        anchor_patches: torch.Tensor,  # [B, A, P, vision_dim] (or [B, P, vision_dim] for a single anchor, frame 0)
        anchor_valid: Optional[torch.Tensor] = None,      # [B, A] bool; anchor 0 must be real (default: all real)
        anchor_timesteps: Optional[torch.Tensor] = None,  # [B, A] long (default: 0 for a single anchor)
    ) -> torch.Tensor:                 # -> gists [B, M, G, d]; empty slots are exactly zero
        B, M, P, _ = patches.shape
        anchor_patches, anchor_valid, anchor_timesteps = self._anchor_inputs(anchor_patches, anchor_valid, anchor_timesteps)
        frame_valid = frame_valid.bool()
        timesteps = timesteps.masked_fill(~frame_valid, 0)                     # empty slots: any value is fine

        frame_time = self.time_emb(timesteps)[:, :, None, :]                  # [B, M, 1, d]
        anchor_time = self.time_emb(anchor_timesteps)[:, :, None, :]          # [B, A, 1, d]

        x = self.in_proj(patches) + frame_time                                # [B, M, P, d]
        anchor = self.in_proj(anchor_patches) + anchor_time                   # [B, A, P, d]
        gists = self.gist_queries.expand(B, M, -1, -1) + frame_time           # [B, M, G, d]

        local_allowed, history_allowed = self.build_masks(frame_valid, P, anchor_valid)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                anchor, x, gists = checkpoint(
                    layer, anchor, x, gists, local_allowed, history_allowed, use_reentrant=False
                )
            else:
                anchor, x, gists = layer(anchor, x, gists, local_allowed, history_allowed)

        gists = self.out_norm(gists)
        return gists * frame_valid[:, :, None, None].to(gists.dtype)          # zero out empty slots

    @staticmethod
    def _anchor_inputs(anchor_patches, anchor_valid, anchor_timesteps):
        if anchor_patches.dim() == 3:                                         # single anchor (frame 0)
            anchor_patches = anchor_patches[:, None]
        B, A = anchor_patches.shape[:2]
        device = anchor_patches.device
        if anchor_valid is None:
            anchor_valid = torch.ones(B, A, dtype=torch.bool, device=device)
        if anchor_timesteps is None:
            assert A == 1, "pass anchor_timesteps with more than one anchor frame"
            anchor_timesteps = torch.zeros(B, 1, dtype=torch.long, device=device)
        anchor_valid = anchor_valid.bool()
        assert anchor_valid[:, 0].all(), "anchor 0 (the first frame) must always be real"
        return anchor_patches, anchor_valid, anchor_timesteps.masked_fill(~anchor_valid, 0)

    # === Incremental path (closed-loop eval) ===
    def _embed_frame(self, patches: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """patches [B, P, vision_dim], timesteps [B] -> (patch states [B, P, d], gist states [B, G, d]) at layer 0."""
        frame_time = self.time_emb(timesteps)[:, None, :]                     # [B, 1, d]
        x = self.in_proj(patches) + frame_time
        gists = self.gist_queries.expand(patches.shape[0], -1, -1) + frame_time
        return x, gists

    @torch.no_grad()
    def init_cache(self, anchor_patches: torch.Tensor, anchor_timesteps: Optional[torch.Tensor] = None) -> GistCache:
        """
        Start the memory of an episode: run the anchors (all real; [B, A, P, vision_dim] with timesteps [B, A], or
        [B, P, vision_dim] for frame 0 alone) through every layer once and cache their keys/values.
        """
        anchor_patches, _, anchor_timesteps = self._anchor_inputs(anchor_patches, None, anchor_timesteps)
        B, A, P, _ = anchor_patches.shape
        anchor = self.in_proj(anchor_patches) + self.time_emb(anchor_timesteps)[:, :, None, :]   # [B, A, P, d]
        anchor = anchor.reshape(B * A, P, -1)
        cache = GistCache(anchor=[], recent_patches=[[] for _ in self.layers], gists=[None for _ in self.layers])
        for layer in self.layers:
            anchor = layer.local_block(anchor, None)
            k, v = layer.history_kv(anchor)                                   # [B·A, H, P, dh]
            cache.anchor.append(tuple(x.reshape(B, A, *x.shape[1:]).transpose(1, 2).flatten(2, 3) for x in (k, v)))
            anchor = layer.mlp_block(anchor)
        return cache

    @torch.no_grad()
    def add_frame(self, cache: GistCache, patches: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Append the next memory frame (patches [B, P, vision_dim], timesteps [B]) and return its gists [B, G, d].
        Only this frame is computed; it attends to the cached anchor, the newest n_recent frames' patches, and every
        earlier frame's gists. Gives the same gists as `forward` over the whole bank (while the bank is under the
        training cap; past the cap, training drops the oldest frames, while the cache keeps every gist like MemoryWAM).
        """
        P = patches.shape[1]
        x, gists = self._embed_frame(patches, timesteps)
        last = len(self.layers) - 1
        for l, layer in enumerate(self.layers):
            # (1) Frame-local attention over [own patches ; own gists] (a real frame sees all of its own tokens)
            frame = layer.local_block(torch.cat([x, gists], dim=1), None)
            x, gists = frame[:, :P], frame[:, P:]

            # (2) History attention against the cache, plus this frame's own patches and gists
            own_patches, own_gists = layer.history_kv(x), layer.history_kv(gists)
            recent = cache.recent_patches[l] + [own_patches]
            past_gists = [] if cache.gists[l] is None else [cache.gists[l]]
            all_gists = past_gists + [own_gists]
            keys = [cache.anchor[l]] + recent + all_gists
            k, v = torch.cat([kv[0] for kv in keys], dim=2), torch.cat([kv[1] for kv in keys], dim=2)
            gists = gists + layer.history_attn.attend(layer.history_q_norm(gists), k, v, None)

            # Update the cache: gists are kept forever, patches only while within the next frame's recent window
            cache.gists[l] = tuple(torch.cat([kv[i] for kv in all_gists], dim=2) for i in range(2))
            cache.recent_patches[l] = recent[max(0, len(recent) - self.n_recent) :] if self.n_recent > 0 else []

            # (3) MLP (the patches are not needed after the last layer)
            gists = layer.mlp_block(gists)
            if l < last:
                x = layer.mlp_block(x)

        cache.n_frames += 1
        return self.out_norm(gists)
