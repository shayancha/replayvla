"""
gist_encoder.py

ReplayVLA's gist encoder: compresses each memory frame (256 ViT patch features) into a few learned "gist" tokens,
in the context of everything that came before it, following MemoryWAM's hybrid memory (arXiv 2606.20562).

When memory frame i was the newest frame, its gists may draw on the "entire historical context":
    - the anchor frame's patches (always)
    - frame i's own patches, and the patches of the `n_recent` frames before it (its recent window)
    - the gists of every frame <= i (the memory bank)
They never see later frames or empty slots, so memory only flows forward in time. This makes the whole bank trainable
in one parallel pass (no BPTT) and identical at training and inference.

Each layer is factorized (cheap, see planning/lessons/0007):
    (1) frame-local attention over [patches_i ; gists_i] (one sequence per frame; the anchor is its own sequence)
    (2) history attention: every frame's gists (queries) attend to anchor + recent patches + earlier gists (keys)
    (3) a per-token MLP
all pre-norm with residuals. Spec & tests: planning/milestones/m1a-gist-encoder.md, tests/test_gist_encoder.py.

Conventions: memory slots are ordered oldest-first with empty slots at the end; empty slots output exactly zero.
"""

from typing import Tuple

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

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """x_q: [N, Lq, d], x_kv: [N, Lk, d], allowed: bool [N, Lq, Lk] (True = may attend) -> [N, Lq, d]."""
        N, Lq, d = x_q.shape
        Lk = x_kv.shape[1]
        q = self.q_proj(x_q).view(N, Lq, self.heads, self.head_dim).transpose(1, 2)    # [N, H, Lq, dh]
        k = self.k_proj(x_kv).view(N, Lk, self.heads, self.head_dim).transpose(1, 2)   # [N, H, Lk, dh]
        v = self.v_proj(x_kv).view(N, Lk, self.heads, self.head_dim).transpose(1, 2)   # [N, H, Lk, dh]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])    # mask broadcast over heads
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

    def forward(
        self,
        anchor: torch.Tensor,          # [B, P, d]
        patches: torch.Tensor,         # [B, M, P, d]
        gists: torch.Tensor,           # [B, M, G, d]
        local_allowed: torch.Tensor,   # [B·M, P+G, P+G] bool
        history_allowed: torch.Tensor, # [B, M·G, P + M·P + M·G] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, P, d = patches.shape
        G = gists.shape[2]

        # (1) Frame-local attention: fold (example, frame) into the batch; each frame = [its patches ; its gists]
        frames = torch.cat([patches, gists], dim=2).reshape(B * M, P + G, d)                # [B·M, P+G, d]
        normed = self.local_norm(frames)
        frames = frames + self.local_attn(normed, normed, local_allowed)
        frames = frames.reshape(B, M, P + G, d)
        patches, gists = frames[:, :, :P], frames[:, :, P:]

        # The anchor is one more frame (always real, no gists); same weights
        normed = self.local_norm(anchor)
        anchor_allowed = torch.ones(B, P, P, dtype=torch.bool, device=anchor.device)
        anchor = anchor + self.local_attn(normed, normed, anchor_allowed)

        # (2) History attention: gists of every frame (queries) attend to anchor, recent patches, and earlier gists
        queries = gists.reshape(B, M * G, d)                                                # [B, M·G, d]
        keys = torch.cat([anchor, patches.reshape(B, M * P, d), queries], dim=1)            # [B, P + M·P + M·G, d]
        queries = queries + self.history_attn(self.history_q_norm(queries), self.history_kv_norm(keys), history_allowed)
        gists = queries.reshape(B, M, G, d)

        # (3) Per-token MLP (shared across anchor, patches, gists)
        anchor = anchor + self.mlp(self.mlp_norm(anchor))
        patches = patches + self.mlp(self.mlp_norm(patches))
        gists = gists + self.mlp(self.mlp_norm(gists))

        return anchor, patches, gists


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

    def build_masks(self, frame_valid: torch.Tensor, P: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build both attention masks from metadata (frame index, token type, validity). True = may attend."""
        B, M = frame_valid.shape
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
        sees_anchor = torch.ones(M * G, P, dtype=torch.bool, device=device)
        sees_patch = (frame_of_patch[None, :] <= i) & (frame_of_patch[None, :] >= i - self.n_recent)
        sees_gist = frame_of_gist[None, :] <= i
        rule = torch.cat([sees_anchor, sees_patch, sees_gist], dim=1)                      # [M·G, P + M·P + M·G]
        key_valid = torch.cat(
            [
                torch.ones(B, P, dtype=torch.bool, device=device),                          # anchor is always real
                frame_valid.repeat_interleave(P, dim=1),
                frame_valid.repeat_interleave(G, dim=1),
            ],
            dim=1,
        )                                                                                  # [B, P + M·P + M·G]
        history_allowed = rule[None] & key_valid[:, None, :]  # anchor always visible => no fully-masked row
        return local_allowed, history_allowed

    def forward(
        self,
        patches: torch.Tensor,         # [B, M, P, vision_dim]
        frame_valid: torch.Tensor,     # [B, M] bool
        timesteps: torch.Tensor,       # [B, M] long, episode step of each memory frame (< max_timestep)
        anchor_patches: torch.Tensor,  # [B, P, vision_dim]
    ) -> torch.Tensor:                 # -> gists [B, M, G, d]; empty slots are exactly zero
        B, M, P, _ = patches.shape
        frame_valid = frame_valid.bool()
        timesteps = timesteps.masked_fill(~frame_valid, 0)                     # empty slots: any value is fine

        frame_time = self.time_emb(timesteps)[:, :, None, :]                  # [B, M, 1, d]
        anchor_time = self.time_emb(torch.zeros(B, dtype=torch.long, device=patches.device))[:, None, :]

        x = self.in_proj(patches) + frame_time                                # [B, M, P, d]
        anchor = self.in_proj(anchor_patches) + anchor_time                   # [B, P, d]
        gists = self.gist_queries.expand(B, M, -1, -1) + frame_time           # [B, M, G, d]

        local_allowed, history_allowed = self.build_masks(frame_valid, P)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                anchor, x, gists = checkpoint(
                    layer, anchor, x, gists, local_allowed, history_allowed, use_reentrant=False
                )
            else:
                anchor, x, gists = layer(anchor, x, gists, local_allowed, history_allowed)

        gists = self.out_norm(gists)
        return gists * frame_valid[:, :, None, None].to(gists.dtype)          # zero out empty slots
