"""
configuration.py

ReplayVLAConfig: OpenVLAConfig plus the memory settings. Everything OpenVLA-related (vision backbone, LLM, action
bins, norm stats) is inherited unchanged, so a ReplayVLA can be initialized from any OpenVLA checkpoint.
"""

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig


class ReplayVLAConfig(OpenVLAConfig):
    model_type: str = "replayvla"

    def __init__(
        self,
        memory_stride: int = 8,             # grid spacing (policy steps) for short-term and memory frames
        n_short: int = 4,                   # short-term window size, current frame included
        max_memory_frames: int = 64,        # memory slots M (LIBERO-10's longest demo needs ~60 at stride 8)
        n_gist: int = 8,                    # gist tokens per memory frame G
        gist_dim: int = 1024,
        gist_depth: int = 4,
        gist_heads: int = 16,
        gist_n_recent: int = 3,             # frames before i whose full patches frame i's gists may see
        gist_max_timestep: int = 1024,      # size of the learned timestep embedding table
        **kwargs,
    ) -> None:
        self.memory_stride, self.n_short, self.max_memory_frames = memory_stride, n_short, max_memory_frames
        self.n_gist, self.gist_dim, self.gist_depth, self.gist_heads = n_gist, gist_dim, gist_depth, gist_heads
        self.gist_n_recent, self.gist_max_timestep = gist_n_recent, gist_max_timestep

        super().__init__(**kwargs)
