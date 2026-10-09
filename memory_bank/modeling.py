"""
modeling.py

ReplayVLAForActionPrediction: OpenVLA plus MemoryWAM-style hybrid memory. Subclasses OpenVLAForActionPrediction
without modifying it; with no memory inputs it behaves exactly like OpenVLA.

LLM input sequence (all visual tokens are spliced in after <BOS>, as OpenVLA does with its single image):

    [BOS] [anchors ×A·P] [gists ×M·G] [short ×(n_short-1)·P] [current ×P] [prompt …] [action tokens …]

  - anchor frames (0, s, … ; A = n_anchor) / short / current frames: vision backbone → OpenVLA's existing projector
  - memory frames: vision backbone (no grad) → GistEncoder → gist projector
  - learned role embeddings (zero-init) are added to each anchor, each short-term age slot, and gists, so at init the
    current-frame pathway is unchanged; empty anchor / short-term / memory slots get attention-mask 0
  - every visual position gets label IGNORE_INDEX (trained only through the action loss)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from prismatic.extern.hf.modeling_prismatic import (
    IGNORE_INDEX,
    OpenVLAForActionPrediction,
    PrismaticCausalLMOutputWithPast,
)

from .configuration import ReplayVLAConfig
from .gist_encoder import GistEncoder

# Extra forward() kwargs carrying memory; passed through generate() on the first (uncached) step only
MEMORY_KWARGS = (
    "anchor_pixel_values",
    "anchor_valid",
    "short_pixel_values",
    "short_valid",
    "memory_pixel_values",
    "memory_valid",
    "memory_timesteps",
    "memory_features",
    "memory_gists",
)


@dataclass
class ReplayVLACausalLMOutputWithPast(PrismaticCausalLMOutputWithPast):
    # Number of visual tokens spliced in after <BOS>. Logits at positions [num_visual_tokens : -1] predict text
    # tokens 1 … T-1 (this replaces the hard-coded `num_patches` slice in finetune.py's metrics)
    num_visual_tokens: Optional[int] = None


class GistProjector(nn.Module):
    def __init__(self, gist_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(gist_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, gists: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(gists)))


class RoleEmbeddings(nn.Module):
    """Learned role vectors [anchor 0 … anchor A-1, short slot 0 … short slot n-1, gist], added to visual tokens; zero-init.
    A module (not a bare Parameter) so PEFT's `modules_to_save` can train and save it alongside LoRA."""

    def __init__(self, n_roles: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_roles, dim))


class ReplayVLAForActionPrediction(OpenVLAForActionPrediction):
    config_class = ReplayVLAConfig

    def __init__(self, config: ReplayVLAConfig) -> None:
        super().__init__(config)
        llm_dim = config.text_config.hidden_size
        self.n_short_past = config.n_short - 1
        self.n_anchor = getattr(config, "n_anchor", 1)

        self.gist_encoder = GistEncoder(
            vision_dim=self.vision_backbone.embed_dim,
            d=config.gist_dim,
            n_gist=config.n_gist,
            depth=config.gist_depth,
            heads=config.gist_heads,
            n_recent=config.gist_n_recent,
            max_timestep=config.gist_max_timestep,
        )
        self.gist_projector = GistProjector(config.gist_dim, llm_dim)

        # Role embeddings: [anchor 0 … A-1, short slot 0 … n_short-2 (oldest → newest age), gist]; zero-init
        self.role_emb = RoleEmbeddings(self.n_anchor + self.n_short_past + 1, llm_dim)

    # === Initialization of new parameters when loading an OpenVLA checkpoint (they show up as "missing keys") ===
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)  # nn.Linear / nn.Embedding: normal(0, initializer_range), zero bias
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GistEncoder):
            nn.init.normal_(module.gist_queries, std=0.02)
        elif isinstance(module, RoleEmbeddings):
            nn.init.zeros_(module.weight)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "ReplayVLAForActionPrediction":
        model = super().from_pretrained(*args, **kwargs)
        model._materialize_meta_parameters()
        return model

    def _materialize_meta_parameters(self) -> None:
        """
        With `low_cpu_mem_usage=True`, HF builds the model on the meta device and only materializes missing keys it
        recognizes; in transformers 4.40 + accelerate, tied-parameter detection on meta tensors can wrongly skip some of
        the new memory parameters (observed: gist_queries, some Linear weights), leaving them on meta. Materialize and
        initialize any such leftovers. No-op when loading a full ReplayVLA checkpoint.
        """
        device = self.get_input_embeddings().weight.device
        for module in self.modules():
            if any(p.is_meta for p in module.parameters(recurse=False)):
                module.to_empty(device=device, recurse=False)
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                self._init_weights(module)

    # === Vision helpers ===
    @property
    def num_patches(self) -> int:
        return self.vision_backbone.featurizer.patch_embed.num_patches

    def _featurize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] -> patch features [N, P, vision_dim]."""
        return self.vision_backbone(pixel_values)

    @torch.no_grad()
    def encode_memory_frames(self, memory_pixel_values: torch.Tensor, memory_valid: torch.Tensor) -> torch.Tensor:
        """[B, M, C, H, W] -> [B, M, P, vision_dim]; only real slots go through the backbone (empty slots stay 0)."""
        B, M = memory_valid.shape
        flat_valid = memory_valid.reshape(B * M).bool()
        param = next(self.vision_backbone.parameters())
        feats = torch.zeros(
            B * M, self.num_patches, self.vision_backbone.embed_dim, dtype=param.dtype, device=memory_pixel_values.device
        )
        if flat_valid.any():
            pixels = memory_pixel_values.reshape(B * M, *memory_pixel_values.shape[2:])[flat_valid]
            feats[flat_valid] = self._featurize(pixels.to(param.dtype)).to(feats.dtype)
        return feats.reshape(B, M, self.num_patches, -1)

    def build_visual_tokens(
        self,
        pixel_values: torch.Tensor,           # [B, C, H, W]        current frame
        anchor_pixel_values: torch.Tensor,    # [B, A, C, H, W]     (or [B, C, H, W] when A = 1)
        short_pixel_values: torch.Tensor,     # [B, S, C, H, W]     S = n_short - 1, age-aligned (last = newest)
        short_valid: torch.Tensor,            # [B, S] bool
        memory_valid: torch.Tensor,           # [B, M] bool         oldest first, empty slots at the end
        memory_timesteps: torch.Tensor,       # [B, M] long
        memory_pixel_values: Optional[torch.Tensor] = None,  # [B, M, C, H, W]
        memory_features: Optional[torch.Tensor] = None,      # [B, M, P, vision_dim] (precomputed ViT features)
        memory_gists: Optional[torch.Tensor] = None,         # [B, M, G, gist_dim] (gist encoder output, cached at eval)
        anchor_valid: Optional[torch.Tensor] = None,         # [B, A] bool (default: all real)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (visual tokens [B, V, D], visual attention mask [B, V], current-frame tokens [B, P, D])."""
        B, S = short_valid.shape
        assert S == self.n_short_past, f"expected {self.n_short_past} short-term slots, got {S}"
        n_memory_inputs = sum(x is not None for x in (memory_pixel_values, memory_features, memory_gists))
        assert n_memory_inputs == 1, "pass exactly one of memory_pixel_values / memory_features / memory_gists"

        if anchor_pixel_values.dim() == 4:
            anchor_pixel_values = anchor_pixel_values[:, None]
        A = anchor_pixel_values.shape[1]
        assert A == self.n_anchor, f"expected {self.n_anchor} anchor frames, got {A}"
        if anchor_valid is None:
            anchor_valid = torch.ones(B, A, dtype=torch.bool, device=anchor_pixel_values.device)
        anchor_valid = anchor_valid.bool()
        anchor_timesteps = torch.arange(A, device=anchor_valid.device)[None] * self.config.memory_stride  # [1, A]
        anchor_timesteps = anchor_timesteps.expand(B, A)

        current = self._featurize(pixel_values)                                               # [B, P, Dv]
        anchor = self._featurize(anchor_pixel_values.flatten(0, 1))
        anchor = anchor.reshape(B, A, *anchor.shape[1:])                                      # [B, A, P, Dv]
        short = self._featurize(short_pixel_values.reshape(B * S, *short_pixel_values.shape[2:]))
        short = short.reshape(B, S, *short.shape[1:])                                         # [B, S, P, Dv]
        if memory_gists is not None:   # eval: gists computed incrementally (GistEncoder.add_frame), one per new frame
            gists = memory_gists.to(anchor.dtype) * memory_valid[:, :, None, None].to(anchor.dtype)
        else:
            if memory_features is None:
                memory_features = self.encode_memory_frames(memory_pixel_values, memory_valid)  # [B, M, P, Dv]
            gists = self.gist_encoder(
                memory_features.to(anchor.dtype), memory_valid, memory_timesteps, anchor, anchor_valid, anchor_timesteps
            )                                                                                 # [B, M, G, dg]

        role = self.role_emb.weight
        current_tokens = self.projector(current)                                              # [B, P, D]
        anchor_tokens = self.projector(anchor) + role[:A][None, :, None, :]                   # [B, A, P, D]
        short_tokens = self.projector(short) + role[A : A + S][None, :, None, :]              # [B, S, P, D]
        gist_tokens = self.gist_projector(gists) + role[-1]                                   # [B, M, G, D]

        P, G = current_tokens.shape[1], gists.shape[2]
        visual = torch.cat(
            [anchor_tokens.flatten(1, 2), gist_tokens.flatten(1, 2), short_tokens.flatten(1, 2), current_tokens], dim=1
        )                                                                                     # [B, V, D]
        ones = torch.ones(B, P, dtype=torch.bool, device=visual.device)
        visual_mask = torch.cat(
            [
                anchor_valid.repeat_interleave(P, dim=1),
                memory_valid.bool().repeat_interleave(G, dim=1),
                short_valid.bool().repeat_interleave(P, dim=1),
                ones,
            ],
            dim=1,
        )                                                                                     # [B, V]
        return visual, visual_mask, current_tokens

    # === Forward ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        anchor_pixel_values: Optional[torch.FloatTensor] = None,
        anchor_valid: Optional[torch.Tensor] = None,
        short_pixel_values: Optional[torch.FloatTensor] = None,
        short_valid: Optional[torch.Tensor] = None,
        memory_pixel_values: Optional[torch.FloatTensor] = None,
        memory_valid: Optional[torch.Tensor] = None,
        memory_timesteps: Optional[torch.LongTensor] = None,
        memory_features: Optional[torch.FloatTensor] = None,
        memory_gists: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, ReplayVLACausalLMOutputWithPast]:
        # No memory inputs (vanilla OpenVLA usage), or a cached generation step: defer to OpenVLA unchanged
        if anchor_pixel_values is None or past_key_values is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_projector_features=output_projector_features,
                return_dict=return_dict,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache and not self.training
        assert input_ids is not None and pixel_values is not None, "memory forward needs input_ids and pixel_values"

        visual, visual_mask, current_tokens = self.build_visual_tokens(
            pixel_values, anchor_pixel_values, short_pixel_values, short_valid, memory_valid, memory_timesteps,
            memory_pixel_values=memory_pixel_values, memory_features=memory_features, memory_gists=memory_gists,
            anchor_valid=anchor_valid,
        )
        B, V = visual_mask.shape

        # Three-way splice after <BOS>: embeddings, attention mask, labels (same cut point, same length)
        input_embeddings = self.get_input_embeddings()(input_ids)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1], visual.to(input_embeddings.dtype), input_embeddings[:, 1:]], dim=1
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape[1] < input_ids.shape[1]:
            # OpenVLA's predict_action appends token 29871 to input_ids but not to attention_mask. Vanilla OpenVLA gets
            # away with it (an all-ones mask is skipped by SDPA); ReplayVLA's mask has zeros, so pad it to match.
            extra = input_ids.shape[1] - attention_mask.shape[1]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], extra)], dim=1)
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], visual_mask.to(attention_mask.dtype), attention_mask[:, 1:]], dim=1
        )
        multimodal_labels = None
        if labels is not None:
            visual_labels = torch.full((B, V), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            multimodal_labels = torch.cat([labels[:, :1], visual_labels, labels[:, 1:]], dim=1)

        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return language_model_output

        return ReplayVLACausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=current_tokens,
            num_visual_tokens=V,
        )

    # === Generation: pass memory inputs through on the first (uncached) step ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, inputs_embeds, pixel_values, attention_mask, **kwargs
        )
        if past_key_values is None:
            model_inputs.update({k: kwargs[k] for k in MEMORY_KWARGS if kwargs.get(k) is not None})
        return model_inputs


def register_replayvla() -> None:
    """Register ReplayVLA with HF Auto classes (needed for AutoProcessor / AutoModelForVision2Seq loading)."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("replayvla", ReplayVLAConfig)
    AutoImageProcessor.register(ReplayVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(ReplayVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(ReplayVLAConfig, ReplayVLAForActionPrediction)
