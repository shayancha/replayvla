"""
training.py

Helpers for training ReplayVLA, kept here (not in the script) so they can be unit-tested on a tiny model:

    load_replayvla           load an OpenVLA checkpoint as ReplayVLA (new memory modules initialized), or a ReplayVLA
                             checkpoint as-is
    wrap_with_lora           LoRA on every pretrained linear layer (like finetune.py's "all-linear"), memory modules
                             trained in full via `modules_to_save`
    merge_lora               rebuild the base, apply a saved adapter, merge -> plain ReplayVLA for eval
    action_metrics           finetune.py's action accuracy / L1, using `num_visual_tokens` instead of a hard-coded 256
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PretrainedConfig

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

from .configuration import ReplayVLAConfig
from .modeling import ReplayVLAForActionPrediction

MEMORY_MODULES = ("gist_encoder", "gist_projector", "role_emb")
MODEL_INPUT_KEYS = (
    "input_ids", "attention_mask", "pixel_values", "labels",
    "anchor_pixel_values", "short_pixel_values", "short_valid",
    "memory_pixel_values", "memory_valid", "memory_timesteps",
)
PIXEL_KEYS = ("pixel_values", "anchor_pixel_values", "short_pixel_values", "memory_pixel_values")


# === Loading ===
def replayvla_config_from_openvla(base: OpenVLAConfig, **memory_kwargs) -> ReplayVLAConfig:
    config_dict = base.to_dict()
    for key in ("model_type", "architectures", "auto_map", "transformers_version", "_name_or_path"):
        config_dict.pop(key, None)
    return ReplayVLAConfig(**config_dict, **memory_kwargs)


def load_replayvla(
    path: Union[str, Path],
    memory_kwargs: Optional[Dict] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    low_cpu_mem_usage: bool = True,
    **kwargs,
) -> ReplayVLAForActionPrediction:
    """`path` may hold an OpenVLA checkpoint (new memory modules get initialized from `memory_kwargs`) or a ReplayVLA
    checkpoint (loaded as-is; `memory_kwargs` must then be empty or match)."""
    config_dict, _ = PretrainedConfig.get_config_dict(str(path))
    if config_dict.get("model_type") == "replayvla":
        config = ReplayVLAConfig.from_pretrained(str(path))
        for key, value in (memory_kwargs or {}).items():
            assert getattr(config, key) == value, f"checkpoint has {key}={getattr(config, key)}, requested {value}"
    else:
        config = replayvla_config_from_openvla(OpenVLAConfig.from_pretrained(str(path)), **(memory_kwargs or {}))
    return ReplayVLAForActionPrediction.from_pretrained(
        str(path), config=config, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, **kwargs
    )


def upcast_memory_modules(model: ReplayVLAForActionPrediction) -> None:
    """Keep the fully-trained (randomly initialized) memory modules in fp32; autocast still runs them in bf16."""
    for name in MEMORY_MODULES:
        getattr(model, name).float()


# === LoRA ===
def lora_target_modules(model: nn.Module) -> List[str]:
    """Every nn.Linear except the memory modules (trained in full) and the LM head, mirroring PEFT's 'all-linear'."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith(MEMORY_MODULES) and "lm_head" not in name
    ]


def wrap_with_lora(model: ReplayVLAForActionPrediction, rank: int = 32, dropout: float = 0.0) -> PeftModel:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=min(rank, 16),
        lora_dropout=dropout,
        target_modules=lora_target_modules(model),
        init_lora_weights="gaussian",
        modules_to_save=list(MEMORY_MODULES),
    )
    return get_peft_model(model, lora_config)


def merge_lora(
    base_path: Union[str, Path], adapter_dir: Union[str, Path], memory_kwargs: Optional[Dict] = None, **load_kwargs
) -> ReplayVLAForActionPrediction:
    """Base (OpenVLA or ReplayVLA checkpoint) + saved adapter -> merged ReplayVLA with trained memory modules."""
    base = load_replayvla(base_path, memory_kwargs=memory_kwargs, **load_kwargs)
    return PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()


# === Metrics (same as finetune.py, but the visual-token offset comes from the model output) ===
def action_metrics(logits: torch.Tensor, labels: torch.Tensor, num_visual_tokens: int, action_tokenizer) -> Dict:
    action_preds = logits[:, num_visual_tokens:-1].argmax(dim=2)
    action_gt = labels[:, 1:].to(action_preds.device)
    mask = action_gt > action_tokenizer.action_token_begin_idx
    accuracy = ((action_preds == action_gt) & mask).sum().float() / mask.sum().float()
    pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
    gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
    return {"action_accuracy": accuracy.item(), "l1_loss": torch.nn.functional.l1_loss(pred, gt).item()}


def batch_to_model_inputs(batch: Dict, device: Union[int, torch.device], dtype: torch.dtype = torch.bfloat16) -> Dict:
    inputs = {}
    for key in MODEL_INPUT_KEYS:
        value = batch[key].to(device)
        inputs[key] = value.to(dtype) if key in PIXEL_KEYS else value
    return inputs


# === Resumable checkpoints (preemption / time-limit safe) ===
# Layout: <run_dir>/resume/{adapter/, optimizer.pt, trainer_state.json}. Written to resume.tmp and swapped in with
# renames, so a job killed mid-save always leaves the previous complete checkpoint behind.
RESUME_DIR, RESUME_TMP, RESUME_OLD = "resume", "resume.tmp", "resume.old"


def save_resume_checkpoint(peft_model, optimizer, trainer_state: Dict, run_dir: Union[str, Path]) -> Path:
    import json
    import shutil

    run_dir = Path(run_dir)
    tmp, final, old = run_dir / RESUME_TMP, run_dir / RESUME_DIR, run_dir / RESUME_OLD
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    peft_model.save_pretrained(tmp / "adapter")                  # LoRA weights + memory modules (modules_to_save)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "trainer_state.json").write_text(json.dumps(trainer_state))
    (tmp / "COMPLETE").touch()                                   # only complete checkpoints are ever resumed from
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    return final


def find_resume_checkpoint(run_dir: Union[str, Path]) -> Optional[Path]:
    """Newest complete checkpoint, also covering a kill between the two renames (then only resume.old exists)."""
    for name in (RESUME_DIR, RESUME_OLD):
        path = Path(run_dir) / name
        if (path / "COMPLETE").exists():
            return path
    return None


def load_resume_checkpoint(base_model: ReplayVLAForActionPrediction, checkpoint: Union[str, Path]):
    """Returns (trainable PeftModel with restored LoRA + memory modules, optimizer state dict, trainer state)."""
    import json

    checkpoint = Path(checkpoint)
    peft_model = PeftModel.from_pretrained(base_model, str(checkpoint / "adapter"), is_trainable=True)
    optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    return peft_model, optimizer_state, trainer_state


class StopRequest:
    """
    Turns SLURM's "about to stop" signals into a clean checkpoint-and-exit, agreed on by all ranks.

      - SIGTERM: preemption / scancel (Slurm sends it to every process; ~30 s before SIGKILL by default)
      - SIGUSR1: time-limit warning, e.g. `#SBATCH --signal=B:USR1@900` forwarded by the batch script
      - a stop file (the batch script touches it on any signal; works even if the signal misses a worker)
    Call `local_request()` once per step and combine it across ranks with `sync_any`, so every rank stops at the
    same step.
    """

    def __init__(self, stop_file: Optional[Union[str, Path]] = None) -> None:
        import signal

        self.requested, self.reason = False, None
        self.stop_file = Path(stop_file) if stop_file else None
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._handler)

    def _handler(self, signum, frame) -> None:
        self.requested, self.reason = True, f"signal {signum}"

    def local_request(self) -> bool:
        if not self.requested and self.stop_file is not None and self.stop_file.exists():
            self.requested, self.reason = True, f"stop file {self.stop_file}"
        return self.requested


def sync_any(flags: List[bool], device=None) -> List[bool]:
    """Element-wise OR of boolean flags across all ranks (so every rank takes the same branch). No-op without DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = torch.tensor([1.0 if f else 0.0 for f in flags], device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return [bool(v > 0) for v in tensor.tolist()]
    return list(flags)
