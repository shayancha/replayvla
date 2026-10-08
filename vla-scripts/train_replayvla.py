"""
train_replayvla.py

LoRA fine-tuning of ReplayVLA (OpenVLA + MemoryWAM-style memory; see memory_bank/) on an RLDS dataset, forked from
finetune.py. Differences from finetune.py:
    - the model is ReplayVLAForActionPrediction, initialized from an OpenVLA checkpoint (memory modules start fresh)
    - LoRA on every pretrained linear layer (as finetune.py's "all-linear"); memory modules trained in full
      (PEFT modules_to_save) and kept in fp32
    - the data pipeline gathers anchor / memory / short-term frames per step (memory_bank.data)
    - metrics slice logits with the model's `num_visual_tokens` (finetune.py hard-codes 256)
    - optional gradient checkpointing (LLM + gist encoder); everything configurable from the CLI (SLURM-friendly)

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train_replayvla.py \
        --data_root_dir <PATH/TO/RLDS/DATASETS> --dataset_name libero_10_no_noops --run_root_dir <PATH/TO/RUNS> ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

import wandb
from memory_bank.data import ReplayCollator, ReplayRLDSBatchTransform, ReplayRLDSDataset
from memory_bank.modeling import register_replayvla
from memory_bank.training import (
    action_metrics,
    batch_to_model_inputs,
    load_replayvla,
    merge_lora,
    upcast_memory_modules,
    wrap_with_lora,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ReplayTrainConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # OpenVLA (or ReplayVLA) checkpoint to start from

    # Directory Paths
    data_root_dir: Path = Path("datasets/modified_libero_rlds")     # Directory containing the RLDS dataset(s)
    dataset_name: str = "libero_10_no_noops"                        # LIBERO-Long
    run_root_dir: Path = Path("runs")                               # Logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # LoRA weights before merging

    # Memory (memory_bank/configuration.py)
    memory_stride: int = 8
    n_short: int = 4
    max_memory_frames: int = 64
    n_gist: int = 8
    gist_dim: int = 1024
    gist_depth: int = 4
    gist_heads: int = 16
    gist_n_recent: int = 3
    gist_max_timestep: int = 1024

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Per-GPU batch size
    max_steps: int = 50_000                                         # Max number of gradient steps
    save_steps: int = 5_000                                         # Checkpoint interval (gradient steps)
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 20_000                               # Each element holds ~69 encoded frames
    save_latest_checkpoint_only: bool = True
    merge_on_save: bool = True                                      # Merge LoRA into a full ReplayVLA at each save
    gradient_checkpointing: bool = True                             # LLM activation checkpointing
    gist_gradient_checkpointing: bool = False                       # Gist encoder activation checkpointing

    # LoRA Arguments
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Tracking Parameters
    wandb_project: str = "replayvla"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"                                      # "offline" if compute nodes have no internet
    run_id_note: Optional[str] = None
    # fmt: on

    def memory_kwargs(self) -> dict:
        return dict(
            memory_stride=self.memory_stride, n_short=self.n_short, max_memory_frames=self.max_memory_frames,
            n_gist=self.n_gist, gist_dim=self.gist_dim, gist_depth=self.gist_depth, gist_heads=self.gist_heads,
            gist_n_recent=self.gist_n_recent, gist_max_timestep=self.gist_max_timestep,
        )


@draccus.wrap()
def train(cfg: ReplayTrainConfig) -> None:
    print(f"Training ReplayVLA from `{cfg.vla_path}` on `{cfg.dataset_name}`")

    assert torch.cuda.is_available(), "Training assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"replayvla+{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps * distributed_state.num_processes}"
        f"+lr-{cfg.learning_rate}+lora-r{cfg.lora_rank}"
        f"+mem{cfg.max_memory_frames}-s{cfg.memory_stride}-g{cfg.n_gist}"
    )
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Register OpenVLA + ReplayVLA with HF Auto classes (processor loading; no remote code needed)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    register_replayvla()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = load_replayvla(cfg.vla_path, memory_kwargs=cfg.memory_kwargs(), torch_dtype=torch.bfloat16)
    upcast_memory_modules(vla)
    if cfg.gradient_checkpointing:
        vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # HF's gradient_checkpointing_enable also flips any submodule with a `gradient_checkpointing` attribute (incl. the
    # gist encoder), so set the gist encoder's flag explicitly afterwards, and before PEFT copies the module
    vla.gist_encoder.gradient_checkpointing = cfg.gist_gradient_checkpointing
    vla = vla.to(device_id)

    vla = wrap_with_lora(vla, rank=cfg.lora_rank, dropout=cfg.lora_dropout)
    vla.print_trainable_parameters()
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = ReplayRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        n_short=cfg.n_short,
        max_memory=cfg.max_memory_frames,
    )
    vla_dataset = ReplayRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        memory_stride=cfg.memory_stride,
        n_short=cfg.n_short,
        max_memory=cfg.max_memory_frames,
    )
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = ReplayCollator(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}", mode=cfg.wandb_mode,
                   config={k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()})

    recent = {k: deque(maxlen=cfg.grad_accumulation_steps) for k in ("loss", "action_accuracy", "l1_loss")}

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch_to_model_inputs(batch, device_id, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(**inputs)
                loss = output.loss

            (loss / cfg.grad_accumulation_steps).backward()

            metrics = action_metrics(output.logits, inputs["labels"], output.num_visual_tokens, action_tokenizer)
            recent["loss"].append(loss.item())
            recent["action_accuracy"].append(metrics["action_accuracy"])
            recent["l1_loss"].append(metrics["l1_loss"])

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                log = {("train_loss" if k == "loss" else k): sum(v) / len(v) for k, v in recent.items()}
                log["memory_frames_per_example"] = inputs["memory_valid"].sum(1).float().mean().item()
                wandb.log(log, step=gradient_step_idx)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0 and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                save_checkpoint(cfg, vla, processor, vla_dataset, run_dir, adapter_dir, gradient_step_idx, distributed_state)

            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


def save_checkpoint(cfg, vla, processor, vla_dataset, run_dir, adapter_dir, step, distributed_state) -> None:
    if distributed_state.is_main_process:
        print(f"Saving checkpoint for step {step}")
        processor.save_pretrained(run_dir)
        vla.module.save_pretrained(adapter_dir)      # LoRA adapter + full memory modules
    dist.barrier()

    if cfg.merge_on_save and distributed_state.is_main_process:
        merged = merge_lora(cfg.vla_path, adapter_dir, memory_kwargs=cfg.memory_kwargs(), torch_dtype=torch.bfloat16)
        out_dir = run_dir if cfg.save_latest_checkpoint_only else Path(f"{run_dir}--{step}_chkpt")
        os.makedirs(out_dir, exist_ok=True)
        if not cfg.save_latest_checkpoint_only:
            save_dataset_statistics(vla_dataset.dataset_statistics, out_dir)
            processor.save_pretrained(out_dir)
        merged.save_pretrained(out_dir)
        print(f"Saved merged ReplayVLA for step {step} at: {out_dir}")
        del merged
    dist.barrier()


if __name__ == "__main__":
    train()
