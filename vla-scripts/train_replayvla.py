"""
train_replayvla.py

LoRA fine-tuning of ReplayVLA (OpenVLA + MemoryWAM-style memory; see memory_bank/) on an RLDS dataset, forked from
finetune.py. With `--use_memory False` it trains the plain OpenVLA baseline with the exact same loop, data, LoRA,
logging and preemption-safe checkpointing (only the memory differs). Differences from finetune.py:
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

import json
import os
import sys
import time
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
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from memory_bank.modeling import register_replayvla
from memory_bank.training import (
    StopRequest,
    action_metrics,
    batch_to_model_inputs,
    find_resume_checkpoint,
    load_openvla,
    load_replayvla,
    load_resume_checkpoint,
    merge_lora,
    save_resume_checkpoint,
    sync_any,
    upcast_memory_modules,
    wrap_with_lora,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def job_memory_gb() -> float:
    """RAM used by this Slurm job's cgroup (all ranks + data pipeline), in GB; falls back to this process's RSS."""
    try:
        cgroup = open("/proc/self/cgroup").read().strip().split("::")[-1]
        path = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        for p in [path, *path.parents]:          # walk up to the job-level cgroup (step -> job)
            if p.name.startswith("job_") and (p / "memory.current").exists():
                return int((p / "memory.current").read_text()) / 2**30
        return int((path / "memory.current").read_text()) / 2**30
    except Exception:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


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
    use_memory: bool = True                                         # False = plain OpenVLA baseline (same recipe)
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

    # Preemption / time-limit safety (SLURM)
    resume: bool = True                                             # Resume from <run_dir>/resume if it exists
    checkpoint_interval_minutes: float = 30.0                       # Periodic resumable checkpoint (safety net)
    stop_file: Optional[Path] = None                                # Touched by the batch script on SIGTERM/SIGUSR1;
                                                                    #   default: <run_dir>/STOP

    # LoRA Arguments
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Tracking Parameters
    wandb_project: str = "replayvla"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"                                      # "offline" if compute nodes have no internet
    log_every: int = 50                                             # Plain-text progress line every N steps (rank 0)
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
    model_name = "ReplayVLA" if cfg.use_memory else "OpenVLA baseline (no memory)"
    print(f"Training {model_name} from `{cfg.vla_path}` on `{cfg.dataset_name}`")

    assert torch.cuda.is_available(), "Training assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"{'replayvla' if cfg.use_memory else 'openvla-baseline'}+{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps * distributed_state.num_processes}"
        f"+lr-{cfg.learning_rate}+lora-r{cfg.lora_rank}"
    )
    if cfg.use_memory:
        exp_id += f"+mem{cfg.max_memory_frames}-s{cfg.memory_stride}-g{cfg.n_gist}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    if distributed_state.is_main_process:  # model settings, needed to rebuild the base when merging/evaluating
        run_config = {"vla_path": cfg.vla_path, "use_memory": cfg.use_memory, **(cfg.memory_kwargs() if cfg.use_memory else {})}
        (run_dir / "replayvla_config.json").write_text(json.dumps(run_config, indent=2))

    # Register OpenVLA + ReplayVLA with HF Auto classes (processor loading; no remote code needed)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    register_replayvla()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    if cfg.use_memory:
        vla = load_replayvla(cfg.vla_path, memory_kwargs=cfg.memory_kwargs(), torch_dtype=torch.bfloat16)
        upcast_memory_modules(vla)
    else:
        vla = load_openvla(cfg.vla_path, torch_dtype=torch.bfloat16)
    if cfg.gradient_checkpointing:
        vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.use_memory:
        # HF's gradient_checkpointing_enable also flips any submodule with a `gradient_checkpointing` attribute (incl.
        # the gist encoder), so set the gist encoder's flag explicitly afterwards, and before PEFT copies the module
        vla.gist_encoder.gradient_checkpointing = cfg.gist_gradient_checkpointing
    vla = vla.to(device_id)

    # Resume (LoRA + memory modules + optimizer + step) if a complete checkpoint exists, else start fresh
    resume_ckpt = find_resume_checkpoint(run_dir) if cfg.resume else None
    trainer_state = {"completed_steps": 0, "wandb_run_id": None}
    optimizer_state = None
    if resume_ckpt is not None:
        vla, optimizer_state, trainer_state = load_resume_checkpoint(vla, resume_ckpt)
        if distributed_state.is_main_process:
            print(f"Resuming from {resume_ckpt} at step {trainer_state['completed_steps']}", flush=True)
    else:
        vla = wrap_with_lora(vla, rank=cfg.lora_rank, dropout=cfg.lora_dropout)
    if distributed_state.is_main_process:
        vla.print_trainable_parameters()
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    completed_steps = trainer_state["completed_steps"]
    if completed_steps >= cfg.max_steps:
        print(f"Already finished ({completed_steps} >= {cfg.max_steps} steps)")
        return
    if distributed_state.is_main_process and (run_dir / "DONE").exists():
        (run_dir / "DONE").unlink()  # max_steps was raised since the last run

    stop = StopRequest(cfg.stop_file or run_dir / "STOP")
    if distributed_state.is_main_process and stop.stop_file.exists():
        stop.stop_file.unlink()  # stale request from the previous job
    dist.barrier()
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    data_kwargs = dict(
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_memory:
        batch_transform = ReplayRLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            n_short=cfg.n_short,
            max_memory=cfg.max_memory_frames,
        )
        vla_dataset = ReplayRLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform, **data_kwargs,
            memory_stride=cfg.memory_stride, n_short=cfg.n_short, max_memory=cfg.max_memory_frames,
        )
        collator_cls = ReplayCollator
    else:
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        vla_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, **data_kwargs)
        collator_cls = PaddedCollatorForActionPrediction
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = collator_cls(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    if distributed_state.is_main_process:
        run = wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}", mode=cfg.wandb_mode,
            id=trainer_state.get("wandb_run_id"), resume="allow",
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        )
        trainer_state["wandb_run_id"] = run.id

    def checkpoint(reason: str) -> None:
        if distributed_state.is_main_process:
            start = time.time()
            state = dict(trainer_state, completed_steps=completed_steps, reason=reason, time=time.time())
            path = save_resume_checkpoint(vla.module, optimizer, state, run_dir)
            print(f"[step {completed_steps}] resumable checkpoint ({reason}) -> {path} in {time.time() - start:.1f}s")
        dist.barrier()

    recent = {k: deque(maxlen=cfg.grad_accumulation_steps) for k in ("loss", "action_accuracy", "l1_loss")}
    window = {"loss": 0.0, "action_accuracy": 0.0, "l1_loss": 0.0, "memory_frames": 0.0, "n": 0}
    last_checkpoint_time = last_log_time = time.time()
    last_log_step = completed_steps
    # OpenVLA's output has no `num_visual_tokens`: its visual prefix is the image's patch count (256)
    baseline_visual_tokens = vla.module.vision_backbone.featurizer.patch_embed.num_patches

    # Note: the RLDS stream is not resumed position-exactly (it is an infinite shuffled stream); a resumed job just
    # continues with fresh shuffled data, which is equivalent in expectation.
    show_bar = distributed_state.is_main_process and sys.stderr.isatty()
    with tqdm.tqdm(initial=completed_steps, total=cfg.max_steps, leave=False, disable=not show_bar) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch_to_model_inputs(batch, device_id, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(**inputs)
                loss = output.loss

            (loss / cfg.grad_accumulation_steps).backward()

            num_visual_tokens = getattr(output, "num_visual_tokens", None) or baseline_visual_tokens
            metrics = action_metrics(output.logits, inputs["labels"], num_visual_tokens, action_tokenizer)
            memory_frames = inputs["memory_valid"].sum(1).float().mean().item() if "memory_valid" in inputs else 0.0
            recent["loss"].append(loss.item())
            recent["action_accuracy"].append(metrics["action_accuracy"])
            recent["l1_loss"].append(metrics["l1_loss"])
            for key, value in (("loss", loss.item()), ("action_accuracy", metrics["action_accuracy"]),
                               ("l1_loss", metrics["l1_loss"]),
                               ("memory_frames", memory_frames)):
                window[key] += value
            window["n"] += 1

            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue

            optimizer.step()
            optimizer.zero_grad()
            completed_steps += 1
            progress.update()

            if distributed_state.is_main_process and completed_steps % cfg.log_every == 0 and window["n"] > 0:
                now = time.time()
                sec_per_step = (now - last_log_time) / max(1, completed_steps - last_log_step)
                eta_h = sec_per_step * (cfg.max_steps - completed_steps) / 3600
                n = window["n"]
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] step {completed_steps}/{cfg.max_steps} | "
                    f"loss {window['loss'] / n:.4f} | action acc {window['action_accuracy'] / n:.3f} | "
                    f"L1 {window['l1_loss'] / n:.4f} | mem frames {window['memory_frames'] / n:.1f} | "
                    f"{sec_per_step:.2f} s/step | ETA {int(eta_h)}h{int((eta_h % 1) * 60):02d}m | "
                    f"RAM {job_memory_gb():.0f} GB",
                    flush=True,
                )
                window = {k: 0.0 for k in window} | {"n": 0}
                last_log_time, last_log_step = now, completed_steps

            if distributed_state.is_main_process and completed_steps % 10 == 0:
                log = {("train_loss" if k == "loss" else k): sum(v) / len(v) for k, v in recent.items()}
                log["memory_frames_per_example"] = memory_frames
                wandb.log(log, step=completed_steps)

            # All ranks agree on: stop requested (SIGTERM / SIGUSR1 / stop file)?  periodic checkpoint due?
            due = time.time() - last_checkpoint_time > cfg.checkpoint_interval_minutes * 60
            stop_now, due = sync_any([stop.local_request(), due], device=device_id)
            finished = completed_steps >= cfg.max_steps

            if stop_now or due or finished:
                checkpoint("stop requested" if stop_now else "finished" if finished else "periodic")
                last_checkpoint_time = time.time()
            if completed_steps % cfg.save_steps == 0 or finished:
                save_merged(cfg, vla, processor, vla_dataset, run_dir, adapter_dir, completed_steps, distributed_state)
            if stop_now:
                print(f"Stopping cleanly at step {completed_steps} ({stop.reason}); resume will pick up from here")
                return
            if finished:
                if distributed_state.is_main_process:
                    (run_dir / "DONE").write_text(json.dumps({"completed_steps": completed_steps}))
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                return


def save_merged(cfg, vla, processor, vla_dataset, run_dir, adapter_dir, step, distributed_state) -> None:
    if distributed_state.is_main_process:
        print(f"Saving checkpoint for step {step}")
        processor.save_pretrained(run_dir)
        vla.module.save_pretrained(adapter_dir)      # LoRA adapter + full memory modules
    dist.barrier()

    if cfg.merge_on_save and distributed_state.is_main_process:
        merged = merge_lora(
            cfg.vla_path, adapter_dir, memory_kwargs=cfg.memory_kwargs() if cfg.use_memory else None,
            use_memory=cfg.use_memory, torch_dtype=torch.bfloat16,
        )
        out_dir = run_dir if cfg.save_latest_checkpoint_only else Path(f"{run_dir}--{step}_chkpt")
        os.makedirs(out_dir, exist_ok=True)
        if not cfg.save_latest_checkpoint_only:
            save_dataset_statistics(vla_dataset.dataset_statistics, out_dir)
            processor.save_pretrained(out_dir)
        merged.save_pretrained(out_dir)
        print(f"Saved merged {'ReplayVLA' if cfg.use_memory else 'OpenVLA baseline'} for step {step} at: {out_dir}")
        del merged
    dist.barrier()


if __name__ == "__main__":
    train()
