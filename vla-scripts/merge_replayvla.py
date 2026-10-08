"""
merge_replayvla.py

Turn a training run's latest resumable checkpoint (LoRA adapter + memory modules) into a standalone ReplayVLA checkpoint
(or, for a `--use_memory False` baseline run, a standalone OpenVLA checkpoint: evaluate it with --model_family openvla)
for evaluation: base model + adapter -> merged weights, plus the run's dataset statistics and processor.

    python vla-scripts/merge_replayvla.py --run_dir /work/$USER/runs/<run> [--out_dir <dir>]
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from transformers import AutoProcessor

from memory_bank.modeling import register_replayvla
from memory_bank.training import find_resume_checkpoint, merge_lora


@dataclass
class MergeConfig:
    run_dir: Path
    out_dir: Optional[Path] = None          # default: <run_dir>/merged-step<N>


@draccus.wrap()
def main(cfg: MergeConfig) -> None:
    register_replayvla()
    ckpt = find_resume_checkpoint(cfg.run_dir)
    assert ckpt is not None, f"no complete resumable checkpoint in {cfg.run_dir}"
    step = json.loads((ckpt / "trainer_state.json").read_text())["completed_steps"]
    config_path = cfg.run_dir / "replayvla_config.json"
    run_cfg = json.loads(config_path.read_text()) if config_path.exists() else {"vla_path": "openvla/openvla-7b"}
    vla_path = run_cfg.pop("vla_path")
    use_memory = run_cfg.pop("use_memory", True)
    out_dir = cfg.out_dir or cfg.run_dir / f"merged-step{step}"

    print(f"Merging {ckpt} (step {step}) onto {vla_path} with memory settings {run_cfg or 'defaults'} -> {out_dir}")
    merged = merge_lora(vla_path, ckpt / "adapter", memory_kwargs=run_cfg if use_memory else None,
                        use_memory=use_memory, torch_dtype=torch.bfloat16)
    merged.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(vla_path, trust_remote_code=True).save_pretrained(out_dir)
    shutil.copy(cfg.run_dir / "dataset_statistics.json", out_dir / "dataset_statistics.json")
    print(f"Saved merged ReplayVLA (step {step}) to {out_dir}")


if __name__ == "__main__":
    main()
