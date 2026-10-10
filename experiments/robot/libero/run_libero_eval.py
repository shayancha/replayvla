"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>

Resumable + splittable (ReplayVLA addition):
    Every finished episode is appended to <results_dir>/task<ID>.jsonl. A rerun with the same settings skips finished
    episodes, so a preempted + requeued job loses at most the episode in progress. `--task_ids 0-4` / `--task_ids 5-9`
    split a suite across jobs that share one results_dir; <results_dir>/SUMMARY.txt always holds the combined rates.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family ("openvla" or "replayvla" = OpenVLA + memory)
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    noop_threshold: Optional[float] = None           # (ReplayVLA) drop near-no-op steps from memory, as in *_no_noops

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    task_ids: Optional[str] = None                   # Subset of tasks, e.g. "0-4" or "0,3,7" (default: all)
    results_dir: Optional[str] = None                # Per-episode results, for resuming (default: under local_log_dir)

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def parse_task_ids(spec: Optional[str], n_tasks: int) -> List[int]:
    if spec is None:
        return list(range(n_tasks))
    ids = []
    for part in str(spec).replace(" ", "").split(","):
        lo, _, hi = part.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    assert all(0 <= i < n_tasks for i in ids), f"task_ids {spec} out of range for {n_tasks} tasks"
    return sorted(set(ids))


def load_results(results_dir: str, task_id: int, num_trials: int) -> Dict[int, bool]:
    """episode index -> success, for finished episodes (a line cut off by a preemption is ignored)."""
    path = os.path.join(results_dir, f"task{task_id:02d}.jsonl")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record["episode"] < num_trials:
                    results[record["episode"]] = bool(record["success"])
    return results


def record_result(results_dir: str, task_id: int, record: Dict) -> None:
    with open(os.path.join(results_dir, f"task{task_id:02d}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_summary(results_dir: str, n_tasks: int, num_trials: int, descriptions: Dict[int, str]) -> str:
    """Combined success rates over every task in results_dir (also tasks run by other jobs)."""
    lines, total, successes = [], 0, 0
    for task_id in range(n_tasks):
        results = load_results(results_dir, task_id, num_trials)
        if not results:
            continue
        n, k = len(results), sum(results.values())
        total, successes = total + n, successes + k
        status = "" if n == num_trials else f"   ({n}/{num_trials} episodes done)"
        lines.append(f"task {task_id}: {k}/{n} = {k / n * 100:.1f}%  {descriptions.get(task_id, '')}{status}")
    lines.append(f"TOTAL: {successes}/{total} = {successes / max(total, 1) * 100:.1f}%"
                 + ("" if total == n_tasks * num_trials else f"   ({total}/{n_tasks * num_trials} episodes done)"))
    text = "\n".join(lines) + "\n"
    tmp = os.path.join(results_dir, f".SUMMARY.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, os.path.join(results_dir, "SUMMARY.txt"))
    return text


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ("openvla", "replayvla"):
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family in ("openvla", "replayvla"):
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Per-episode results (stable path, no timestamp, so a restarted job finds them)
    task_ids = parse_task_ids(cfg.task_ids, num_tasks_in_suite)
    if cfg.results_dir is None:
        cfg.results_dir = os.path.join(
            cfg.local_log_dir, f"RESULTS-{cfg.task_suite_name}-{cfg.model_family}--{cfg.run_id_note or 'default'}"
        )
    os.makedirs(cfg.results_dir, exist_ok=True)
    descriptions = {i: task_suite.get_task(i).language for i in range(num_tasks_in_suite)}
    print(f"Tasks {task_ids}; per-episode results in {cfg.results_dir}")
    log_file.write(f"Tasks {task_ids}; per-episode results in {cfg.results_dir}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        finished = load_results(cfg.results_dir, task_id, cfg.num_trials_per_task)
        if len(finished) == cfg.num_trials_per_task:
            print(f"Task {task_id}: all {len(finished)} episodes already done, skipping")
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = len(finished), sum(finished.values())
        if finished:
            print(f"Task {task_id}: resuming, {len(finished)} episodes already done")
            log_file.write(f"Task {task_id}: resuming, {len(finished)} episodes already done\n")
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if episode_idx in finished:
                continue
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            if cfg.model_family == "replayvla":
                model.replay_buffer.reset()  # new episode: empty memory, anchor = first frame the policy sees

            # Setup
            t = 0
            done = False
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family in ("openvla", "replayvla"):
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            record_result(cfg.results_dir, task_id, {"episode": episode_idx, "success": bool(done), "steps": t})
            write_summary(cfg.results_dir, num_tasks_in_suite, cfg.num_trials_per_task, descriptions)

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, task_id * cfg.num_trials_per_task + episode_idx + 1, success=done,
                task_description=task_description, log_file=log_file,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results (task counts include episodes finished before a restart; totals are this job's only)
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Combined results over every task in results_dir (including tasks run by other jobs)
    summary = write_summary(cfg.results_dir, num_tasks_in_suite, cfg.num_trials_per_task, descriptions)
    print(f"\n=== Results so far ({cfg.results_dir}/SUMMARY.txt) ===\n{summary}")
    log_file.write(f"\n=== Results so far ({cfg.results_dir}/SUMMARY.txt) ===\n{summary}")

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(max(total_episodes, 1)),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
