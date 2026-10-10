# Running ReplayVLA on the Duke Compute Cluster (DCC)

Guide covers setting up and running ReplayVLA, and a plain OpenVLA baseline, on DCC's H200 GPUs via Slurm. It goes from a fresh account to a training run and a LIBERO evaluation.

---

## 1. Storage layout (all under `/work/$USER`)

| Path | What |
|---|---|
| `/work/$USER/replayvla` | This repo |
| `/work/$USER/.conda/envs/replayvla` | Conda env |
| `/work/$USER/.cache/huggingface` | `HF_HOME`: OpenVLA-7B weights |
| `/work/$USER/data/modified_libero_rlds` | LIBERO RLDS data (`libero_10_no_noops` = LIBERO-Long) |
| `/work/$USER/runs/<run>` | Training runs: checkpoints, `resume/`, dataset stats, `DONE` |
| `/work/$USER/logs` | Slurm job output: `<job-name>-<jobid>.out` |
| `/work/$USER/LIBERO` | LIBERO simulator (eval only) |

`slurm_scripts/common.sh` sets all of these, and every script sources it. Override any of them by exporting the variable first, e.g. `export WORK_DIR=/work/$USER/other`.

---

## 2. Get interactive shell for setup

From a DCC login node:
```bash
srun -p interactive --cpus-per-task=8 --mem=32G --time=4:00:00 --pty bash -l
```

---

## 3. Clone code

```bash
cd /work/$USER
git clone https://github.com/shayancha/replayvla.git
cd replayvla
```


---

## 4. Create environment 

```bash
bash slurm_scripts/setup_env.sh
```

To use the env in your own shell later:
```bash
source slurm_scripts/common.sh
```

## 5. Download the model and data 

```bash
bash slurm_scripts/download_assets.sh
```

## 6. Run the tests (CPU)

```bash
source slurm_scripts/common.sh
export TMPDIR=/tmp      # temp dirs on /work (NFS) fail to clean up
for t in tests/test_*.py; do echo "$t: $(python $t 2>&1 | tail -1)"; done
```
All files should print `N/N passed`.

Now setup is done: leave the interactive node with `exit`. We submit training jobs from the login node.

---

## 7. Smoke test (1× H200, ≤1.5 h): run this first

```bash
cd /work/$USER/replayvla
sbatch slurm_scripts/smoke_test.sbatch          # optional arg: per-GPU batch size (default 8)
```
It trains 30 steps from scratch, resumes to 45, then resumes again and stops itself cleanly via the STOP file. Check `/work/$USER/logs/replayvla-smoke-<jobid>.out` for:
- `Resuming from …/resume at step 30`
- `Stopping cleanly at step …`
- a final listing of the `resume/` folder

Expect ~3 s/step on one H200 at batch 8.

---

## 8. Training

Defaults: LIBERO-Long (`libero_10_no_noops`), **2× H200, batch 8 per GPU (16 total)**, lr 5e-4, LoRA r=32 on all pretrained linear layers, **50,000 steps**, 48 CPUs.

**ReplayVLA:**
```bash
cd /work/$USER/replayvla
AUTO_RESUBMIT=1 sbatch slurm_scripts/train_replayvla.sbatch
```

**Vanilla OpenVLA baseline** (same script, data, LoRA, logging and checkpointing; only the memory is removed):
```bash
cd /work/$USER/replayvla
AUTO_RESUBMIT=1 RUN_NOTE=baseline sbatch slurm_scripts/train_replayvla.sbatch --use_memory False
```

- `AUTO_RESUBMIT=1`: when a job hits the 24 h limit, it checkpoints and resubmits itself.
- `RUN_NOTE` names the run and its STOP file. Use a different note for each concurrent run (default `main`).
- Any training option goes after the script name, e.g. `... train_replayvla.sbatch --max_steps 30000 --batch_size 6`. See `ReplayTrainConfig` in `vla-scripts/train_replayvla.py`, or `python vla-scripts/train_replayvla.py --help`.
- Speed: ReplayVLA ~2.8 s/step on 2× H200 (≈38 h for 50k steps, i.e. 2 chained jobs). The baseline is faster.

### How a run survives the 24 h limit and preemption
- A **resumable checkpoint** (LoRA + memory modules + optimizer + step) is written every 30 min (`--checkpoint_interval_minutes`). It is also written when the job gets SIGTERM (preemption: ~30 s warning), and 15 min before the time limit (SIGUSR1 → STOP file). 
- **Preemption:** Slurm requeues the job (`--requeue`), and it resumes from the checkpoint automatically.
- **Time limit:** the job checkpoints, exits cleanly, and (with `AUTO_RESUBMIT=1`) submits the next one, which resumes.
- The job does **not** save merged models itself (`--merge_on_save False`): loading a second 7B copy in the job caused host-RAM OOMs. Merge offline when you want to evaluate (9b).

### Monitoring and control
```bash
squeue -u $USER                                          # PD = queued, R = running
tail -f /work/$USER/logs/replayvla-train-<jobid>.out     # live log
```
Every 50 steps the log prints a line like:
```
[2026-10-08 12:00:00] step 4750/50000 | loss 0.41 | action acc 0.83 | L1 0.061 | mem frames 12.3 | 2.85 s/step | ETA 35h50m
```
- **Stop cleanly** (checkpoint + exit): `touch /work/$USER/runs/STOP-<RUN_NOTE>`. A new job with the same settings resumes from there.
- **Stop for good:** `scancel <jobid>`. With `AUTO_RESUBMIT`, also check `squeue` for a freshly resubmitted job.
- Training state: `cat /work/$USER/runs/<run>/resume/trainer_state.json`

---

## 9. LIBERO evaluation

### 9a. Install the simulator (once, in an interactive session: step 2)
**Do not run this while one of your training jobs is running or queued.** It installs packages into the same env, and a job that starts mid-install can fail (this happened).
```bash
source slurm_scripts/common.sh && bash slurm_scripts/setup_libero.sh
```
It clones LIBERO into `/work/$USER/LIBERO`, installs robosuite 1.4.1 and friends (pins `bddl==1.0.1`, `opencv-python==4.10.0.84`, `numpy==1.26.4`), and pre-writes LIBERO's config so it never prompts. `common.sh` puts LIBERO on `PYTHONPATH` and sets headless rendering (`MUJOCO_GL=egl`).

### 9b. Make an evaluatable checkpoint
Merge the run's latest resumable checkpoint into a standalone model. Run this in an interactive session with ≥32 GB RAM. It takes a few minutes and needs no GPU:
```bash
source slurm_scripts/common.sh
python vla-scripts/merge_replayvla.py --run_dir /work/$USER/runs/<run>
# -> /work/$USER/runs/<run>/merged-step<N>   (works for baseline runs too)
```
Training also keeps the trainable weights every 5,000 steps (`<run>/snapshots/step<N>`, ~0.6 GB each), so an earlier step can be evaluated too: add `--step <N>`, e.g. `--step 60000`.

### 9c. Run the eval (submit from the login node)
Split LIBERO-Long's 10 tasks across 2 H200 jobs (your 2-GPU allowance):
```bash
cd /work/$USER/replayvla
CKPT=/work/$USER/runs/<run>/merged-step<N>
TASK_IDS=0-4 sbatch slurm_scripts/eval_libero.sbatch "$CKPT" 50     # 50 trials per task (the standard number)
TASK_IDS=5-9 sbatch slurm_scripts/eval_libero.sbatch "$CKPT" 50
```
- Leave out `TASK_IDS` to run all 10 tasks in one job. Use e.g. `10` trials instead of `50` for a quick read (it gets its own results).
- Baseline: `MODEL_FAMILY=openvla TASK_IDS=0-4 sbatch slurm_scripts/eval_libero.sbatch <baseline merged dir> 50` (and `5-9`).
- **Preemption is handled:** each finished episode is saved immediately, and the requeued job skips finished episodes. Rerunning the same command also resumes.
- **Results:** `/work/$USER/logs/libero_eval/RESULTS-libero_10-<family>--<run>--merged-step<N>/SUMMARY.txt` has per-task and total success rates, combined across both jobs (it says how many episodes are done so far). Live progress: `tail -f /work/$USER/logs/replayvla-eval-<jobid>.out`. Rollout videos: `/work/$USER/replayvla/rollouts/`.
- To use a non-H200 GPU instead: `sbatch --account=<account> --partition=gpu-common --gres=gpu:5000_ada:1 slurm_scripts/eval_libero.sbatch ...`

---