"""
Tests for ReplayVLA's data pipeline (memory_bank/rlds_transforms.py, memory_bank/data.py) and the `chunk_fn` hook in
prismatic/vla/datasets/rlds/dataset.py. Synthetic trajectories only (no RLDS files needed). Run from the repo root:

    python tests/test_replay_data.py
"""

import os
import sys
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dlimp as dl  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

tf.config.set_visible_devices([], "GPU")

from memory_bank.data import ReplayCollator, ReplayRLDSBatchTransform  # noqa: E402
from memory_bank.frame_indices import replay_frame_indices  # noqa: E402
from memory_bank.rlds_transforms import replay_chunk_obs, replay_window_indices  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets.rlds.dataset import apply_trajectory_transforms  # noqa: E402

STRIDE, N_SHORT = 8, 4


def python_window(t, max_memory):
    idx = replay_frame_indices(t, STRIDE, N_SHORT, max_memory)
    return [idx.anchor] + idx.memory + idx.short + [idx.current], [True] + idx.memory_valid + idx.short_valid + [True]


def as_dlataset(traj):
    """A one-trajectory DLataset (dlimp only wraps datasets created via its own methods, so wrap like dlimp does)."""
    ds = tf.data.Dataset.from_tensors(traj)
    ds.__class__ = type("DLataset", (dl.DLataset, type(ds)), dl.DLataset.__dict__.copy())
    ds.is_flattened = False
    return ds


def synthetic_traj(T, A=7):
    return {
        "observation": {
            "image_primary": tf.strings.as_string(tf.range(T)),   # "encoded image" = its frame number
            "timestep": tf.range(T),
        },
        "task": {"language_instruction": tf.fill([T], "put the bowl on the plate")},
        "action": tf.cast(tf.reshape(tf.range(T * A), [T, A]), tf.float32),
        "dataset_name": tf.fill([T], "toy"),
    }


# === TF transform mirrors the Python source of truth ==================================================================

def test_tf_indices_match_python():
    for max_memory in [4, 64]:
        for T in [1, 2, 8, 9, 17, 40, 123, 505, 700]:
            idx, valid = replay_window_indices(tf.constant(T), STRIDE, N_SHORT, max_memory)
            idx, valid = idx.numpy(), valid.numpy()
            assert idx.shape == (T, 1 + max_memory + N_SHORT), idx.shape
            for t in range(T):
                exp_idx, exp_valid = python_window(t, max_memory)
                assert idx[t].tolist() == exp_idx, f"T={T}, M={max_memory}, t={t}: {idx[t].tolist()} != {exp_idx}"
                assert valid[t].tolist() == exp_valid, f"T={T}, M={max_memory}, t={t}: validity differs"


def test_replay_chunk_obs_gathers_frames_and_current_action():
    T, M = 60, 4
    out = replay_chunk_obs(synthetic_traj(T), stride=STRIDE, n_short=N_SHORT, max_memory=M)
    imgs = out["observation"]["image_primary"].numpy()
    assert imgs.shape == (T, 1 + M + N_SHORT)
    for t in [0, 13, 59]:
        exp_idx, _ = python_window(t, M)
        assert [int(x) for x in imgs[t]] == exp_idx, f"t={t}: gathered wrong frames"
        assert out["observation"]["timestep"].numpy()[t].tolist() == exp_idx
    act = out["action"].numpy()
    assert act.shape == (T, 1, 7) and np.array_equal(act[:, 0], synthetic_traj(T)["action"].numpy())


def test_chunk_fn_hook_through_apply_trajectory_transforms():
    from functools import partial

    T, M = 30, 4
    ds = as_dlataset(synthetic_traj(T))
    replay = apply_trajectory_transforms(
        ds, train=False, chunk_fn=partial(replay_chunk_obs, stride=STRIDE, n_short=N_SHORT, max_memory=M)
    ).flatten()
    frames = list(replay.as_numpy_iterator())
    assert len(frames) == T
    assert frames[25]["observation"]["image_primary"].shape == (1 + M + N_SHORT,)
    assert frames[25]["observation"]["pad_mask_dict"]["image_primary"].shape == (1 + M + N_SHORT,)

    # Default path (no chunk_fn) is unchanged: OpenVLA's window_size=1 chunking
    default = list(apply_trajectory_transforms(as_dlataset(synthetic_traj(T)), train=False).flatten().as_numpy_iterator())
    assert default[25]["observation"]["image_primary"].shape == (1,) and default[25]["action"].shape == (1, 7)


def test_real_jpegs_through_frame_transforms():
    """Decode + resize + augment run over the whole window (vmap), with the same augmentation for every slot."""
    from functools import partial

    from prismatic.vla.datasets.rlds.dataset import apply_frame_transforms

    T, M = 20, 4
    traj = synthetic_traj(T)
    base = tf.random.uniform([T, 64, 64, 3], maxval=255, dtype=tf.int32, seed=0)
    traj["observation"]["image_primary"] = tf.map_fn(
        lambda im: tf.io.encode_jpeg(tf.cast(im, tf.uint8)), base, fn_output_signature=tf.string
    )
    ds = apply_trajectory_transforms(
        as_dlataset(traj), train=True,
        chunk_fn=partial(replay_chunk_obs, stride=STRIDE, n_short=N_SHORT, max_memory=M),
    ).flatten()
    aug = dict(random_brightness=[0.2], augment_order=["random_brightness"])
    ds = apply_frame_transforms(ds, train=True, resize_size={"primary": (224, 224)}, image_augment_kwargs=aug)
    frame = next(iter(ds.skip(17).as_numpy_iterator()))
    imgs = frame["observation"]["image_primary"]
    assert imgs.shape == (1 + M + N_SHORT, 224, 224, 3) and imgs.dtype == np.uint8, imgs.shape


# === Batch transform + collator =======================================================================================

class StubTokenizer:
    """Character-level stand-in for the Llama tokenizer (ids < 32000, BOS = 1)."""
    vocab_size = 32000

    def __call__(self, text, add_special_tokens=True):
        ids = ([1] if add_special_tokens else []) + [3 + (ord(c) % 31000) for c in text]
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids):
        return "".join(chr(i % 1000 + 200) for i in ids)


def stub_image_transform(img, size=8):
    """Encodes the image's frame number (stored as its pixel value) into a [6, size, size] tensor."""
    return torch.full((6, size, size), float(np.asarray(img)[0, 0, 0]))


def rlds_example(t, M):
    """What the TF pipeline yields for step t after decoding: images whose pixel value = frame number."""
    idx, valid = python_window(t, M)
    return {
        "dataset_name": b"toy",
        "action": np.zeros((1, 7), dtype=np.float32),
        "task": {"language_instruction": b"put the bowl on the plate"},
        "observation": {
            "image_primary": np.stack([np.full((4, 4, 3), i % 256, dtype=np.uint8) for i in idx]),
            "pad_mask": np.array(valid),
            "replay_frame_index": np.array(idx),
        },
    }


def make_transform(M, image_transform=stub_image_transform):
    tok = StubTokenizer()
    return ReplayRLDSBatchTransform(ActionTokenizer(tok), tok, image_transform, PurePromptBuilder, n_short=N_SHORT, max_memory=M)


def test_batch_transform_splits_window():
    M, t = 4, 60   # grid < 60: 8..56 → short 40, 48, 56; memory 8, 16, 24, 32
    out = make_transform(M)(rlds_example(t, M))
    assert out["pixel_values"][0, 0, 0].item() == t % 256, "pixel_values must be the CURRENT frame"
    assert out["anchor_pixel_values"][0, 0, 0].item() == 0
    assert [int(x[0, 0, 0]) for x in out["short_pixel_values"]] == [40, 48, 56] and out["short_valid"].all()
    assert [int(x[0, 0, 0]) for x in out["memory_pixel_values"]] == [8, 16, 24, 32]
    assert out["memory_timesteps"].tolist() == [8, 16, 24, 32]
    assert (out["labels"] != -100).sum() > 0 and out["input_ids"][0] == 1


def test_batch_transform_early_episode():
    M, t = 4, 10   # grid < 10: 8 → short [_, _, 8]; no memory
    out = make_transform(M)(rlds_example(t, M))
    assert out["short_valid"].tolist() == [False, False, True]
    assert out["memory_pixel_values"] == [] and out["memory_timesteps"].numel() == 0


def test_collator_pads_memory_to_batch_max():
    M = 4
    transform = make_transform(M)
    instances = [transform(rlds_example(t, M)) for t in [10, 50, 60]]  # 0, 3, 4 memory frames (50: grid 8..48 → 3 old)
    batch = ReplayCollator(model_max_length=2048, pad_token_id=0)(instances)
    assert batch["memory_pixel_values"].shape[:2] == (3, 4), batch["memory_pixel_values"].shape
    assert batch["memory_valid"].tolist() == [[False] * 4, [True, True, True, False], [True] * 4]
    assert batch["memory_timesteps"][2].tolist() == [8, 16, 24, 32]
    assert batch["short_pixel_values"].shape[:2] == (3, N_SHORT - 1) and batch["anchor_pixel_values"].shape[0] == 3
    only_empty = ReplayCollator(model_max_length=2048, pad_token_id=0)([transform(rlds_example(3, M))])
    assert only_empty["memory_pixel_values"].shape[1] == 1 and not only_empty["memory_valid"].any()


def test_end_to_end_into_tiny_model():
    from test_replayvla_model import IGNORE_INDEX, make_model  # tiny ReplayVLA (max_memory_frames=4, n_gist=2)

    M = 4
    g = torch.Generator().manual_seed(0)
    transform = make_transform(M, image_transform=lambda img: torch.randn(6, 224, 224, generator=g))
    instances = [transform(rlds_example(t, M)) for t in [5, 50, 200]]
    batch = ReplayCollator(model_max_length=2048, pad_token_id=0)(instances)
    model = make_model().train()
    out = model(**{k: v for k, v in batch.items() if k != "dataset_names"})
    assert torch.isfinite(out.loss), "non-finite loss on a collated batch"
    n_text = batch["input_ids"].shape[1]
    assert out.logits.shape[1] == n_text + out.num_visual_tokens
    out.loss.backward()
    assert model.gist_encoder.gist_queries.grad is not None
    assert (batch["labels"] != IGNORE_INDEX).any()


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"✓ {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"✗ {test.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"✗ {test.__name__} crashed:\n{traceback.format_exc()}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
