"""
Tests for memory_bank/inference.py: the eval-time memory buffer must give the model exactly the inputs the training
pipeline gives it for the same step. Run from the repo root:

    python tests/test_inference.py
"""

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_bank.data import ReplayCollator  # noqa: E402
from memory_bank.inference import ReplayMemoryBuffer  # noqa: E402
from test_replay_data import make_transform, rlds_example  # noqa: E402
from test_replayvla_model import M, make_model  # noqa: E402

T = 80  # episode length (stride 8 → up to 9 grid frames; tiny model keeps M=4 memory slots)


def episode_pixels(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, 6, 224, 224, generator=g)


def train_visual_tokens(model, pixels, t):
    """Visual tokens for step t built through the training path: TF-equivalent window -> batch transform -> collator."""
    lookup = lambda img: pixels[int(np.asarray(img)[0, 0, 0])]  # noqa: E731  (image pixel value = frame index)
    example = rlds_example(t, M)
    example["observation"]["image_primary"] = np.stack(
        [np.full((4, 4, 3), f, dtype=np.uint8) for f in example["observation"]["replay_frame_index"]]
    )
    batch = ReplayCollator(model_max_length=2048, pad_token_id=0)([make_transform(M, image_transform=lookup)(example)])
    with torch.no_grad():
        return model.build_visual_tokens(
            batch["pixel_values"], batch["anchor_pixel_values"], batch["short_pixel_values"], batch["short_valid"],
            batch["memory_valid"], batch["memory_timesteps"], memory_pixel_values=batch["memory_pixel_values"],
        )[:2]


def test_buffer_matches_training_pipeline():
    model = make_model()
    pixels = episode_pixels()
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    checked = 0
    for t in range(T):
        assert buffer.add_frame(pixels[t]) == t
        if t in (0, 5, 9, 17, 33, 50, 64, 79):   # episode start, first grid frames, memory filling, cap reached
            mem = buffer.model_inputs()
            with torch.no_grad():
                visual, mask, _ = model.build_visual_tokens(
                    pixels[t][None], mem["anchor_pixel_values"], mem["short_pixel_values"], mem["short_valid"],
                    mem["memory_valid"], mem["memory_timesteps"], memory_features=mem["memory_features"],
                )
            ref_visual, ref_mask = train_visual_tokens(model, pixels, t)
            assert torch.equal(mask, ref_mask), f"t={t}: visual mask differs from training"
            # Empty slots may hold different filler (zeros here, a copy of frame 0 in training); they are masked out and
            # invisible to the model (tests/test_replayvla_model.py), so compare the real tokens only
            assert torch.allclose(visual[mask], ref_visual[ref_mask], atol=1e-5), f"t={t}: visual tokens differ from training"
            checked += 1
    assert checked == 8


def test_memory_frames_encoded_once_and_pixels_released():
    model = make_model()
    pixels = episode_pixels(1)
    calls = []
    original = model._featurize
    model._featurize = lambda x: (calls.append(x.shape[0]), original(x))[1]
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    for t in range(T):
        buffer.add_frame(pixels[t])
        buffer.model_inputs()
    # Grid frames 8..72 (< 79) minus the 3 short ones = 6 memory frames, newest 4 kept... each encoded exactly once
    assert len(buffer.features) == len(calls), f"{len(calls)} encodes for {len(buffer.features)} cached memory frames"
    assert set(buffer.pixels) <= {0, 56, 64, 72}, f"pixels kept for {sorted(buffer.pixels)}: memory frames should be dropped"


def test_reset_starts_a_new_episode():
    model = make_model()
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    for t in range(40):
        buffer.add_frame(episode_pixels(2)[t])
    buffer.reset()
    assert buffer.t == -1 and not buffer.pixels and not buffer.features
    assert buffer.add_frame(episode_pixels(3)[0]) == 0
    assert not buffer.model_inputs()["memory_valid"].any() and not buffer.model_inputs()["short_valid"].any()


def test_noop_threshold_drops_steps():
    model = make_model()
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32, noop_threshold=1e-3)
    move, still = np.array([0.1, 0, 0, 0, 0, 0, 1.0]), np.array([0, 0, 0, 0, 0, 0, 1.0])
    buffer.add_frame(torch.zeros(6, 224, 224)); buffer.record_action(move)       # t=0 kept (anchor)
    buffer.add_frame(torch.zeros(6, 224, 224)); buffer.record_action(still)      # t=1 near no-op -> dropped
    assert buffer.t == 0
    assert buffer.add_frame(torch.zeros(6, 224, 224)) == 1                       # next observation reuses step 1
    default = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    default.add_frame(torch.zeros(6, 224, 224)); default.add_frame(torch.zeros(6, 224, 224)); default.record_action(still)
    assert default.t == 1, "without noop_threshold every step counts"


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
