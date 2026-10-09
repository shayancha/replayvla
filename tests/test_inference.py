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

T = 80  # episode length (stride 8 → up to 9 grid frames; tiny model keeps M=4 memory slots, reached at t=64)


def episode_pixels(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, 6, 224, 224, generator=g)


def train_visual_tokens(model, pixels, t, n_anchor):
    """Visual tokens for step t built through the training path: TF-equivalent window -> batch transform -> collator."""
    lookup = lambda img: pixels[int(np.asarray(img)[0, 0, 0])]  # noqa: E731  (image pixel value = frame index)
    example = rlds_example(t, M, n_anchor)
    example["observation"]["image_primary"] = np.stack(
        [np.full((4, 4, 3), f, dtype=np.uint8) for f in example["observation"]["replay_frame_index"]]
    )
    transform = make_transform(M, image_transform=lookup, n_anchor=n_anchor)
    batch = ReplayCollator(model_max_length=2048, pad_token_id=0)([transform(example)])
    with torch.no_grad():
        return model.build_visual_tokens(
            batch["pixel_values"], batch["anchor_pixel_values"], batch["short_pixel_values"], batch["short_valid"],
            batch["memory_valid"], batch["memory_timesteps"], memory_pixel_values=batch["memory_pixel_values"],
            anchor_valid=batch["anchor_valid"],
        )[:2]


def test_buffer_matches_training_pipeline():
    # The tiny model's training cap is M=4 memory frames: reached at t=64 (1 anchor) / t=72 (2 anchors). Within the cap,
    # the cached gists (computed once per frame) must equal training's full pass over the bank.
    for n_anchor, steps in [(1, (0, 5, 9, 17, 33, 50, 57, 64)), (2, (0, 8, 9, 17, 33, 50, 57, 72))]:
        model = make_model(n_anchor)
        pixels = episode_pixels()
        buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
        checked = 0
        for t in range(T):
            assert buffer.add_frame(pixels[t]) == t
            mem = buffer.model_inputs()                # every step, as in the eval loop (memory frames enter on time)
            if t in steps:                             # episode start, first grid frames, memory filling, cap full
                with torch.no_grad():
                    visual, mask, _ = model.build_visual_tokens(
                        pixels[t][None], mem["anchor_pixel_values"], mem["short_pixel_values"], mem["short_valid"],
                        mem["memory_valid"], mem["memory_timesteps"], memory_gists=mem["memory_gists"],
                        anchor_valid=mem["anchor_valid"],
                    )
                ref_visual, ref_mask = train_visual_tokens(model, pixels, t, n_anchor)
                where = f"n_anchor={n_anchor}, t={t}"
                assert torch.equal(mask, ref_mask), f"{where}: visual mask differs from training"
                # Empty slots may hold different filler (zeros here, a copy of frame 0 in training); they are masked
                # out and invisible to the model (tests/test_replayvla_model.py), so compare the real tokens only
                assert torch.allclose(visual[mask], ref_visual[ref_mask], atol=1e-5), f"{where}: visual tokens differ"
                checked += 1
        assert checked == 8


def test_memory_frames_encoded_once_and_pixels_released():
    # 1 anchor: memory = grid frames 8..72 (< 79) minus the 3 short ones = 8..48.  2 anchors: frame 8 is an anchor → 16..48
    for n_anchor, memory, kept in [(1, [8, 16, 24, 32, 40, 48], {0, 56, 64, 72}), (2, [16, 24, 32, 40, 48], {0, 8, 56, 64, 72})]:
        model = make_model(n_anchor)
        pixels = episode_pixels(1)
        featurized, added = [], []
        original_featurize, original_add = model._featurize, model.gist_encoder.add_frame
        model._featurize = lambda x: (featurized.append(x.shape[0]), original_featurize(x))[1]
        model.gist_encoder.add_frame = lambda *a: (added.append(1), original_add(*a))[1]
        model.gist_encoder.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("full gist pass at eval"))
        buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
        for t in range(T):
            buffer.add_frame(pixels[t])
            buffer.model_inputs()
        assert buffer.memory_frames == memory, buffer.memory_frames
        assert len(added) == len(memory), f"gist encoder ran {len(added)} times for {len(memory)} memory frames"
        # One ViT call per memory frame, plus one (batched) call for the anchors when the gist cache starts
        assert featurized == [1] * len(memory) + [n_anchor] or featurized == [n_anchor] + [1] * len(memory), featurized
        assert set(buffer.pixels) == kept, f"pixels kept for {sorted(buffer.pixels)}: memory frames should be dropped"


def test_every_gist_is_kept_and_fed():
    """MemoryWAM keeps every frame's gists (no cap): past the training cap, the LLM still sees all of them, unchanged."""
    model = make_model()
    pixels = episode_pixels(4)
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    for t in range(T):
        buffer.add_frame(pixels[t])
        mem = buffer.model_inputs()
        if t == 64:
            at_cap = mem["memory_gists"][0].clone()                  # frames 8..32 (= the tiny model's cap of 4)
    assert mem["memory_timesteps"].tolist() == [[8, 16, 24, 32, 40, 48]] and mem["memory_valid"].all()
    assert torch.equal(mem["memory_gists"][0, :4], at_cap), "earlier gists are reused as-is, never recomputed"
    assert buffer.cache.gists[0][0].shape[2] == 6 * model.gist_encoder.n_gist, "the cache keeps every frame's gists"
    assert all(len(r) == model.gist_encoder.n_recent for r in buffer.cache.recent_patches), "old patches are evicted"


def test_reset_starts_a_new_episode():
    model = make_model()
    buffer = ReplayMemoryBuffer(model, device=torch.device("cpu"), dtype=torch.float32)
    for t in range(40):
        buffer.add_frame(episode_pixels(2)[t])
    buffer.reset()
    assert buffer.t == -1 and not buffer.pixels and not buffer.gists and buffer.cache is None
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
