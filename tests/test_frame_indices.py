"""
Tests for memory_bank/frame_indices.py. Run from the repo root:

    python tests/test_frame_indices.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_bank.frame_indices import replay_frame_indices  # noqa: E402


def test_episode_start():
    idx = replay_frame_indices(0, stride=8, n_short=4, max_memory=4)
    assert idx.anchor == 0 and idx.current == 0
    assert idx.short_valid == [False, False, False]
    assert idx.memory_valid == [False] * 4


def test_before_first_grid_frame():
    for t in range(0, 9):  # grid frame 8 only counts once t > 8
        idx = replay_frame_indices(t, stride=8, n_short=4, max_memory=4)
        assert not any(idx.short_valid), f"t={t}: no grid frame < t yet"


def test_short_is_age_aligned():
    idx = replay_frame_indices(20, stride=8, n_short=4, max_memory=4)   # grid < 20: 8, 16
    assert idx.short == [0, 8, 16] and idx.short_valid == [False, True, True], idx
    assert not any(idx.memory_valid)


def test_current_on_grid_is_not_also_short():
    idx = replay_frame_indices(32, stride=8, n_short=4, max_memory=4)   # grid < 32: 8, 16, 24
    assert idx.current == 32
    assert idx.short == [8, 16, 24] and all(idx.short_valid)
    assert not any(idx.memory_valid)


def test_memory_fills_oldest_first():
    idx = replay_frame_indices(50, stride=8, n_short=4, max_memory=4)   # grid < 50: 8..48 (6 frames)
    assert idx.short == [32, 40, 48]
    assert idx.memory == [8, 16, 24, 0] and idx.memory_valid == [True, True, True, False], idx


def test_memory_cap_keeps_newest():
    idx = replay_frame_indices(100, stride=8, n_short=4, max_memory=4)  # grid < 100: 8..96 (12 frames)
    assert idx.short == [80, 88, 96]
    assert idx.memory == [48, 56, 64, 72] and all(idx.memory_valid), idx


def test_no_frame_repeats_and_all_before_t():
    for t in range(0, 600, 7):
        idx = replay_frame_indices(t, stride=8, n_short=4, max_memory=64)
        used = [f for f, v in zip(idx.memory, idx.memory_valid) if v] + [f for f, v in zip(idx.short, idx.short_valid) if v]
        assert len(used) == len(set(used)), f"t={t}: a frame appears twice"
        assert all(0 < f < t for f in used), f"t={t}: memory/short frames must lie strictly between the anchor and t"
        assert used == sorted(used), f"t={t}: memory then short must be in time order"


def test_libero_long_fits_without_cap():
    idx = replay_frame_indices(505, stride=8, n_short=4, max_memory=64)  # longest LIBERO-10 demo
    assert sum(idx.memory_valid) == 63 - 3 and all(idx.short_valid)


def test_fixed_grid_roles_are_stable():
    """A grid frame keeps the same frame index as t grows (needed for eval feature caching)."""
    a = replay_frame_indices(57, 8, 4, 64)
    b = replay_frame_indices(63, 8, 4, 64)
    assert a.short == b.short and a.memory == b.memory  # no new grid frame between 57 and 63 (56 < 57)


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
