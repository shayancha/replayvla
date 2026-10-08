"""
Behaviour tests for ReplayVLA's GistEncoder (Milestone 1a).

Spec: planning/milestones/m1a-gist-encoder.md. These tests check *behaviour* (what may and may not influence
what), not how the encoder is built. Run from the repo root:

    conda activate replayvla
    python tests/test_gist_encoder.py

The encoder lives in memory_bank/gist_encoder.py. It only depends on torch.
"""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_bank import GistEncoder  # noqa: E402

# Small sizes so everything runs on a laptop CPU in seconds
VD, D, G, DEPTH, HEADS, N_RECENT, MAX_T = 24, 32, 2, 2, 4, 3, 1024
B, M, P = 2, 7, 6
ATOL = 1e-5


def make_encoder(**overrides):
    torch.manual_seed(0)
    kwargs = dict(vision_dim=VD, d=D, n_gist=G, depth=DEPTH, heads=HEADS, n_recent=N_RECENT, max_timestep=MAX_T)
    kwargs.update(overrides)
    return GistEncoder(**kwargs).eval()


def make_inputs(seed=1):
    """Example 0: 5 real memory frames + 2 empty slots at the end. Example 1: all 7 real."""
    g = torch.Generator().manual_seed(seed)
    patches = torch.randn(B, M, P, VD, generator=g)
    anchor = torch.randn(B, P, VD, generator=g)
    valid = torch.ones(B, M, dtype=torch.bool)
    valid[0, 5:] = False
    timesteps = (torch.arange(M) + 1).repeat(B, 1) * 8          # 8, 16, 24, … (stride 8), oldest first
    timesteps[~valid] = 0                                        # empty slots: arbitrary, must not matter
    return patches, valid, timesteps, anchor


def run(enc, patches, valid, timesteps, anchor):
    with torch.no_grad():
        return enc(patches, valid, timesteps, anchor)


def noise_like(t, seed):
    return 3.0 * torch.randn(t.shape, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------------------------------

def test_shapes():
    enc = make_encoder()
    p, v, t, a = make_inputs()
    out = run(enc, p, v, t, a)
    assert out.shape == (B, M, G, D), f"expected gists [B, M, G, d] = {(B, M, G, D)}, got {tuple(out.shape)}"
    # Nothing hard-coded: other numbers of frames / patches must work too
    out2 = run(enc, p[:, :3, :4], v[:, :3], t[:, :3], a[:, :4])
    assert out2.shape == (B, 3, G, D), f"with M=3, P=4 expected {(B, 3, G, D)}, got {tuple(out2.shape)}: is M or P hard-coded?"


def test_empty_frames_output_zeros():
    enc = make_encoder()
    p, v, t, a = make_inputs()
    out = run(enc, p, v, t, a)
    assert torch.equal(out[~v], torch.zeros_like(out[~v])), "gists of empty memory slots must be exactly zero"
    assert out[v].abs().sum() > 0, "gists of real frames are all zero"


def test_no_future_leakage():
    """Changing frame j must not change the gists of any earlier frame, but must change frame j's own gists."""
    enc = make_encoder()
    p, v, t, a = make_inputs()
    base = run(enc, p, v, t, a)
    j = 3
    p2 = p.clone()
    p2[:, j] += noise_like(p2[:, j], seed=2)
    out = run(enc, p2, v, t, a)
    assert torch.allclose(base[:, :j], out[:, :j], atol=ATOL), \
        f"changing frame {j}'s patches changed gists of EARLIER frames (< {j}): memory must only flow forward in time"
    assert not torch.allclose(base[:, j], out[:, j], atol=ATOL), \
        f"changing frame {j}'s patches did not change frame {j}'s own gists: do gists attend to their own frame?"
    # Same for the timestep of frame j
    t2 = t.clone()
    t2[:, j] += 1
    out_t = run(enc, p, v, t2, a)
    assert torch.allclose(base[:, :j], out_t[:, :j], atol=ATOL), \
        f"changing frame {j}'s timestep changed gists of earlier frames"


def test_memory_flows_forward():
    """Frame 0 must still influence the gists of a frame far beyond any recent window.

    With 12 frames and depth 2, information can't hop there through recent-window patches alone
    (that path covers ~n_recent frames per layer), so it has to travel through the gist bank.
    """
    enc = make_encoder()
    g = torch.Generator().manual_seed(7)
    m = 12
    p = torch.randn(1, m, P, VD, generator=g)
    a = torch.randn(1, P, VD, generator=g)
    v = torch.ones(1, m, dtype=torch.bool)
    t = (torch.arange(m) + 1)[None] * 8
    base = run(enc, p, v, t, a)
    p2 = p.clone()
    p2[:, 0] += noise_like(p2[:, 0], seed=3)
    out = run(enc, p2, v, t, a)
    assert not torch.allclose(base[0, m - 1], out[0, m - 1], atol=ATOL), (
        f"changing frame 0 did not change the gists of frame {m - 1}. Do gists attend to the gists of ALL "
        "earlier frames (the memory bank), not just their own frame and the recent window?"
    )


def test_anchor_is_used():
    enc = make_encoder()
    p, v, t, a = make_inputs()
    base = run(enc, p, v, t, a)
    out = run(enc, p, v, t, a + noise_like(a, seed=4))
    for b in range(B):
        for i in range(M):
            if v[b, i]:
                assert not torch.allclose(base[b, i], out[b, i], atol=ATOL), \
                    f"changing the anchor did not change gists of real frame {i} (example {b}): do gists attend to the anchor?"


def test_empty_frames_invisible():
    """Patches and timesteps of empty slots must not influence any real frame's gists."""
    enc = make_encoder()
    p, v, t, a = make_inputs()
    base = run(enc, p, v, t, a)
    p2, t2 = p.clone(), t.clone()
    p2[~v] += noise_like(p2[~v], seed=5)
    t2[~v] = 999
    out = run(enc, p2, v, t2, a)
    assert torch.allclose(base[v], out[v], atol=ATOL), \
        "real frames' gists changed when only EMPTY slots changed: empty frames must be masked out as keys"


def test_no_nan_at_episode_start():
    """Step 0 of an episode: no memory frames at all. Then: exactly one."""
    enc = make_encoder()
    p, v, t, a = make_inputs()
    none = torch.zeros_like(v)
    pg = p.clone().requires_grad_(True)
    out = enc(pg, none, t, a)
    assert not torch.isnan(out).any(), "NaN in output when every memory slot is empty"
    assert torch.equal(out, torch.zeros_like(out)), "with every slot empty, every gist must be zero"
    (out.sum() + 0 * pg.sum()).backward()
    assert pg.grad is not None and not torch.isnan(pg.grad).any(), "NaN in gradients when every memory slot is empty"
    one = torch.zeros_like(v)
    one[:, 0] = True
    out1 = run(enc, p, one, t, a)
    assert not torch.isnan(out1).any(), "NaN in output with a single memory frame"
    assert out1[:, 0].abs().sum() > 0, "with a single real frame, its gists must not be zero"


def test_timesteps_are_used():
    enc = make_encoder()
    p, v, t, a = make_inputs()
    base = run(enc, p, v, t, a)
    t2 = t.clone()
    t2[v] += 1
    out = run(enc, p, v, t2, a)
    assert not torch.allclose(base[v], out[v], atol=ATOL), \
        "shifting every timestep did not change any gist: is the time embedding added?"


def test_batch_independence():
    enc = make_encoder()
    p, v, t, a = make_inputs()
    full = run(enc, p, v, t, a)
    alone = run(enc, p[:1], v[:1], t[:1], a[:1])
    assert torch.allclose(full[:1], alone, atol=ATOL), "example 0's gists depend on example 1: something mixes the batch axis"


def test_gradients_reach_every_parameter():
    enc = make_encoder().train()
    p, v, t, a = make_inputs()
    out = enc(p, v, t, a)
    out[v].pow(2).mean().backward()
    dead = [n for n, prm in enc.named_parameters() if prm.grad is None or prm.grad.abs().sum() == 0]
    assert not dead, f"these parameters got no gradient (unused in forward?): {dead}"


TESTS = [
    test_shapes,
    test_empty_frames_output_zeros,
    test_no_future_leakage,
    test_memory_flows_forward,
    test_anchor_is_used,
    test_empty_frames_invisible,
    test_no_nan_at_episode_start,
    test_timesteps_are_used,
    test_batch_independence,
    test_gradients_reach_every_parameter,
]

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
