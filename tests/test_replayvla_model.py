"""
Tests for memory_bank/modeling.py on a tiny ReplayVLA (two vit_tiny backbones + a 2-layer Llama), CPU-only.
Run from the repo root:

    python tests/test_replayvla_model.py
"""

import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from memory_bank.configuration import ReplayVLAConfig  # noqa: E402
from memory_bank.modeling import ReplayVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import IGNORE_INDEX, OpenVLAForActionPrediction  # noqa: E402

B, S, M, G, T = 2, 3, 4, 2, 10        # batch, short slots, memory slots, gists/frame, text length
VOCAB = 32064                         # real Llama-2 vocab (+64 pad): predict_action appends token 29871
TEXT = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=4096)
NORM_STATS = {"toy": {"action": {"q01": [-1.0] * 7, "q99": [1.0] * 7}}}


def tiny(cfg):
    """Swap the 600M-parameter DINOv2 + SigLIP for two tiny ViTs (196 patches each, fused width 384)."""
    cfg.timm_model_ids = ["vit_tiny_patch16_224", "vit_tiny_patch16_224"]
    cfg.timm_override_act_layers = [None, None]
    cfg.image_sizes = [224, 224]
    return cfg


def replay_config():
    return tiny(ReplayVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px", llm_backbone_id="llama2-7b-pure", text_config=TEXT,
        norm_stats=NORM_STATS, n_short=S + 1, max_memory_frames=M, n_gist=G, gist_dim=32, gist_depth=1, gist_heads=4,
    ))


def make_model():
    torch.manual_seed(0)
    return ReplayVLAForActionPrediction(replay_config()).eval()


def make_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    px = lambda *shape: torch.randn(*shape, 6, 224, 224, generator=g)  # noqa: E731
    input_ids = torch.randint(1, 128, (B, T), generator=g)
    input_ids[:, 0] = 1  # BOS
    short_valid = torch.tensor([[False, True, True], [True, True, True]])
    memory_valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones(B, T, dtype=torch.long),
        pixel_values=px(B),
        anchor_pixel_values=px(B),
        short_pixel_values=px(B, S),
        short_valid=short_valid,
        memory_pixel_values=px(B, M),
        memory_valid=memory_valid,
        memory_timesteps=(torch.arange(M) + 1).repeat(B, 1) * 8,
    )


def run(model, **kw):
    with torch.no_grad():
        return model(**kw)


def noise_like(t, seed):
    return torch.randn(t.shape, generator=torch.Generator().manual_seed(seed))


P = 196
N_VISUAL = P + M * G + S * P + P


# ----------------------------------------------------------------------------------------------------------------------

def test_without_memory_equals_openvla():
    model = make_model()
    openvla = OpenVLAForActionPrediction(tiny(OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px", llm_backbone_id="llama2-7b-pure", text_config=TEXT,
        norm_stats=NORM_STATS))).eval()
    missing, unexpected = openvla.load_state_dict(model.state_dict(), strict=False)
    assert not missing, f"OpenVLA weights missing from ReplayVLA: {missing[:5]}"
    assert all(k.startswith(("gist_encoder.", "gist_projector.", "role_emb")) for k in unexpected), unexpected[:5]
    inp = make_inputs()
    kw = dict(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"])
    a, b = run(model, **kw).logits, run(openvla, **kw).logits
    assert torch.allclose(a, b, atol=1e-5), "without memory inputs, ReplayVLA must behave exactly like OpenVLA"


def test_shapes_and_num_visual_tokens():
    model = make_model()
    out = run(model, **make_inputs())
    assert out.num_visual_tokens == N_VISUAL, f"expected {N_VISUAL} visual tokens, got {out.num_visual_tokens}"
    assert out.logits.shape == (B, T + N_VISUAL, VOCAB), tuple(out.logits.shape)


def test_loss_only_on_text_labels():
    model = make_model()
    inp = make_inputs()
    labels = inp["input_ids"].clone()
    labels[:, :-4] = IGNORE_INDEX   # grade only the last 4 tokens, like OpenVLA's action tokens
    out = run(model, **inp, labels=labels)
    # Reproduce: logits at [V : -1] predict text tokens 1 … T-1
    pred = out.logits[:, out.num_visual_tokens : -1]
    expected = F.cross_entropy(pred.reshape(-1, VOCAB).float(), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
    assert torch.allclose(out.loss, expected, atol=1e-5), "loss must cover exactly the labelled text tokens"


def test_empty_slots_are_invisible():
    model = make_model()
    inp = make_inputs()
    base = run(model, **inp).logits
    inp2 = dict(inp)
    inp2["memory_pixel_values"] = inp["memory_pixel_values"].clone()
    inp2["short_pixel_values"] = inp["short_pixel_values"].clone()
    inp2["memory_pixel_values"][~inp["memory_valid"]] += noise_like(inp["memory_pixel_values"][~inp["memory_valid"]], 2)
    inp2["short_pixel_values"][~inp["short_valid"]] += noise_like(inp["short_pixel_values"][~inp["short_valid"]], 3)
    out = run(model, **inp2).logits
    assert torch.allclose(base[:, N_VISUAL:], out[:, N_VISUAL:], atol=1e-5), \
        "text/action logits changed when only EMPTY memory/short slots changed"


def test_memory_and_short_frames_are_used():
    model = make_model()
    with torch.no_grad():  # give role/gist pathways non-trivial weights so effects are visible
        model.role_emb.weight.normal_(std=0.02)
    inp = make_inputs()
    base = run(model, **inp).logits[:, N_VISUAL:]
    for key, valid_key, seed in [("memory_pixel_values", "memory_valid", 4), ("short_pixel_values", "short_valid", 5)]:
        inp2 = dict(inp)
        inp2[key] = inp[key].clone()
        inp2[key][inp[valid_key]] += noise_like(inp[key][inp[valid_key]], seed)
        out = run(model, **inp2).logits[:, N_VISUAL:]
        assert not torch.allclose(base, out, atol=1e-5), f"changing real {key} did not change the text logits"


def test_memory_features_path_matches_pixels():
    model = make_model()
    inp = make_inputs()
    base = run(model, **inp).logits
    feats = model.encode_memory_frames(inp["memory_pixel_values"], inp["memory_valid"])
    inp2 = {k: v for k, v in inp.items() if k != "memory_pixel_values"}
    out = run(model, **inp2, memory_features=feats).logits
    assert torch.allclose(base, out, atol=1e-5), "precomputed memory_features must give the same result as pixels"


def test_generate_uses_memory_and_matches_forward():
    model = make_model()
    inp = make_inputs()
    one = {k: v[:1] for k, v in inp.items()}
    with torch.no_grad():
        gen = model.generate(**one, max_new_tokens=3, do_sample=False)
        # Greedy reference without the KV cache: re-run the full forward each step
        ids = one["input_ids"]
        for _ in range(3):
            step = dict(one, input_ids=ids, attention_mask=torch.ones_like(ids))
            nxt = model(**step).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    assert torch.equal(gen[:, -3:], ids[:, -3:]), f"cached generate {gen[:, -3:].tolist()} != full forward {ids[:, -3:].tolist()}"


def test_predict_action_runs_with_memory():
    model = make_model()
    one = {k: v[:1] for k, v in make_inputs().items()}
    action = model.predict_action(**one, unnorm_key="toy", do_sample=False)
    assert action.shape == (7,), action.shape


def test_gradients_reach_memory_modules():
    model = make_model().train()
    inp = make_inputs()
    labels = inp["input_ids"].clone()
    labels[:, :-4] = IGNORE_INDEX
    model(**inp, labels=labels).loss.backward()
    for name in ["role_emb.weight", "gist_projector.fc1.weight", "gist_encoder.gist_queries", "gist_encoder.in_proj.weight"]:
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} got no gradient"


def test_from_openvla_checkpoint_initializes_new_modules():
    """Loading an OpenVLA checkpoint into ReplayVLA: base weights load, new modules get sane (non-garbage) init."""
    torch.manual_seed(0)
    openvla = OpenVLAForActionPrediction(tiny(OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px", llm_backbone_id="llama2-7b-pure", text_config=TEXT,
        norm_stats=NORM_STATS)))
    with tempfile.TemporaryDirectory() as tmp:
        openvla.save_pretrained(tmp)
        for low_cpu in [False, True]:
            model = ReplayVLAForActionPrediction.from_pretrained(tmp, config=replay_config(), low_cpu_mem_usage=low_cpu)
            sd, base = model.state_dict(), openvla.state_dict()
            assert all(torch.equal(sd[k], base[k]) for k in base), f"low_cpu={low_cpu}: OpenVLA weights not loaded"
            assert torch.equal(model.role_emb.weight, torch.zeros_like(model.role_emb.weight)), f"low_cpu={low_cpu}: role_emb not zero"
            for name, mod in model.gist_encoder.named_modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    assert torch.equal(mod.weight, torch.ones_like(mod.weight)) and torch.equal(mod.bias, torch.zeros_like(mod.bias)), \
                        f"low_cpu={low_cpu}: gist_encoder.{name} LayerNorm not initialized"
            for name, prm in model.named_parameters():
                assert torch.isfinite(prm).all(), f"low_cpu={low_cpu}: {name} has non-finite values"
            q = model.gist_encoder.gist_queries
            assert 0.005 < q.std().item() < 0.05, f"low_cpu={low_cpu}: gist_queries std {q.std().item():.4f}, expected ~0.02"
            out = run(model.eval(), **make_inputs())
            assert torch.isfinite(out.logits).all(), f"low_cpu={low_cpu}: non-finite logits"


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
