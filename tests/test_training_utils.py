"""
Tests for memory_bank/training.py on a tiny model: loading OpenVLA weights as ReplayVLA, LoRA wrapping with memory
modules trained in full, metrics, and the save-adapter -> merge round trip. Run from the repo root:

    python tests/test_training_utils.py
"""

import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_bank.modeling import ReplayVLAForActionPrediction  # noqa: E402
from memory_bank.training import (  # noqa: E402
    MEMORY_MODULES,
    action_metrics,
    batch_to_model_inputs,
    load_replayvla,
    lora_target_modules,
    merge_lora,
    upcast_memory_modules,
    wrap_with_lora,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import IGNORE_INDEX, OpenVLAForActionPrediction  # noqa: E402
from test_replayvla_model import M, G, NORM_STATS, S, TEXT, make_inputs, tiny  # noqa: E402

MEMORY_KWARGS = dict(n_short=S + 1, max_memory_frames=M, n_gist=G, gist_dim=32, gist_depth=1, gist_heads=4)


def save_tiny_openvla(path):
    torch.manual_seed(0)
    OpenVLAForActionPrediction(tiny(OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px", llm_backbone_id="llama2-7b-pure", text_config=TEXT,
        norm_stats=NORM_STATS))).save_pretrained(path)


def labelled_inputs():
    inp = make_inputs()
    labels = inp["input_ids"].clone()
    labels[:, :-4] = IGNORE_INDEX
    return dict(inp, labels=labels)


def test_load_openvla_checkpoint_as_replayvla():
    with tempfile.TemporaryDirectory() as tmp:
        save_tiny_openvla(tmp)
        model = load_replayvla(tmp, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
        assert isinstance(model, ReplayVLAForActionPrediction) and model.config.model_type == "replayvla"
        assert model.config.max_memory_frames == M and model.config.timm_model_ids[0] == "vit_tiny_patch16_224"
        assert model.config.norm_stats == NORM_STATS, "norm stats (for un-normalizing actions) must carry over"


def test_lora_targets():
    with tempfile.TemporaryDirectory() as tmp:
        save_tiny_openvla(tmp)
        model = load_replayvla(tmp, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
    targets = lora_target_modules(model)
    assert not any(t.startswith(MEMORY_MODULES) for t in targets), "memory modules must be trained in full, not LoRA'd"
    assert not any("lm_head" in t for t in targets)
    for part in ["language_model.model.layers.0.self_attn.q_proj", "vision_backbone.featurizer", "projector.fc1"]:
        assert any(t.startswith(part) for t in targets), f"expected LoRA on {part}"


def test_lora_model_trains_memory_in_full_and_merges():
    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, merged_dir = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "merged"
        save_tiny_openvla(base_dir)
        model = load_replayvla(base_dir, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
        upcast_memory_modules(model)
        vla = wrap_with_lora(model, rank=4)

        trainable = {n for n, p in vla.named_parameters() if p.requires_grad}
        assert any("lora_A" in n for n in trainable), "no LoRA parameters are trainable"
        for mod in MEMORY_MODULES:
            assert any(f"{mod}.modules_to_save" in n for n in trainable), f"{mod} is not trained in full"
        assert not any("q_proj.base_layer" in n for n in trainable), "frozen base weights became trainable"

        # A few optimizer steps so LoRA and memory modules move away from their init
        opt = torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad], lr=1e-2)
        inp = labelled_inputs()
        vla.train()
        for _ in range(3):
            loss = vla(**inp).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        trained_role = vla.base_model.model.role_emb.modules_to_save["default"].weight.detach().clone()
        assert trained_role.abs().sum() > 0, "role embeddings did not train"

        vla.eval()
        with torch.no_grad():
            ref = vla(**make_inputs()).logits
        vla.save_pretrained(adapter_dir)

        merged = merge_lora(base_dir, adapter_dir, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32).eval()
        assert torch.allclose(merged.role_emb.weight, trained_role), "merging lost the trained role embeddings"
        with torch.no_grad():
            out = merged(**make_inputs()).logits
        assert torch.allclose(ref, out, atol=1e-4), f"merged model differs from LoRA model (max {(ref - out).abs().max():.2e})"

        # Merged checkpoint reloads as ReplayVLA with identical outputs
        merged.save_pretrained(merged_dir)
        reloaded = load_replayvla(merged_dir, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            again = reloaded(**make_inputs()).logits
        assert torch.allclose(out, again, atol=1e-5), "reloading the merged ReplayVLA checkpoint changed outputs"


def test_action_metrics_and_batch_inputs():
    class Tok:  # minimal ActionTokenizer stand-in
        action_token_begin_idx = 100

        def decode_token_ids_to_actions(self, ids):
            return ids.astype("float32")

    logits = torch.zeros(1, 3 + 5, 200)
    labels = torch.tensor([[1, 150, 151, IGNORE_INDEX, 160]])
    for pos, tok in enumerate([150, 151, 7, 160]):   # logits[3 + pos] predicts labels[pos + 1]
        logits[0, 3 + pos, tok] = 1.0
    m = action_metrics(logits, labels, num_visual_tokens=3, action_tokenizer=Tok())
    assert m["action_accuracy"] == 1.0 and m["l1_loss"] == 0.0, m

    batch = dict(labelled_inputs(), dataset_names=["toy"] * 2)
    inputs = batch_to_model_inputs(batch, device="cpu", dtype=torch.bfloat16)
    assert "dataset_names" not in inputs and inputs["memory_pixel_values"].dtype == torch.bfloat16
    assert inputs["memory_valid"].dtype == torch.bool and inputs["memory_timesteps"].dtype == torch.long


def test_bf16_autocast_with_fp32_memory_modules():
    """The training setup: base in bf16, memory modules upcast to fp32, forward under bf16 autocast."""
    with tempfile.TemporaryDirectory() as tmp:
        save_tiny_openvla(tmp)
        model = load_replayvla(tmp, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.bfloat16)
    upcast_memory_modules(model)
    model.train()
    inp = batch_to_model_inputs(labelled_inputs(), device="cpu", dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**inp)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.gist_encoder.gist_queries.grad is not None and model.gist_encoder.gist_queries.dtype == torch.float32


def test_gradient_checkpointing_with_lora():
    """The training script's setup order: enable HF checkpointing, then set the gist flag, then wrap with LoRA."""
    with tempfile.TemporaryDirectory() as tmp:
        save_tiny_openvla(tmp)
        model = load_replayvla(tmp, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.language_model.model.gradient_checkpointing, "LLM checkpointing not enabled"
    flipped = model.gist_encoder.gradient_checkpointing
    model.gist_encoder.gradient_checkpointing = False
    vla = wrap_with_lora(model, rank=4).train()
    loss = vla(**labelled_inputs()).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for n, p in vla.named_parameters() if "lora_A" in n), "LoRA got no gradient"
    gist = vla.base_model.model.gist_encoder.modules_to_save["default"]
    assert gist.gist_queries.grad is not None and gist.gradient_checkpointing is False
    print(f"   (HF gradient_checkpointing_enable flipped the gist encoder flag: {flipped})")


def test_resume_checkpoint_round_trip():
    """Save mid-training, resume into a fresh base: identical weights, optimizer state, and next step."""
    from memory_bank.training import find_resume_checkpoint, load_resume_checkpoint, save_resume_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        base_dir, run_dir = Path(tmp) / "base", Path(tmp) / "run"
        save_tiny_openvla(base_dir)
        run_dir.mkdir()
        inp = labelled_inputs()

        def fresh_base():
            m = load_replayvla(base_dir, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
            upcast_memory_modules(m)
            return m

        def step(model, opt):
            model.train()
            loss = model(**inp).loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.manual_seed(0)
        vla = wrap_with_lora(fresh_base(), rank=4)
        opt = torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(2):
            step(vla, opt)
        save_resume_checkpoint(vla, opt, {"completed_steps": 2, "wandb_run_id": "abc"}, run_dir)

        # A job killed mid-save leaves a partial resume.tmp.<id>; it must be ignored
        (run_dir / "resume.tmp.dead").mkdir()
        assert find_resume_checkpoint(run_dir) == run_dir / "resume"

        resumed, opt_state, state = load_resume_checkpoint(fresh_base(), find_resume_checkpoint(run_dir))
        opt2 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-2)
        opt2.load_state_dict(opt_state)
        assert state == {"completed_steps": 2, "wandb_run_id": "abc"}

        a = {n: p for n, p in vla.named_parameters() if p.requires_grad}
        b = {n: p for n, p in resumed.named_parameters() if p.requires_grad}
        assert a.keys() == b.keys(), "trainable parameter sets differ after resume"
        assert all(torch.equal(a[n], b[n]) for n in a), "weights differ after resume"

        torch.manual_seed(1); step(vla, opt)
        torch.manual_seed(1); step(resumed, opt2)
        diff = max((a[n] - b[n]).abs().max().item() for n in a)
        assert diff < 1e-5, f"the step after resuming differs from continuing (max diff {diff:.2e})"

        # Second save swaps atomically; a kill between the two renames leaves only resume.old, which is still found
        save_resume_checkpoint(vla, opt, {"completed_steps": 3}, run_dir)
        assert not list(run_dir.glob("resume.tmp.*")) and not list(run_dir.glob("resume.old.*")), "stale dirs left"
        (run_dir / "resume").rename(run_dir / "resume.old.x")
        assert find_resume_checkpoint(run_dir) == run_dir / "resume.old.x"


def test_snapshots_keep_the_adapter_per_step():
    from memory_bank.training import find_snapshot, save_snapshot

    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, run_dir = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "run"
        save_tiny_openvla(base_dir)
        torch.manual_seed(0)
        vla = wrap_with_lora(load_replayvla(base_dir, memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32), rank=4)
        with torch.no_grad():
            vla.base_model.model.role_emb.modules_to_save["default"].weight.fill_(0.5)
        vla.save_pretrained(adapter_dir)
        snap = save_snapshot(adapter_dir, run_dir, 5000)
        assert snap == run_dir / "snapshots" / "step5000" and find_snapshot(run_dir, 5000) == snap

        # The adapter dir is overwritten at the next save; the snapshot keeps step 5000's weights
        with torch.no_grad():
            vla.base_model.model.role_emb.modules_to_save["default"].weight.fill_(-1.0)
        vla.save_pretrained(adapter_dir)
        save_snapshot(adapter_dir, run_dir, 10000)
        merged = merge_lora(base_dir, snap / "adapter", memory_kwargs=MEMORY_KWARGS, torch_dtype=torch.float32)
        assert torch.all(merged.role_emb.weight == 0.5), "the snapshot must hold the weights of its own step"
        assert not list((run_dir / "snapshots").glob(".tmp-*")), "temporary snapshot dirs left behind"

        # Missing step: a clear error listing what exists; an incomplete leftover is not used
        (run_dir / "snapshots" / "step15000").mkdir()
        try:
            find_snapshot(run_dir, 15000)
            raise AssertionError("an incomplete snapshot was accepted")
        except FileNotFoundError as e:
            assert "[5000, 10000]" in str(e), str(e)


def test_stop_request_signals_and_file():
    import os
    import signal

    from memory_bank.training import StopRequest, sync_any

    with tempfile.TemporaryDirectory() as tmp:
        stop_file = Path(tmp) / "STOP"
        req = StopRequest(stop_file)
        assert not req.local_request()
        stop_file.touch()
        assert req.local_request() and "stop file" in req.reason
        req2 = StopRequest(None)
        os.kill(os.getpid(), signal.SIGUSR1)
        assert req2.local_request() and "signal" in req2.reason
        assert sync_any([True, False]) == [True, False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):  # restore defaults for the rest of the test run
        signal.signal(sig, signal.SIG_DFL)


def test_openvla_baseline_mode():
    """--use_memory False: plain OpenVLA with the same LoRA recipe, metrics offset, merge and resume machinery."""
    from memory_bank.training import find_resume_checkpoint, load_openvla, load_resume_checkpoint, save_resume_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        base_dir, adapter_dir, run_dir = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "run"
        save_tiny_openvla(base_dir)
        run_dir.mkdir()
        base = load_openvla(base_dir, torch_dtype=torch.float32)
        assert type(base) is OpenVLAForActionPrediction
        vla = wrap_with_lora(base, rank=4)
        trainable = [n for n, p in vla.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable), "baseline should train LoRA only (no memory modules)"
        # The training loop reads the visual-token count through DDP(PeftModel(...)).module
        assert vla.vision_backbone.featurizer.patch_embed.num_patches == 196

        inp = labelled_inputs()
        batch = {k: inp[k] for k in ("input_ids", "attention_mask", "pixel_values", "labels")}
        inputs = batch_to_model_inputs(batch, device="cpu", dtype=torch.float32)
        assert set(inputs) == set(batch), "memory keys must be skipped when the batch has none"
        opt = torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad], lr=1e-2)
        vla.train()
        for _ in range(2):
            loss = vla(**inputs).loss
            opt.zero_grad(); loss.backward(); opt.step()
        out = vla(**inputs)
        assert getattr(out, "num_visual_tokens", None) is None   # -> loop falls back to the patch count
        # begin_idx 0: every labelled token counts as an "action" token, so the accuracy is never 0/0 = NaN
        m = action_metrics(out.logits, inputs["labels"], 196, type("T", (), {"action_token_begin_idx": 0,
                           "decode_token_ids_to_actions": staticmethod(lambda ids: ids.astype("float32"))})())
        assert 0.0 <= m["action_accuracy"] <= 1.0, f"accuracy {m['action_accuracy']}"

        vla.eval()
        with torch.no_grad():
            ref = vla(**inputs).logits
        vla.save_pretrained(adapter_dir)
        merged = merge_lora(base_dir, adapter_dir, use_memory=False, torch_dtype=torch.float32).eval()
        assert type(merged) is OpenVLAForActionPrediction
        with torch.no_grad():
            assert torch.allclose(ref, merged(**inputs).logits, atol=1e-4), "merged baseline differs from LoRA model"

        save_resume_checkpoint(vla, opt, {"completed_steps": 2}, run_dir)
        resumed, opt_state, state = load_resume_checkpoint(load_openvla(base_dir, torch_dtype=torch.float32),
                                                           find_resume_checkpoint(run_dir))
        a = {n: p for n, p in vla.named_parameters() if p.requires_grad}
        b = {n: p for n, p in resumed.named_parameters() if p.requires_grad}
        assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a) and state["completed_steps"] == 2


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
