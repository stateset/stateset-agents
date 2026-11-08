"""Reload actual PEFT artifacts and verify behavior, identity and failure paths."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    BehaviorLimits,
    BudgetExhausted,
    LowRankDelta,
    WeightSpaceSession,
)

MODULE = "model.layers.0.mlp.down_proj"


def setup(dtype=torch.float32, budget=10):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        config = transformers.Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        model = transformers.Qwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(0, 0.3)
        model.to(dtype)
    fresh = copy.deepcopy(model)
    session = WeightSpaceSession(
        model, lambda p: [ord(c) % 16 for c in p], budget=budget, batch_size=1
    )
    delta = LowRankDelta(torch.linspace(-0.1, 0.1, 8)[:, None], torch.ones(1, 16))
    return session, fresh, delta


def limits(**kwargs):
    return BehaviorLimits(
        **{
            "max_mean_kl": 1e-6,
            "max_token_kl": 1e-5,
            "max_nll_increase": 1e-4,
            **kwargs,
        }
    )


def state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state(model, expected):
    assert model.state_dict().keys() == expected.keys()
    assert all(torch.equal(v, expected[n]) for n, v in model.state_dict().items())


@pytest.mark.parametrize("mode", ["adapter", "merged"])
def test_real_peft_reload_matches_live_model_and_checks_references(tmp_path, mode):
    s, fresh, delta = setup()
    edit = s.apply(MODULE, delta)
    ref = s.add_behavior_reference(locality_prompts=["abc"], limits=limits())
    original = state(s.model)
    out = s.export(tmp_path / "adapter")
    model, report = s.verify_export(
        out,
        fresh,
        locality_prompts=["def"],
        target_pairs=[("ab", "c")],
        limits=limits(min_each_target_gain=-1e-4),
        mode=mode,
    )
    assert report.accepted and model is not None
    assert report.mode == mode and not report.violations
    assert report.reference_checks[0].reference_id == ref.reference_id
    assert not report.reference_checks[0].violations
    assert s.active_edit_ids == [edit] and s.budget.used == 4
    assert_state(s.model, original)
    for name, expected in report.artifact_sha256:
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    stored = json.loads((out / "weight_space_verification.json").read_text())
    assert stored["status"] == "passed" and stored["accepted"]
    assert stored["base_state_sha256"] == report.base_state_sha256
    assert dict(report.runtime)["torch"] == str(torch.__version__)
    if mode == "adapter":
        assert "default" in model.peft_config
    else:
        assert type(model) is type(s.model)
        assert not any("lora_" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("tied", [False, True])
def test_verification_with_independently_reloaded_base_checkpoint(tmp_path, tied):
    s, _, delta = setup()
    s.model.config.tie_word_embeddings = tied
    s.model.tie_weights()
    base_path = tmp_path / "base"
    s.model.save_pretrained(base_path, safe_serialization=True)
    fresh = transformers.AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval()
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    model, report = s.verify_export(
        out, fresh, locality_prompts=["abc"], limits=limits()
    )
    assert report.accepted and model is not None


def test_bfloat16_sequential_rounding_can_fail_reloaded_model_gate(tmp_path):
    s, _, _ = setup(torch.bfloat16)
    with torch.no_grad():
        s.model.get_submodule(MODULE).weight.fill_(1)
    fresh = copy.deepcopy(s.model)
    b = torch.zeros(8, 1)
    b[0] = 0.003
    delta = LowRankDelta(b, torch.ones(1, 16))
    original = state(s.model)
    s.apply(MODULE, delta)
    s.apply(MODULE, delta)
    assert_state(s.model, original)  # Each small increment rounds away.
    out = s.export(tmp_path / "adapter")
    manifest = json.loads((out / "weight_space_manifest.json").read_text())
    assert manifest["merge_weight_max_errors"][MODULE] == 0.0078125
    model, report = s.verify_export(
        out,
        fresh,
        locality_prompts=["ab cd ef", "gh ij kl"],
        limits=limits(max_mean_kl=0, max_token_kl=0),
        mode="merged",
    )
    assert model is None and not report.accepted
    assert report.metrics.mean_kl > 0 and "mean_kl" in report.violations
    assert_state(s.model, original)
    assert len(s.active_edit_ids) == 2
    assert (
        json.loads((out / "weight_space_verification.json").read_text())["status"]
        == "rejected"
    )


def test_export_cannot_bypass_active_reference_even_if_reload_matches_live(tmp_path):
    s, fresh, delta = setup()
    ref = s.add_behavior_reference(
        locality_prompts=["abc"], limits=limits(max_mean_kl=0, max_token_kl=0)
    )
    s.apply(MODULE, delta)  # Deliberately use the ungated low-level API.
    before = state(s.model)
    out = s.export(tmp_path / "adapter")
    model, report = s.verify_export(
        out, fresh, locality_prompts=["def"], limits=limits()
    )
    assert model is None and not report.accepted
    assert report.metrics.mean_kl < 1e-6
    assert f"reference:{ref.reference_id}:mean_kl" in report.violations
    assert_state(s.model, before)


@pytest.mark.parametrize(
    "name",
    ["adapter_config.json", "adapter_model.safetensors", "weight_space_manifest.json"],
)
def test_changed_artifact_is_rejected_before_loading_or_spending(tmp_path, name):
    s, fresh, delta = setup()
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    with (out / name).open("ab") as stream:
        stream.write(b" ")
    before = state(fresh)
    with pytest.raises(ConfigurationError, match="artifacts changed"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 0
    assert_state(fresh, before)


@pytest.mark.parametrize("change", ["edited", "other", "buffer", "dtype"])
def test_wrong_base_is_rejected_before_any_model_forward(tmp_path, change):
    s, fresh, delta = setup()
    if change == "buffer":
        s.model.register_buffer("probe", torch.tensor([1]))
        fresh.register_buffer("probe", torch.tensor([2]))
    elif change == "dtype":
        fresh.to(torch.bfloat16)
    else:
        weight = (
            fresh.get_submodule(MODULE).weight
            if change == "edited"
            else fresh.lm_head.weight
        )
        with torch.no_grad():
            weight.add_(0.01)
    s.apply(MODULE, delta)
    original = state(s.model)
    out = s.export(tmp_path / "adapter")
    with pytest.raises(ConfigurationError, match="fresh base tensor differs"):
        s.verify_export(out, fresh, locality_prompts=["abc"], limits=limits())
    assert s.budget.used == 0
    assert_state(s.model, original)
    assert s.export_verifications[-1]["status"] == "failed"


@pytest.mark.parametrize("share", ["model", "module", "storage"])
def test_shared_live_model_is_never_modified_by_peft_loading(tmp_path, share):
    s, fresh, delta = setup()
    s.apply(MODULE, delta)
    if share == "model":
        fresh = s.model
    elif share == "module":
        fresh.model = s.model.model
    else:
        fresh.lm_head.weight = torch.nn.Parameter(s.model.lm_head.weight.detach())
    before = state(s.model)
    out = s.export(tmp_path / "adapter")
    with pytest.raises(ConfigurationError, match="shares"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 0
    assert_state(s.model, before)


def test_identical_tensor_values_with_different_aliases_are_rejected(tmp_path):
    s, _, delta = setup()
    # Equal values with different aliases would create extra mutation paths
    # during a future adapter merge.
    with torch.no_grad():
        s.model.lm_head.weight.copy_(s.model.model.embed_tokens.weight)
    fresh = copy.deepcopy(s.model)
    fresh.lm_head.weight = fresh.model.embed_tokens.weight
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    with pytest.raises(ConfigurationError, match="storage aliases differ"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 0


def test_export_stack_and_budget_preflight(tmp_path):
    s, fresh, delta = setup(budget=1)
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    with pytest.raises(BudgetExhausted, match="2 measurement"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert not hasattr(fresh, "peft_config")
    s.apply(MODULE, delta)
    with pytest.raises(ConfigurationError, match="active edit stack"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 0


@pytest.mark.parametrize("failure", ["forward", "changed_artifact"])
def test_post_load_failure_records_failure_without_touching_session(tmp_path, failure):
    s, fresh, delta = setup()
    s.apply(MODULE, delta)
    original = state(s.model)
    out = s.export(tmp_path / "adapter")
    forward = fresh.forward

    def fail(*args, **kwargs):
        if failure == "forward":
            raise RuntimeError("candidate failed")
        with (out / "adapter_config.json").open("ab") as stream:
            stream.write(b" ")
        return forward(*args, **kwargs)

    fresh.forward = fail
    with pytest.raises((RuntimeError, ConfigurationError)):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert s.budget.used == 2
    assert_state(s.model, original)
    assert s.export_verifications[-1]["status"] == "failed"
    assert not json.loads((out / "weight_space_verification.json").read_text())[
        "accepted"
    ]


def test_reexport_resets_report_and_retains_prior_check_in_session(tmp_path):
    s, fresh, delta = setup()
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    _, report = s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert report.accepted
    s.export(out)
    stored = json.loads((out / "weight_space_verification.json").read_text())
    assert stored["status"] == "unverified" and not stored["accepted"]
    assert s.export_verifications[-1]["verification_id"] == report.verification_id


def test_failed_reexport_invalidates_old_passed_report(tmp_path, monkeypatch):
    import safetensors.torch

    s, fresh, delta = setup()
    s.apply(MODULE, delta)
    out = s.export(tmp_path / "adapter")
    _, report = s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
    assert report.accepted

    def fail_save(*args, **kwargs):
        raise OSError("storage failure")

    monkeypatch.setattr(safetensors.torch, "save_file", fail_save)
    with pytest.raises(OSError, match="storage failure"):
        s.export(out)
    stored = json.loads((out / "weight_space_verification.json").read_text())
    assert stored["status"] == "unverified" and not stored["accepted"]
    with pytest.raises(ConfigurationError, match="not created by this session"):
        s.verify_export(out, fresh, locality_prompts=["ab"], limits=limits())
