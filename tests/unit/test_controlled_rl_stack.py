"""Comprehensive unit & integration tests for the Controlled Multi-Model RL Stack:
- Closed-Universe Client Matter Sandbox (7th domain in MultiDomainSandboxPool + River bridge)
- Hierarchical Issue-Group Binary Rubric Reward (LLM judge + deterministic, zero length penalty)
- Rank-64 LoRA + Dr. GRPO / CISPO (`dr_cispo`) optimization & River `dr_grpo` normalization
- 5-Model Frontier Catalog, Nemotron 3.5 Super VL Anchor Sweep, Control Verification & Appendix Export
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from stateset_agents.core.client_matter_sandbox import (
    ClientMatterSandbox,
    default_client_matter_tasks,
)
from stateset_agents.core.sandbox_pool import MultiDomainSandboxPool
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.remote.model_catalog import get_model_support
from stateset_agents.remote.river_environment import river_mcp_sandbox_factory
from stateset_agents.remote.river_rl import RiverRLConfig, normalize_datums
from stateset_agents.rewards.rubric_reward import (
    CriterionVerdict,
    HierarchicalRubricReward,
    LLMRubricJudge,
    RubricCriterion,
    RubricRewardConfig,
    score_rubric_verdicts,
)
from stateset_agents.training.controlled_study import (
    ControlledStudyConfig,
    SweepTrialResult,
    build_controlled_run_plans,
    compute_dr_grpo_group_advantages,
    export_study_bundle,
    render_appendix_markdown,
    run_anchor_hyperparameter_sweep,
    run_offline_sandbox_stack_smoke,
    validate_study_controls,
)
from stateset_agents.training.objectives import (
    OBJECTIVES,
    compute_advantages,
    policy_loss,
)


def test_new_stack_modules_have_zero_eager_torch_imports() -> None:
    code = (
        "import sys; "
        "import stateset_agents.rewards.rubric_reward; "
        "import stateset_agents.core.client_matter_sandbox; "
        "import stateset_agents.training.controlled_study; "
        "assert 'torch' not in sys.modules, 'torch eagerly imported'"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_hierarchical_rubric_reward_exact_formula_and_zero_length_penalty() -> None:
    cfg = RubricRewardConfig(
        criteria_weight=0.50,
        issue_group_weight=0.35,
        all_pass_bonus=0.15,
    )
    assert cfg.length_penalty == 0.0

    with pytest.raises(ValueError, match="length_penalty"):
        RubricRewardConfig(length_penalty=0.05)

    # 4 criteria across 2 issue groups: group A passes in full (2/2), group B passes 1/2
    verdicts = [
        CriterionVerdict("c1", "group_a", True),
        CriterionVerdict("c2", "group_a", True),
        CriterionVerdict("c3", "group_b", True),
        CriterionVerdict("c4", "group_b", False),
    ]
    partial = score_rubric_verdicts(verdicts, cfg)
    assert partial["criteria_fraction"] == pytest.approx(0.75)
    assert partial["issue_group_fraction"] == pytest.approx(0.50)
    assert partial["all_criteria_passed"] is False
    assert partial["all_pass_bonus_awarded"] == 0.0
    assert partial["length_penalty"] == 0.0
    assert partial["reward"] == pytest.approx(0.50 * 0.75 + 0.35 * 0.50)

    # All 4 criteria pass -> full criteria + full issue groups + all-pass bonus = 1.0
    all_pass = score_rubric_verdicts([replace(v, passed=True) for v in verdicts], cfg)
    assert all_pass["criteria_fraction"] == pytest.approx(1.0)
    assert all_pass["issue_group_fraction"] == pytest.approx(1.0)
    assert all_pass["all_criteria_passed"] is True
    assert all_pass["all_pass_bonus_awarded"] == pytest.approx(0.15)
    assert all_pass["reward"] == pytest.approx(1.0)


def test_llm_rubric_judge_parses_json_and_fails_closed() -> None:
    criteria = (
        RubricCriterion("c1", "States the 15-day notice rule", "notice"),
        RubricCriterion("c2", "States the $1,500 fee", "fees"),
    )

    async def _fake_llm(prompt: str) -> str:
        if "c1" in prompt:
            return '```json\n{"passed": true, "rationale": "Notice period cited."}\n```'
        return "not valid json at all"

    judge = LLMRubricJudge(_fake_llm, judge_model="mock-judge", strict_fail_closed=True)
    reward_fn = HierarchicalRubricReward(criteria=criteria, judge=judge)
    result = asyncio.run(
        reward_fn.compute_reward(
            [
                ConversationTurn(
                    role="assistant", content="15 days written notice required."
                )
            ],
            context={"instruction": "Summarize notice and fees."},
        )
    )
    assert result.metadata["criteria_passed"] == 1
    assert result.metadata["criteria_total"] == 2
    assert result.metadata["issue_groups_passed"] == 1
    assert result.metadata["all_criteria_passed"] is False
    assert result.breakdown["length_penalty"] == 0.0
    assert result.score == pytest.approx(0.50 * 0.5 + 0.35 * 0.5)


def test_client_matter_sandbox_tools_confinement_and_grading() -> None:
    tasks = default_client_matter_tasks()
    assert len(tasks) == 3
    task = tasks[0]
    sb = ClientMatterSandbox(task=task)

    files_res = sb.call_tool("list_matter_files", {})
    assert not files_res["isError"]
    names = {f["filename"] for f in files_res["files"]}
    assert set(task.key_filenames).issubset(names)
    assert set(task.peripheral_filenames).issubset(names)

    # Path traversal and outside-matter reads must fail closed
    assert sb.call_tool("read_document", {"filename": "../etc/passwd"})["isError"]
    assert sb.call_tool("read_document", {"filename": "missing.txt"})["isError"]

    # Search across closed-universe matter finds controlling amendment
    search_res = sb.call_tool("search_matter", {"query": "Permitted Transfer"})
    assert not search_res["isError"]
    assert any(
        m["filename"] == "first_amendment_2024.txt" for m in search_res["matches"]
    )

    doc_res = sb.call_tool("read_document", {"filename": "first_amendment_2024.txt"})
    assert not doc_res["isError"]
    assert "Permitted Transfer" in doc_res["content"]

    draft_res = sb.call_tool(
        "draft_work_product",
        {
            "title": "Acme Lease Assignment Memo",
            "content": (
                "Master Lease Section 12.2 deems a merger an assignment requiring consent, "
                "but First Amendment Section 3 creates a Permitted Transfer carve-out for "
                "Vertex's $62M net worth (exceeding $50M). Acme must send 15 days prior to "
                "closing written notice with certified financial statements and pay $1,500. "
                "Harborview counsel's October 2026 email reserves Section 18.1 remedies, "
                "subject to a 15 business days cure period."
            ),
        },
    )
    assert draft_res["saved"] is True
    assert sb.call_tool("read_draft", {})["revision"] == 1

    done_res = sb.call_tool("signal_complete", {"summary": "Ready for partner review."})
    assert done_res["done"] is True
    assert sb.is_done is True

    score, details = sb.verify(None, [])
    assert score == pytest.approx(1.0)
    assert details["verified"] is True
    assert details["all_criteria_passed"] is True
    assert details["workspace"]["key_files_read"] == ["first_amendment_2024.txt"]
    assert details["workspace"]["peripheral_files_read"] == []


def test_client_matter_execution_limit_and_river_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    tiny_task = replace(default_client_matter_tasks()[0], max_steps=2)
    sb = ClientMatterSandbox(task=tiny_task)
    assert not sb.call_tool("list_matter_files", {})["isError"]
    assert not sb.call_tool("list_matter_files", {})["isError"]
    assert sb.execution_limit_reached is True
    over = sb.call_tool("search_matter", {"query": "Lease"})
    assert over["isError"] is True
    assert over["done"] is True

    # Verify River sandbox factory supports client_matter cleanly (even without river-client wheel)
    assert "client_matter" in MultiDomainSandboxPool.SUPPORTED_DOMAINS

    class _StubEnv:
        pass

    class _StubStepResult:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake_rl = types.SimpleNamespace(Env=_StubEnv, StepResult=_StubStepResult)
    fake_river = types.ModuleType("river_client")
    fake_river.rl = fake_rl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "river_client", fake_river)

    factory = river_mcp_sandbox_factory("client_matter", max_turns=4)
    env = factory()
    msgs = asyncio.run(env.reset({"case_id": "river-matter-smoke"}))
    assert isinstance(msgs, list) and len(msgs) == 1


def test_dr_cispo_objective_chunked_loss_and_river_dr_grpo_normalization() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.chunked_rl_loss import chunked_rl_policy_loss

    # 1. PolicyObjective preset matches Dr. GRPO + CISPO exactly
    obj = OBJECTIVES["dr_cispo"]
    assert obj.advantage == "group_mean"
    assert obj.clip == "cispo"
    assert obj.aggregate == "seq_sum_const"

    rewards = torch.tensor([1.0, 0.5, 0.0, 0.5])
    group_ids = torch.tensor([0, 0, 0, 0])
    adv = compute_advantages(rewards, group_ids, obj)
    # Exact reward - mean(rewards) with NO std division
    assert torch.allclose(adv, torch.tensor([0.5, 0.0, -0.5, 0.0]), atol=1e-6)
    assert compute_dr_grpo_group_advantages([1.0, 0.5, 0.0, 0.5]) == [
        0.5,
        0.0,
        -0.5,
        0.0,
    ]

    # 2. Chunked dr_cispo loss produces finite gradients and respects constant-length norm
    torch.manual_seed(11)
    bsz, seq_len, hidden_dim, vocab_size = 2, 16, 12, 64
    h = torch.randn(bsz, seq_len, hidden_dim, requires_grad=True)
    w = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    ids = torch.randint(0, vocab_size, (bsz, seq_len))
    mask = torch.ones(bsz, seq_len)
    mask[:, :4] = 0.0
    old_lp = torch.full((bsz, seq_len - 1), -2.5)
    roll_lp = old_lp - 0.15

    res = chunked_rl_policy_loss(
        h,
        w,
        ids,
        mask,
        old_lp,
        torch.tensor([0.5, -0.5]),
        algorithm="dr_cispo",
        rollout_logprobs=roll_lp,
        tis_clip_max=2.0,
        chunk_size=8,
        max_completion_length=32,
    )
    res["loss"].backward()
    assert torch.isfinite(res["loss"])
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()

    # Cross-check with declarative objectives.policy_loss on the same logprobs
    logits = torch.nn.functional.linear(h.detach(), w.detach())
    from stateset_agents.training.rl_losses import gather_token_logprobs

    cur_lp, shifted_mask = gather_token_logprobs(logits, ids, mask)
    ref_obj = obj.with_(max_completion_length=32)
    ref_out = policy_loss(
        logp_cur=cur_lp,
        logp_old=old_lp,
        mask=shifted_mask,
        advantages=torch.tensor([0.5, -0.5]),
        objective=ref_obj,
    )
    chunk_no_tis = chunked_rl_policy_loss(
        h.detach(),
        w.detach(),
        ids,
        mask,
        old_lp,
        torch.tensor([0.5, -0.5]),
        algorithm="dr_cispo",
        chunk_size=8,
        max_completion_length=32,
    )
    assert float(chunk_no_tis["pg_loss"]) == pytest.approx(
        float(ref_out.loss.item()), abs=1e-5
    )

    # 3. River RL dr_grpo normalization divides by retained * constant_length (no length bias)
    rcfg = RiverRLConfig(loss_fn="cispo", normalization="dr_grpo")
    assert rcfg.normalization == "dr_grpo"
    datums = [
        {"advantages": [0.0, 0.5, 0.5]},  # 2 active tokens
        {"advantages": [0.0, -0.5, -0.5, -0.5, -0.5]},  # 4 active tokens
    ]
    normed, total_toks = normalize_datums(datums, "dr_grpo", constant_length=10)
    assert total_toks == 6
    # Both sequences use denominator = retained (2) * constant_length (10) = 20
    assert normed[0]["advantages"] == pytest.approx([0.0, 0.025, 0.025])
    assert normed[1]["advantages"] == pytest.approx(
        [0.0, -0.025, -0.025, -0.025, -0.025]
    )


def test_five_model_catalog_sweep_transfer_controls_and_appendix(
    tmp_path: Path,
) -> None:
    # 1. All 5 study models resolve in model_catalog
    for name, expected_provider in [
        ("Muse Spark Open", "river"),
        ("Nemotron 3.5 Super VL", "river"),
        ("Beam 1.0", "river"),
        ("zai-org/GLM-5.3", "runpod"),
        ("Inkling 2 Small", "tinker"),
    ]:
        entry = get_model_support(name)
        assert entry is not None, f"Missing model_catalog entry for {name}"
        assert entry.recommended_provider == expected_provider

    # 2. Sweep exclusively on Nemotron 3.5 Super VL, then transfer to all 5 models
    cfg = ControlledStudyConfig()
    seen_models: set[str] = set()

    def _eval(spec, lr: float, group_size: int) -> SweepTrialResult:
        seen_models.add(spec.display_name)
        # Peak reward at lr=1e-4, group_size=8
        score = 0.88 if (lr == 1e-4 and group_size == 8) else 0.72
        return SweepTrialResult(
            anchor_model=spec.display_name,
            learning_rate=lr,
            rollout_group_size=group_size,
            validation_reward=score,
            criteria_pass_rate=score,
            issue_group_pass_rate=score - 0.05,
            all_pass_rate=score - 0.15,
        )

    selection = run_anchor_hyperparameter_sweep(cfg, _eval)
    assert seen_models == {"Nemotron 3.5 Super VL"}
    assert selection.selected_learning_rate == pytest.approx(1e-4)
    assert selection.selected_rollout_group_size == 8

    plans = build_controlled_run_plans(cfg, selection)
    assert len(plans) == 5
    controls = validate_study_controls(plans)
    assert controls["controls_verified"] is True
    assert controls["shared_lora_rank"] == 64
    assert controls["shared_objective"] == "dr_cispo"
    assert controls["length_penalty"] == 0.0
    assert {p.model.display_name: p.model.training_backend for p in plans} == {
        "Muse Spark Open": "river",
        "Nemotron 3.5 Super VL": "river",
        "Beam 1.0": "river",
        "GLM-5.3": "river",
        "Inkling 2 Small": "tinker",
    }

    # Control drift must fail closed
    drifted = list(plans)
    drifted[1] = replace(
        drifted[1], optimization=replace(drifted[1].optimization, learning_rate=3e-4)
    )
    with pytest.raises(ValueError, match="hyperparameter drift"):
        validate_study_controls(drifted)

    # Offline sandbox stack smoke proves ClientMatterSandbox + Dr. GRPO group_mean advantages
    smoke = asyncio.run(run_offline_sandbox_stack_smoke(cfg))
    assert smoke["strong_beats_partial"] is True
    assert smoke["group_advantages"][0] > 0.0
    assert smoke["group_advantages"][1] < 0.0

    # Export SHA-256 manifest + Appendix Markdown
    bundle = export_study_bundle(tmp_path, cfg, plans, selection=selection)
    assert len(bundle["manifest_sha256"]) == 64
    manifest = json.loads(Path(bundle["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["controls"]["controls_verified"] is True
    appendix = render_appendix_markdown(cfg, plans, selection=selection)
    for expected in (
        "Muse Spark Open",
        "Nemotron 3.5 Super VL",
        "Beam 1.0",
        "GLM-5.3",
        "Inkling 2 Small",
        "StateSet Sandboxes",
        "dr_cispo",
        "rank-64 LoRA",
        "Appendix C",
    ):
        assert expected in appendix
