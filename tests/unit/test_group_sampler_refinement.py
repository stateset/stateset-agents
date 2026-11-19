"""Compare the optimizer's action ratios with the distribution used to draw those actions."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer  # noqa: E402
from stateset_agents.training.trainer_runtime import (  # noqa: E402
    group_old_logprobs,
    hf_generate_group,
)
from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer  # noqa: E402
from stateset_agents.training.vllm_backend import VLLMConfig  # noqa: E402


class Tokenizer:
    eos_token_id = 8
    pad_token_id = 9

    def __call__(self, prompt, **kwargs):
        ids = torch.tensor([[3, 4]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **kwargs):
        return str(ids.tolist())


def learner(kind, source="sampler", top_p=0.65, **config_overrides):
    torch.manual_seed(370)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10,
            n_embd=12,
            n_head=2,
            n_layer=1,
            n_positions=16,
            bos_token_id=3,
            eos_token_id=8,
            pad_token_id=9,
            resid_pdrop=0.4,
            attn_pdrop=0.4,
            embd_pdrop=0.4,
        )
    )
    model.generation_config.do_sample = True
    model.generation_config.top_k = 0
    rewards = iter([1.0, -0.5])
    args = {
        "group_size": 2,
        "num_gradient_updates": 2,
        "generation_batch_size": 2,
        "max_prompt_length": 4,
        "max_completion_length": 3,
        "top_p": top_p,
        "temperature": 0.7,
        "old_logprobs_source": source,
        "report_to": "none",
    }
    args.update(config_overrides)
    kwargs = {
        "model": model,
        "tokenizer": Tokenizer(),
        "reward_fn": lambda *_: next(rewards),
    }
    if kind == "dapo":
        return DAPOTrainer(
            DAPOConfig(**args, use_dynamic_sampling=False, use_vllm=False), **kwargs
        )
    t = VAPOTrainer(VAPOConfig(**args), **kwargs)
    t.value_warmup_complete = True
    return t


async def test_vapo_entropy_changes_a_seeded_gpt2_update():
    async def update(coefficient, overrides):
        trainer = learner(
            "vapo",
            entropy_coef=coefficient,
            objective_overrides=overrides,
            warmup_ratio=0.0,
            actor_learning_rate=0.001,
        )
        result = await trainer.train_step(["context"])
        weights = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
        return result["policy_loss"], weights

    base, weights = await update(0.0, {})
    regularized, changed = await update(0.7, {})
    overridden, restored = await update(0.7, {"entropy_coef": 0.0})
    assert regularized < base and (changed - weights).abs().max() > 0
    assert overridden == base
    torch.testing.assert_close(restored, weights, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["dapo", "vapo"])
@pytest.mark.parametrize("top_p", [1e-9, 0.65])
async def test_optimizer_uses_actual_hf_multinomial_probabilities(
    kind, top_p, monkeypatch
):
    t = learner(kind, top_p=top_p)
    draws = []
    multinomial = torch.multinomial

    def record(probabilities, *args, **kwargs):
        ids = multinomial(probabilities, *args, **kwargs)
        draws.append(probabilities.gather(1, ids).squeeze(1).log().detach().double())
        return ids

    monkeypatch.setattr(torch, "multinomial", record)
    name = (
        "compute_dapo_loss_from_log_probs" if kind == "dapo" else "compute_vapo_losses"
    )
    objective = getattr(t, name)
    seen = []

    def inspect(current, old, advantage, *extra):
        mask = (extra[0] if kind == "dapo" else extra[3]).bool()
        result = objective(current, old, advantage, *extra)
        measured = torch.stack(draws, 1)
        # Column 0 is the first prompt transition. Both rows have two prompt tokens.
        expected = torch.zeros_like(current, dtype=torch.float64)
        expected[:, 1:] = measured[:, : expected.shape[1] - 1]
        expected.masked_fill_(~mask, 0)
        torch.testing.assert_close(old.double(), expected, rtol=1e-6, atol=2e-7)
        if top_p == 1e-9:
            assert torch.all(old[mask] == 0)
        if not seen:
            assert (current[mask].double() - expected[mask]).abs().max() > 0.1
        else:
            torch.testing.assert_close(old, seen[0])
        seen.append(old.detach().clone())
        # Differentiate the clipped token objective independently at each retained draw.
        a = advantage.detach()
        if a.ndim == 1:
            a = a[:, None]
        a = a.expand_as(current)
        gradient = torch.zeros_like(current)
        value = 0.0
        count = int(mask.sum())
        for row, col in mask.nonzero().tolist():
            g = float(current[row, col].detach()) - float(expected[row, col])
            ratio = math.exp(min(20.0, max(-20.0, g)))
            lo, hi = 1 - t._objective.clip_low, 1 + t._objective.clip_high
            weight = float(a[row, col])
            value -= min(ratio * weight, min(hi, max(lo, ratio)) * weight) / count
            if abs(g) < 20 and not (
                (weight > 0 and ratio > hi) or (weight < 0 and ratio < lo)
            ):
                gradient[row, col] = -ratio * weight / count
        assert result[0].item() == pytest.approx(value, abs=4e-7)
        (actual,) = torch.autograd.grad(result[0], current, retain_graph=True)
        torch.testing.assert_close(actual, gradient, atol=3e-7, rtol=3e-6)
        return result

    monkeypatch.setattr(t, name, inspect)
    await t.train_step(["prompt"])
    assert len(seen) == 2


@pytest.mark.parametrize("kind", ["dapo", "vapo"])
def test_recomputed_actor_scores_restore_dropout_and_keep_gradients(kind):
    t = learner(kind, source="recompute")
    t.model.train()
    t.model.transformer.drop.eval()
    before = {m: m.training for m in t.model.modules()}
    ids = torch.tensor([[3, 4, 0]])
    mask = torch.tensor([[0.0, 0.0, 1.0]])

    def scores():
        value = t.compute_token_log_probs(ids, torch.ones_like(ids), mask)
        return value[0] if kind == "dapo" else value

    a, b = scores(), scores()
    assert torch.equal(a, b) and a.requires_grad
    assert {m: m.training for m in t.model.modules()} == before
    a.sum().backward()
    assert t.model.lm_head.weight.grad.abs().sum() > 0


async def test_processed_score_padding_and_multiple_eos_ids():
    class Model(torch.nn.Module):
        config = SimpleNamespace(use_cache=False)
        generation_config = SimpleNamespace(eos_token_id=[7, 8])

        def generate(self, **kwargs):
            assert (
                kwargs["return_dict_in_generate"]
                and kwargs["output_scores"]
                and kwargs["num_beams"] == 1
            )
            assert kwargs["eos_token_id"] == [7, 8]
            tokens = torch.tensor([[0, 7, 9], [1, 2, 8]])
            scores = []
            for k, weights in enumerate(((3, 2), (5, 3), (1, 7))):
                step = torch.full((2, 10), -math.inf, dtype=torch.float64)
                for i, w in enumerate(weights):
                    step[i, tokens[i, k]] = 100 + math.log(w)
                    step[i, 6] = 100
                if k == 2:
                    step[0] = math.nan
                scores.append(step)
            return SimpleNamespace(
                sequences=torch.cat((kwargs["input_ids"].expand(2, -1), tokens), 1),
                scores=scores,
            )

    c = learner("dapo").config
    rows = await hf_generate_group(Model(), Tokenizer(), c, "cpu", "prompt", 2)
    assert [r["input_ids"].tolist() for r in rows] == [[3, 4, 0, 7], [3, 4, 1, 2, 8]]
    for row, weights in zip(rows, ([3, 5], [2, 3, 7]), strict=True):
        assert row["sampler_log_probs"].tolist() == pytest.approx(
            [math.log(w / (1 + w)) for w in weights]
        )


def test_mixed_evidence_uses_only_explicitly_permitted_fallback():
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    rows = [{"sampler_log_probs": [-0.3, -1.2]}, {}, {}]
    cfg = SimpleNamespace(
        old_logprobs_source="sampler", allow_missing_sampler_logprobs=False
    )
    calls = []

    def rescore():
        calls.append(True)
        return torch.tensor(
            [[1.0, 1.0, 1.0], [math.nan, -2.0, math.nan], [3.0, 3.0, 3.0]]
        )

    with pytest.raises(ValueError, match="missing"):
        group_old_logprobs(rows, mask, cfg, rescore)
    assert not calls
    cfg.allow_missing_sampler_logprobs = True
    frozen = group_old_logprobs(rows, mask, cfg, rescore)
    assert len(calls) == 1
    torch.testing.assert_close(
        frozen,
        torch.tensor(
            [[-0.3, -1.2, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]], dtype=frozen.dtype
        ),
    )
    for invalid in [[0.0], [0.0, math.nan], [0.1, -1.0]]:
        with pytest.raises(ValueError):
            group_old_logprobs([{"sampler_log_probs": invalid}], mask[:1], cfg, rescore)


async def test_vllm_dapo_uses_processed_selected_scores():
    t = learner("dapo")
    cfg = VLLMConfig(logprobs_mode="processed_logprobs")
    assert cfg.to_vllm_kwargs()["logprobs_mode"] == "processed_logprobs"

    async def generate_groups(**kwargs):
        return {
            "q": [
                SimpleNamespace(
                    prompt="q",
                    response=str(i),
                    prompt_token_ids=[4],
                    response_token_ids=[0, 8],
                    sequence_length=2,
                    token_logprobs=[-i / 10.0, -0.7],
                    cumulative_logprob=-0.7 - i / 10.0,
                )
                for i in range(2)
            ]
        }

    t.vllm_generator = SimpleNamespace(config=cfg, generate_groups=generate_groups)
    rows = await t._generate_with_vllm("q")
    ids, attn, mask = t._build_batch_tensors(rows)
    frozen = t._rollout_logprobs(rows, ids, attn, mask)
    torch.testing.assert_close(
        frozen, torch.tensor([[0.0, -0.7], [-0.1, -0.7]], dtype=frozen.dtype)
    )


def test_rational_replay_matches_all_two_step_model_traces():
    from fractions import Fraction
    from itertools import product

    cfg = SimpleNamespace(
        old_logprobs_source="sampler", allow_missing_sampler_logprobs=False
    )

    def paths(prompt, sharp, narrow):
        def walk(previous, ids, probabilities):
            weights = {
                a: (2 if a == previous else 1) ** (2 if sharp else 1) for a in (1, 2)
            }
            if narrow:
                weights = {previous: weights[previous]}
            for a, w in weights.items():
                path = ids + [a]
                scores = probabilities + [Fraction(w, sum(weights.values()))]
                if a == 2 or len(path) == 2:
                    yield path, scores
                else:
                    yield from walk(a, path, scores)

        return [([], []), *walk(prompt, [], [])]

    comparisons = 0
    for sharp, narrow in product((False, True), repeat=2):
        for rows in product(paths(1, sharp, narrow), paths(2, sharp, narrow)):
            masks = torch.zeros((2, 4))
            records = []
            for i, (ids, probabilities) in enumerate(rows):
                masks[i, i + 1 : i + 1 + len(ids)] = 1.0
                records.append(
                    {"sampler_log_probs": [math.log(p) for p in probabilities]}
                )

            def forbidden():
                pytest.fail("complete recorded draws must never be rescored")

            frozen = group_old_logprobs(records, masks, cfg, forbidden)
            for head in (0, 1):
                raw = torch.zeros_like(frozen)
                expected = torch.zeros_like(frozen)
                for r, (tokens, _) in enumerate(rows):
                    prev = r + 1
                    for k, a in enumerate(tokens):
                        raw_mass = Fraction(
                            2 if a == (prev if head == 0 else 3 - prev) else 1, 3
                        )
                        # Independent table for the fixed behavior conditional.
                        mu = (
                            Fraction(1)
                            if narrow
                            else (
                                Fraction(4 if a == prev else 1, 5)
                                if sharp
                                else Fraction(2 if a == prev else 1, 3)
                            )
                        )
                        raw[r, r + k] = math.log(raw_mass)
                        expected[r, r + k] = float(raw_mass / mu)
                        prev = a
                actual = (raw - frozen).exp().masked_fill(~masks[:, 1:].bool(), 0)
                torch.testing.assert_close(actual, expected, rtol=3e-15, atol=0)
                comparisons += 1
    assert comparisons == 80


async def test_dapo_prepares_synchronous_engine_before_batch(monkeypatch):
    from stateset_agents.training import dapo_trainer as module
    from stateset_agents.training.vllm_backend import VLLMGenerator

    state = []

    class Engine:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            return True

        def sync_weights(self, model):
            state.append("sync")

        async def generate_groups(self, prompts, num_generations_per_prompt):
            assert state == ["sync"]
            state.append("draw")
            p = prompts[0]
            return {
                p: [
                    SimpleNamespace(
                        prompt=p,
                        response=str(i),
                        prompt_token_ids=[4],
                        response_token_ids=[0, 8],
                        sequence_length=2,
                        token_logprobs=[-0.2, -0.1],
                        cumulative_logprob=-0.3,
                    )
                    for i in range(num_generations_per_prompt)
                ]
            }

    monkeypatch.setattr(module, "_load_vllm_backend", lambda: True)
    monkeypatch.setattr(module, "VLLM_BACKEND_AVAILABLE", False)
    monkeypatch.setattr(module, "VLLMConfig", VLLMConfig)
    monkeypatch.setattr(module, "VLLMGenerator", Engine)
    previous = learner("dapo")
    previous.config.use_vllm = True
    t = DAPOTrainer(previous.config, previous.model, previous.tokenizer, lambda *_: 1.0)
    assert await t.initialize_vllm()
    assert t.vllm_generator.config.logprobs_mode == "processed_logprobs"
    samples, _ = await t.collect_samples_with_dynamic_sampling(["q"], 1)
    assert state == ["sync", "draw"]
    torch.testing.assert_close(
        samples[0]["old_token_log_probs"],
        torch.tensor([[-0.2, -0.1]] * 2, dtype=torch.float64),
    )
    # A weight publication can advertise the new version only after invalidating
    # KV entries for the old policy.
    backend = VLLMGenerator(VLLMConfig())
    calls = []
    backend.weight_loader = lambda weights: calls.append("load")

    def reset():
        calls.append("reset")
        return False

    backend.engine = SimpleNamespace(
        llm_engine=SimpleNamespace(reset_prefix_cache=reset)
    )
    with pytest.raises(RuntimeError, match="prefix cache"):
        backend.sync_weights(t.model)
    assert backend.weight_sync_count == 0 and calls == ["load", "reset"]
    backend.engine.llm_engine.reset_prefix_cache = lambda: True
    backend.sync_weights(t.model)
    assert backend.weight_sync_count == 1


@pytest.mark.parametrize("kind", ["dapo", "vapo"])
def test_resume_preserves_the_denominator_measure(kind, tmp_path):
    trainer = learner(kind)
    trainer.tokenizer = None  # Only the learner state is needed for this checkpoint.
    trainer.save_checkpoint(str(tmp_path / "sampled"))
    trainer.config.old_logprobs_source = "recompute"
    with pytest.raises(ValueError, match="old_logprobs_source"):
        trainer.load_runtime_checkpoint(str(tmp_path / "sampled"))
    trainer.config.old_logprobs_source = "sampler"
    assert trainer.load_runtime_checkpoint(str(tmp_path / "sampled"))


def empty_group(return_rows):
    async def generate(prompt):
        return [
            {
                "response": "",
                "input_ids": torch.tensor([3]),
                "attention_mask": torch.ones(1, dtype=torch.long),
                "response_mask": torch.zeros(1),
                "sequence_length": 0,
                "prompt_length": 1,
                "sampler_log_probs": [],
            }
            for _ in range(2 if return_rows else 0)
        ]

    return generate


@pytest.mark.parametrize("kind", ["dapo", "vapo"])
@pytest.mark.parametrize("return_rows", [False, True])
async def test_empty_draws_preserve_weights_moments_and_schedulers(
    kind, return_rows, monkeypatch
):
    from copy import deepcopy

    t = learner(kind)
    await t.train_step(["warm optimizer"])
    t.reward_fn = lambda *_: 0.5
    monkeypatch.setattr(t, "generate_group_responses", empty_group(return_rows))
    components = (
        (t.model, t.optimizer, t.scheduler)
        if kind == "dapo"
        else (
            t.model,
            t.value_head,
            t.actor_optimizer,
            t.critic_optimizer,
            t.actor_scheduler,
            t.critic_scheduler,
        )
    )
    before = [deepcopy(c.state_dict()) for c in components]
    metrics = await t.train_step(["empty"])
    assert metrics["policy_loss"] == 0
    for c, snapshot in zip(components, before, strict=True):
        torch.testing.assert_close(c.state_dict(), snapshot, atol=0, rtol=0)


async def test_value_warmup_counts_only_committed_critic_steps_across_resume(
    tmp_path, monkeypatch
):
    t = learner("vapo")
    t.value_warmup_complete = False
    t.config.value_warmup_steps = 2
    t.reward_fn = lambda *_: 1.0
    generate = t.generate_group_responses
    batches = iter([True, False])

    async def partial(prompt):
        if next(batches):
            return await generate(prompt)
        return await empty_group(True)(prompt)

    monkeypatch.setattr(t, "generate_group_responses", partial)
    await t.train_step(["prompt"])
    assert not t.value_warmup_complete
    assert {int(v["step"]) for v in t.critic_optimizer.state.values()} == {1}
    t.tokenizer = None
    t.save_checkpoint(str(tmp_path))
    resumed = learner("vapo")
    resumed.config.value_warmup_steps = 2
    resumed.load_checkpoint(str(tmp_path))
    await resumed.train_step([])
    assert not resumed.value_warmup_complete
    await resumed.train_step(["prompt"])
    assert resumed.value_warmup_complete
    assert {int(v["step"]) for v in resumed.critic_optimizer.state.values()} == {2}
