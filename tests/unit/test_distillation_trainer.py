"""DistillationTrainer mechanics on tiny GPT-2 students and teachers."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationDataset,
    DistillationTrainer,
    TeacherCache,
    VocabularyMismatchError,
    precompute_teacher_cache,
    train_distillation,
)
from stateset_agents.training.distillation import (  # noqa: E402
    trainer as trainer_module,
)
from stateset_agents.training.preference.sampling import (  # noqa: E402
    sample_completions,
)
from tests.learning._toy import VOCAB_SIZE, make_model, make_tokenizer  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return make_tokenizer()


TRAIN = ["hello", "count", "say a number", "abc def"]
HELD = ["xyz", "ok go"]
LINEAR_ADAM = {"adam_epsilon": 1.0}


def _data(prompts: list[str], *, completions: bool = True, **extra: Any) -> Any:
    return DistillationDataset.from_records(
        {
            "prompt": [{"role": "user", "content": prompt}],
            **({"completion": "7" * (1 + index % 3) + "ab"} if completions else {}),
            **extra,
        }
        for index, prompt in enumerate(prompts)
    )


def _config(**overrides: Any) -> DistillationConfig:
    values: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "max_steps": 2,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
        "lr_scheduler": "constant",
        "max_new_tokens": 4,
    }
    values.update(overrides)
    return DistillationConfig(**values)


def _trainer(
    tokenizer: Any,
    seed: int = 0,
    train: Any = None,
    teacher: Any = "default",
    **overrides: Any,
) -> DistillationTrainer:
    return DistillationTrainer(
        make_model(seed, n_embd=16),
        tokenizer,
        _config(**overrides),
        train if train is not None else _data(TRAIN),
        _data(HELD),
        teacher=make_model(50, n_embd=16) if teacher == "default" else teacher,
    )


def _params(model: Any) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _max_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------- the basics


def test_teacher_is_frozen_and_student_learns(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer)
    teacher_before, student_before = _params(trainer.teacher), _params(trainer.model)
    before = trainer.evaluate()
    summary = trainer.train()
    assert summary.global_step == 2 and summary.skipped_steps == 0
    assert all(not p.requires_grad for p in trainer.teacher.parameters())
    assert not trainer.teacher.training
    assert _max_diff(teacher_before, _params(trainer.teacher)) == 0.0
    assert _max_diff(student_before, _params(trainer.model)) > 0.0
    assert summary.eval_metrics["eval/forward_kl"] < before["eval/forward_kl"]
    assert {"forward_kl", "reverse_kl", "divergence", "tokens"} <= set(
        summary.history[-1]
    )


def test_accumulation_equals_one_large_token_weighted_batch(tokenizer: Any) -> None:
    weighted = _data(TRAIN)
    weighted = DistillationDataset(
        tuple(
            type(example)(example.prompt, example.completion, weight=1.0 + index)
            for index, example in enumerate(weighted.examples)
        )
    )
    big = _trainer(tokenizer, train=weighted, max_steps=1, **LINEAR_ADAM)
    small = _trainer(
        tokenizer,
        train=weighted,
        max_steps=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        **LINEAR_ADAM,
    )
    big.train()
    small.train()
    assert big.global_step == small.global_step == 1
    assert _max_diff(_params(big.model), _params(small.model)) < 1e-6


def test_only_response_tokens_count(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer)
    sample = trainer_module._Sample(trainer._train_items[0].sequence, 1.0, False)
    loss = trainer._compute([sample])
    assert loss.tokens == len(trainer._train_items[0].sequence.response_ids)
    assert loss.weights.tolist() == [float(loss.tokens)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"teacher": None}, "teacher"),
        ({"train": _data(TRAIN, completions=False)}, "completion"),
    ],
)
def test_construction_errors(tokenizer: Any, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _trainer(tokenizer, **kwargs)


def test_teacher_must_be_a_separate_model(tokenizer: Any) -> None:
    model = make_model(0, n_embd=16)
    with pytest.raises(ValueError, match="separate"):
        DistillationTrainer(model, tokenizer, _config(), _data(TRAIN), teacher=model)


def test_eval_data_must_not_share_prompts(tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="overlap"):
        DistillationTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            _config(),
            _data(TRAIN),
            _data(["HELLO"]),
            teacher=make_model(1, n_embd=16),
        )


# ----------------------------------------------------------- vocabularies


def _permuted_tokenizer(permutation: list[int]) -> Any:
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import PreTrainedTokenizerFast

    vocab = make_tokenizer().get_vocab()
    backend = Tokenizer(
        WordLevel({t: permutation[i] for t, i in vocab.items()}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Split(pattern="", behavior="isolated")
    backend.decoder = Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _permuted_teacher(student: Any, permutation: list[int]) -> Any:
    """The student with token ids renumbered: embedding row ``π(i)`` holds the
    student's row ``i`` (GPT-2 ties the LM head to it)."""
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        weight = student.transformer.wte.weight.detach().clone()
        teacher.transformer.wte.weight[torch.tensor(permutation)] = weight
    return teacher


def test_strict_alignment_rejects_different_vocabularies(tokenizer: Any) -> None:
    permutation = list(reversed(range(VOCAB_SIZE)))
    student = make_model(0, n_embd=16)
    with pytest.raises(VocabularyMismatchError, match="shared"):
        DistillationTrainer(
            student,
            tokenizer,
            _config(),
            _data(TRAIN),
            teacher=_permuted_teacher(student, permutation),
            teacher_tokenizer=_permuted_tokenizer(permutation),
        )


def test_shared_alignment_translates_ids_and_columns(tokenizer: Any) -> None:
    permutation = list(range(VOCAB_SIZE))
    random.Random(0).shuffle(permutation)
    student = make_model(0, n_embd=16)
    trainer = DistillationTrainer(
        student,
        tokenizer,
        _config(vocab_alignment="shared"),
        _data(TRAIN),
        teacher=_permuted_teacher(student, permutation),
        teacher_tokenizer=_permuted_tokenizer(permutation),
    )
    assert trainer.data_stats["vocab"]["shared_tokens"] == VOCAB_SIZE
    # The renumbered copy is the same distribution: zero divergence.
    report = trainer.divergence_report([{"role": "user", "content": "hello"}], "12ab")
    assert report["forward_kl"] < 1e-9 and report["reverse_kl"] < 1e-9


def test_padded_teacher_vocab_needs_shared_alignment(tokenizer: Any) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(3)
    padded = GPT2LMHeadModel(
        GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=VOCAB_SIZE + 7)
    )
    with pytest.raises(VocabularyMismatchError, match="padding"):
        _trainer(tokenizer, teacher=padded)
    trainer = _trainer(tokenizer, teacher=padded, vocab_alignment="shared")
    assert trainer.data_stats["vocab"]["teacher_only"] == 7
    assert trainer.train().global_step == 2


def test_shared_alignment_suppresses_student_only_tokens(tokenizer: Any) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(4)
    student = GPT2LMHeadModel(
        GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=VOCAB_SIZE + 3)
    )
    with torch.no_grad():  # make padding row VOCAB_SIZE + 1 the only likely token
        student.transformer.ln_f.weight.zero_()
        student.transformer.ln_f.bias.fill_(1.0)
        student.transformer.wte.weight[VOCAB_SIZE + 1] = 50.0
    trainer = DistillationTrainer(
        student,
        tokenizer,
        _config(vocab_alignment="shared", lmbda=1.0),
        _data(TRAIN),
        teacher=make_model(5, n_embd=16),
    )
    assert trainer._suppressed == [VOCAB_SIZE, VOCAB_SIZE + 1, VOCAB_SIZE + 2]
    prompts = [trainer._train_items[0].prompt_ids]
    common = {
        "num_samples": 4,
        "max_new_tokens": 3,
        "temperature": 1.0,
        "top_p": 1.0,
        "pad_token_id": 0,
        "stop_ids": [1],
        "batch_size": 4,
    }
    unsuppressed = sample_completions(student, prompts, **common)
    assert all(set(c) == {VOCAB_SIZE + 1} for c in unsuppressed[0])
    for completion in trainer._sample(trainer._train_items)[:4]:
        assert max(completion.response_ids) < VOCAB_SIZE


# -------------------------------------------------------------- on-policy


def test_on_policy_samples_are_exact_student_tokens(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, lmbda=1.0)
    torch.manual_seed(0)
    samples, batches = trainer._prepare([[0, 1, 2, 3]])
    assert batches == [[0, 1, 2, 3]] and all(s.on_policy for s in samples)
    for sample, item in zip(samples, trainer._train_items, strict=True):
        assert sample.sequence.prompt_ids == item.prompt_ids
        assert 1 <= len(sample.sequence.response_ids) <= 4
        eos = tokenizer.eos_token_id
        if eos in sample.sequence.response_ids:
            assert sample.sequence.response_ids[-1] == eos
    assert trainer.train().history[-1]["on_policy_fraction"] == 1.0


def test_lambda_draws_are_deterministic_and_calibrated(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, lmbda=0.3)
    draws = []
    for attempt in range(400):
        trainer.update_attempts = attempt
        draws.append(trainer._draw_on_policy(0))
    assert 0.22 < sum(draws) / len(draws) < 0.38
    trainer.update_attempts = 17
    assert trainer._draw_on_policy(1) == trainer._draw_on_policy(1)
    assert not _trainer(tokenizer)._draw_on_policy(0)


def test_sampled_reverse_kl_objective_trains(tokenizer: Any) -> None:
    trainer = _trainer(
        tokenizer, lmbda=1.0, divergence="reverse_kl", on_policy_loss="sampled"
    )
    summary = trainer.train()
    assert "kl_estimate" in summary.history[-1] and summary.global_step == 2


def test_prompt_only_data_trains_fully_on_policy(tokenizer: Any) -> None:
    trainer = DistillationTrainer(
        make_model(0, n_embd=16),
        tokenizer,
        _config(lmbda=1.0),
        _data(TRAIN, completions=False),
        _data(HELD, completions=False),
        teacher=make_model(50, n_embd=16),
    )
    summary = trainer.train()
    assert "eval/on_policy/forward_kl" in summary.eval_metrics
    with pytest.raises(ValueError, match="completions"):
        trainer.evaluate()


# ----------------------------------------------------------- teacher cache


def test_full_vocabulary_cache_matches_the_live_teacher(tokenizer: Any) -> None:
    live = _trainer(tokenizer, max_steps=1, **LINEAR_ADAM)
    cached = _trainer(tokenizer, max_steps=1, teacher_top_k=VOCAB_SIZE, **LINEAR_ADAM)
    assert cached.teacher is None and len(cached.teacher_cache) == len(TRAIN + HELD)
    live_eval, cached_eval = live.evaluate(), cached.evaluate()
    assert cached_eval["eval/forward_kl"] == pytest.approx(
        live_eval["eval/forward_kl"], rel=1e-4
    )
    live.train()
    cached.train()
    assert _max_diff(_params(live.model), _params(cached.model)) < 1e-5


def test_cache_round_trips_and_trains_without_a_teacher(
    tokenizer: Any, tmp_path: Path
) -> None:
    config = _config(teacher_top_k=5)
    data = _data(TRAIN)
    cache = precompute_teacher_cache(
        make_model(50, n_embd=16), tokenizer, data, config, student=make_model(0)
    )
    path = cache.save(tmp_path / "cache.pt")
    loaded = TeacherCache.load(path)
    assert loaded.top_k == 5 and set(loaded.entries) == set(cache.entries)
    for key, (ids, logps) in cache.entries.items():
        assert torch.equal(ids, loaded.entries[key][0])
        assert torch.equal(logps, loaded.entries[key][1])
    trainer = DistillationTrainer(
        make_model(0, n_embd=16), tokenizer, config, data, teacher_cache=loaded
    )
    assert trainer.teacher is None and trainer.train().global_step == 2
    with pytest.raises(ValueError, match="live teacher"):
        trainer.evaluate(_data(HELD), on_policy=True)


def test_cache_mismatches_are_rejected(tokenizer: Any) -> None:
    config = _config(teacher_top_k=5)
    cache = precompute_teacher_cache(
        make_model(50, n_embd=16), tokenizer, _data(TRAIN), config
    )
    with pytest.raises(ValueError, match="temperature"):
        DistillationTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            _config(teacher_top_k=5, temperature=2.0),
            _data(TRAIN),
            teacher_cache=cache,
        )
    with pytest.raises(ValueError, match="no entry"):
        DistillationTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            config,
            _data(["a fresh prompt"]),
            teacher_cache=cache,
        )
    with pytest.raises(ValueError, match="teacher_top_k"):
        DistillationTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            _config(),
            _data(TRAIN),
            teacher=make_model(50, n_embd=16),
            teacher_cache=cache,
        )


# ----------------------------------------------------- robustness, resume


def test_nonfinite_loss_raises_or_skips(
    tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = trainer_module.divergence_from_logprobs

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs) * math.nan

    monkeypatch.setattr(trainer_module, "divergence_from_logprobs", poisoned)
    with pytest.raises(ValueError, match="non-finite"):
        _trainer(tokenizer).train()
    skipping = _trainer(tokenizer, nonfinite_policy="skip")
    before = _params(skipping.model)
    monkeypatch.setattr(skipping, "_final_eval", lambda: None)
    summary = skipping.train()
    assert (summary.global_step, summary.skipped_steps) == (0, 2)
    assert _max_diff(before, _params(skipping.model)) == 0.0


def test_checkpoint_resume_is_exact_with_on_policy_sampling(
    tokenizer: Any, tmp_path: Path
) -> None:
    settings = {"max_steps": 4, "batch_size": 2, "lr_scheduler": "linear", "lmbda": 0.5}
    straight = _trainer(tokenizer, **settings)
    straight.train()
    interrupted = _trainer(tokenizer, **settings)
    interrupted.config.max_steps = 2
    interrupted.train()
    checkpoint = interrupted.save_checkpoint(tmp_path / "ckpt")
    meta = json.loads((checkpoint / "distillation_checkpoint.json").read_text())
    assert meta["schema"] == "stateset-distillation-checkpoint-v1"
    assert meta["global_step"] == 2
    resumed = _trainer(tokenizer, **settings)
    summary = resumed.train(resume_from=checkpoint)
    assert summary.global_step == 4
    assert _max_diff(_params(straight.model), _params(resumed.model)) == 0.0
    assert straight.history[-1] == resumed.history[-1]
    with pytest.raises(ValueError, match="divergence"):
        _trainer(tokenizer, divergence="reverse_kl").load_checkpoint(checkpoint)


def test_lora_student_trains_adapters_only(tokenizer: Any) -> None:
    pytest.importorskip("peft")
    trainer = _trainer(tokenizer, use_lora=True, lora_target_modules=["c_attn"])
    assert hasattr(trainer.model, "peft_config")
    frozen = [
        p.detach().clone() for p in trainer.model.parameters() if not p.requires_grad
    ]
    trainer.train()
    after = [p.detach() for p in trainer.model.parameters() if not p.requires_grad]
    assert _max_diff(frozen, after) == 0.0


def test_train_distillation_api_writes_artifacts(
    tokenizer: Any, tmp_path: Path
) -> None:
    rows = [
        {"prompt": [{"role": "user", "content": f"question {i}"}], "completion": "12"}
        for i in range(8)
    ]
    result = train_distillation(
        make_model(0, n_embd=16),
        rows,
        teacher=make_model(50, n_embd=16),
        tokenizer=tokenizer,
        config=_config(),
        validation_fraction=0.25,
        output_dir=str(tmp_path / "out"),
    )
    assert result.checkpoint == tmp_path / "out" / "final"
    assert (result.checkpoint / "distillation_checkpoint.json").exists()
    metrics = json.loads((tmp_path / "out" / "distillation_metrics.json").read_text())
    assert metrics["method"] == "distillation" and metrics["eval_metrics"]
    assert metrics["data"]["train_examples"] + metrics["data"]["eval_examples"] == 8
