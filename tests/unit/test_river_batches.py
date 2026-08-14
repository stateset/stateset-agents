"""Tests for River batch construction — the part that must be right.

River is not live-verifiable here (no API key, ``river-client`` not
installable), so these tests pin down the *shape* we commit to: prompt tokens
excluded from the loss, completion tokens included, the causal shift applied
exactly once, and bad rows skipped rather than fatal.
"""

from __future__ import annotations

import logging

import pytest

from stateset_agents.core.trajectory import Trajectory
from stateset_agents.remote.river_batches import (
    DOCUMENTED_BASE_MODELS,
    build_rl_batch,
    build_sft_batch,
    validate_base_model,
    validate_lora_rank,
)


class FakeTokenizer:
    """Whitespace tokenizer with a deterministic chat template.

    Deliberately simple and prefix-stable, so a weight-mask assertion is a
    statement about ``build_sft_batch`` rather than about BPE.
    """

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_tok: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._to_id:
            new = len(self._to_id) + 100
            self._to_id[token] = new
            self._to_tok[new] = token
        return self._to_id[token]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        assert tokenize is False
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|> {message['content']} <|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def encode(self, text, add_special_tokens=True):
        return [self._id(tok) for tok in text.split()]

    def decode(self, ids):
        return " ".join(self._to_tok[i] for i in ids)


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def _row(*pairs):
    messages = []
    for role, content in pairs:
        messages.append({"role": role, "content": content})
    return {"messages": messages}


class TestSftBatchShape:
    def test_emits_the_three_documented_keys(self, tokenizer):
        (datum,) = build_sft_batch(
            [_row(("user", "hi"), ("assistant", "hello"))], tokenizer
        )
        assert set(datum) == {"input_ids", "target_tokens", "weights"}

    def test_all_three_lists_are_index_aligned(self, tokenizer):
        (datum,) = build_sft_batch(
            [_row(("user", "hi"), ("assistant", "hello"))], tokenizer
        )
        assert len(datum["input_ids"]) == len(datum["target_tokens"])
        assert len(datum["input_ids"]) == len(datum["weights"])

    def test_targets_are_inputs_shifted_by_one(self, tokenizer):
        """The single most consequential assumption in the integration."""
        (datum,) = build_sft_batch(
            [_row(("user", "hi"), ("assistant", "hello"))], tokenizer
        )
        # Reconstruct the unshifted sequence: inputs + the final target.
        full = datum["input_ids"] + [datum["target_tokens"][-1]]
        assert datum["target_tokens"] == full[1:]

    def test_prompt_tokens_have_weight_zero(self, tokenizer):
        (datum,) = build_sft_batch(
            [_row(("user", "hi"), ("assistant", "hello"))], tokenizer
        )
        zero = [
            t
            for t, w in zip(datum["target_tokens"], datum["weights"], strict=True)
            if w == 0.0
        ]
        assert "hi" in tokenizer.decode(zero)
        assert "hello" not in tokenizer.decode(zero)

    def test_completion_tokens_have_weight_one(self, tokenizer):
        (datum,) = build_sft_batch(
            [_row(("user", "hi"), ("assistant", "hello"))], tokenizer
        )
        assert set(datum["weights"]) <= {0.0, 1.0}
        assert 1.0 in datum["weights"]

    def test_decoding_the_weighted_tokens_recovers_the_assistant_text(self, tokenizer):
        """Round trip: what we train on is exactly what we wanted to say."""
        (datum,) = build_sft_batch(
            [_row(("user", "capital of france?"), ("assistant", "paris"))], tokenizer
        )
        weighted = [
            t
            for t, w in zip(datum["target_tokens"], datum["weights"], strict=True)
            if w == 1.0
        ]
        decoded = tokenizer.decode(weighted)
        assert "paris" in decoded
        assert "capital" not in decoded


class TestMultiTurn:
    def test_every_assistant_turn_is_weighted(self, tokenizer):
        (datum,) = build_sft_batch(
            [
                _row(
                    ("system", "be terse"),
                    ("user", "one"),
                    ("assistant", "first"),
                    ("user", "two"),
                    ("assistant", "second"),
                )
            ],
            tokenizer,
        )
        weighted = tokenizer.decode(
            [
                t
                for t, w in zip(datum["target_tokens"], datum["weights"], strict=True)
                if w == 1.0
            ]
        )
        assert "first" in weighted
        assert "second" in weighted

    def test_intermediate_user_turns_stay_unweighted(self, tokenizer):
        (datum,) = build_sft_batch(
            [
                _row(
                    ("user", "one"),
                    ("assistant", "first"),
                    ("user", "two"),
                    ("assistant", "second"),
                )
            ],
            tokenizer,
        )
        weighted = tokenizer.decode(
            [
                t
                for t, w in zip(datum["target_tokens"], datum["weights"], strict=True)
                if w == 1.0
            ]
        )
        assert "two" not in weighted

    def test_trailing_user_turn_after_last_assistant_is_dropped(self, tokenizer):
        (datum,) = build_sft_batch(
            [_row(("user", "one"), ("assistant", "first"), ("user", "dangling"))],
            tokenizer,
        )
        assert "dangling" not in tokenizer.decode(datum["target_tokens"])


class TestTruncation:
    def test_truncates_to_max_length(self, tokenizer):
        row = _row(("user", "q"), ("assistant", "b " * 200))
        (datum,) = build_sft_batch([row], tokenizer, max_length=64)
        # max_length applies before the shift, which drops one position.
        assert len(datum["input_ids"]) == 63

    def test_row_truncated_before_any_completion_token_is_dropped(self, tokenizer):
        row = _row(("user", "x " * 500), ("assistant", "answer"))
        assert build_sft_batch([row], tokenizer, max_length=10) == []

    def test_rejects_degenerate_max_length(self, tokenizer):
        with pytest.raises(ValueError, match="max_length"):
            build_sft_batch([], tokenizer, max_length=1)


class TestBadRows:
    @pytest.mark.parametrize(
        "row",
        [
            {},
            {"messages": []},
            {"messages": "not a list"},
            {"messages": [{"role": "user"}]},
            {"messages": [{"role": "user", "content": "only a question"}]},
            {"messages": [{"role": "assistant", "content": ""}]},
            "not a dict",
            None,
        ],
    )
    def test_unusable_rows_are_skipped_not_fatal(self, tokenizer, row):
        assert build_sft_batch([row], tokenizer) == []

    def test_good_rows_survive_a_bad_neighbour(self, tokenizer):
        rows = [{}, _row(("user", "hi"), ("assistant", "hello")), {"messages": []}]
        assert len(build_sft_batch(rows, tokenizer)) == 1

    def test_skipping_is_logged(self, tokenizer, caplog):
        with caplog.at_level(logging.WARNING):
            build_sft_batch([{}], tokenizer)
        assert "skipping row 0" in caplog.text


class TestValidateLoraRank:
    @pytest.mark.parametrize("rank", [1, 8, 16, 32])
    def test_accepts_the_documented_window(self, rank):
        assert validate_lora_rank(rank) == rank

    @pytest.mark.parametrize("rank", [0, -1, 33, 64])
    def test_rejects_outside_it_naming_rivers_limit(self, rank):
        with pytest.raises(ValueError, match="1-32"):
            validate_lora_rank(rank)

    def test_error_names_river(self):
        with pytest.raises(ValueError, match="River"):
            validate_lora_rank(33)

    def test_rejects_non_integers(self):
        with pytest.raises(ValueError, match="int"):
            validate_lora_rank(16.0)  # type: ignore[arg-type]


class TestValidateBaseModel:
    @pytest.mark.parametrize("name", DOCUMENTED_BASE_MODELS)
    def test_documented_models_pass_silently(self, name, caplog):
        with caplog.at_level(logging.WARNING):
            assert validate_base_model(name) == name
        assert caplog.text == ""

    def test_unknown_model_warns_but_proceeds(self, caplog):
        """Account-scoped entitlements mean we cannot honestly refuse."""
        with caplog.at_level(logging.WARNING):
            assert validate_base_model("acme/secret-model") == "acme/secret-model"
        assert "acme/secret-model" in caplog.text

    def test_explicit_allowlist_overrides_the_doc_list(self, caplog):
        with caplog.at_level(logging.WARNING):
            validate_base_model("acme/secret-model", allowed=["acme/secret-model"])
        assert caplog.text == ""

    def test_empty_name_is_an_error(self):
        with pytest.raises(ValueError):
            validate_base_model("   ")


class TestRlBatch:
    def _traj(self, prompt="ask me", response="an answer here", reward=1.0):
        return Trajectory(prompt=prompt, response=response, reward=reward)

    def test_emits_the_four_documented_keys(self, tokenizer):
        traj = self._traj()
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch([traj], tokenizer, old_logprobs=[lp])
        assert set(datum) == {
            "input_ids",
            "old_logprobs",
            "advantages",
            "attention_mask",
        }

    def test_every_list_matches_input_ids_length(self, tokenizer):
        traj = self._traj()
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch([traj], tokenizer, old_logprobs=[lp])
        n = len(datum["input_ids"])
        assert all(len(datum[k]) == n for k in datum)

    def test_prompt_positions_carry_zero_advantage(self, tokenizer):
        traj = self._traj()
        prompt_len = len(tokenizer.encode(traj.prompt))
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch([traj], tokenizer, old_logprobs=[lp])
        assert datum["advantages"][:prompt_len] == [0.0] * prompt_len
        assert all(a == 1.0 for a in datum["advantages"][prompt_len:])

    def test_reward_is_the_default_advantage(self, tokenizer):
        traj = self._traj(reward=0.25)
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch([traj], tokenizer, old_logprobs=[lp])
        assert 0.25 in datum["advantages"]

    def test_explicit_scalar_advantages_win(self, tokenizer):
        traj = self._traj(reward=0.25)
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch(
            [traj], tokenizer, old_logprobs=[lp], advantages=[-2.0]
        )
        assert -2.0 in datum["advantages"]
        assert 0.25 not in datum["advantages"]

    def test_per_token_advantages_are_accepted(self, tokenizer):
        traj = self._traj()
        n = len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch(
            [traj],
            tokenizer,
            old_logprobs=[[-0.5] * n],
            advantages=[[float(i) + 1 for i in range(n)]],
        )
        assert datum["advantages"][-1] == float(n)

    def test_attention_mask_is_all_ones(self, tokenizer):
        traj = self._traj()
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        (datum,) = build_rl_batch([traj], tokenizer, old_logprobs=[lp])
        assert set(datum["attention_mask"]) == {1}

    def test_misaligned_logprobs_are_a_loud_error(self, tokenizer):
        """Silently misaligned logprobs would train in the wrong direction."""
        with pytest.raises(ValueError, match="old_logprobs"):
            build_rl_batch([self._traj()], tokenizer, old_logprobs=[[-0.5]])

    def test_logprob_count_mismatch_across_trajectories_is_an_error(self, tokenizer):
        with pytest.raises(ValueError, match="one per trajectory"):
            build_rl_batch([self._traj(), self._traj()], tokenizer, old_logprobs=[[]])

    def test_advantage_count_mismatch_is_an_error(self, tokenizer):
        traj = self._traj()
        lp = [-0.5] * len(tokenizer.encode(traj.response))
        with pytest.raises(ValueError, match="advantages has"):
            build_rl_batch([traj], tokenizer, old_logprobs=[lp], advantages=[1.0, 2.0])

    def test_empty_response_is_skipped(self, tokenizer):
        assert (
            build_rl_batch([self._traj(response="")], tokenizer, old_logprobs=[[]])
            == []
        )
