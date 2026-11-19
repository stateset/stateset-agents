"""Independent checks for the client side of remote importance-weighted updates."""

import itertools
import json
import math
from types import SimpleNamespace

import pytest

from stateset_agents.remote.river_batches import build_group_rl_datums, build_rl_batch
from stateset_agents.remote.river_rl import normalize_datums
from stateset_agents.remote.tinker import TinkerExecutor


class _Characters:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def test_river_truncated_fields_match_next_retained_token_at_every_cutoff():
    tok = _Characters()
    for p, c in itertools.product(range(1, 5), repeat=2):
        prompt = "pqrs"[:p]
        completion = "abcd"[:c]
        scores = [-(j + 1) / 8 for j in range(c)]
        credit = [(-1) ** j * (j + 1) for j in range(c)]
        trajectory = SimpleNamespace(prompt=prompt, response=completion)
        for limit in range(1, p + c + 2):
            batch = build_rl_batch(
                [trajectory], tok, [scores], [credit], max_length=limit
            )
            if limit <= p:
                assert batch == []
                continue
            (row,) = batch
            ids = tok.encode((prompt + completion)[:limit])
            expected = [(0.0, 0.0)] * len(ids)
            for i, target in enumerate(ids[1:]):
                if chr(target) in completion:
                    j = completion.index(chr(target))
                    expected[i] = (scores[j], credit[j])
            assert row["input_ids"] == ids
            assert (
                list(zip(row["old_logprobs"], row["advantages"], strict=True))
                == expected
            )
            norm, count = normalize_datums([row], "token")
            assert count == max(0, len(ids) - p)
            assert len(norm) == int(count > 0)


@pytest.mark.parametrize("mode", ["sum", "token", "sequence"])
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo"])
def test_normalized_wire_credits_reproduce_the_logical_gradient(mode, loss_fn):
    torch = pytest.importorskip("torch")
    rows = [{"advantages": a} for a in ([0, 2, 0], [0, 0], [-3, 0, 4, 2, 0])]
    wire, count = normalize_datums(rows, mode)
    assert count == 4
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    # Distinct score directions produce ratios below, within, and above the PPO band.
    directions = [[0, 2, 0], [-2, 0, 0.25, -3, 0]]
    wire_objective = 0.0
    expected_gradient = 0.0
    for source, datum, direction in zip(
        (rows[0], rows[2]), wire, directions, strict=True
    ):
        active = sum(a != 0 for a in source["advantages"])
        denominator = {"sum": 1, "token": 4, "sequence": 2 * active}[mode]
        direction_t = torch.tensor(direction, dtype=x.dtype)
        logp = x * direction_t - 5
        ratio = (logp + 5).exp()
        adv = torch.tensor(datum["advantages"], dtype=x.dtype)
        if loss_fn == "ppo":
            term = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
        elif loss_fn == "cispo":
            term = -ratio.clamp(max=1.2).detach() * adv * logp
        else:
            term = -ratio * adv
        wire_objective += term.sum()
        for a, d in zip(source["advantages"], direction, strict=True):
            r = math.exp(0.5 * d)
            coefficient = r
            if loss_fn == "ppo" and ((a > 0 and r > 1.2) or (a < 0 and r < 0.8)):
                coefficient = 0
            elif loss_fn == "cispo":
                coefficient = min(1.2, r)
            expected_gradient -= a * coefficient * d / denominator
    (actual,) = torch.autograd.grad(wire_objective, x)
    assert float(actual) == pytest.approx(expected_gradient, abs=1e-12)


def test_large_finite_group_rewards_center_before_transport():
    samples = [{"tokens": [11], "logprobs": [-0.2]}] * 2
    rows = build_group_rl_datums([3], samples, [1e308, 8e307])
    assert [row["advantages"][0] for row in rows] == pytest.approx([1e307, -1e307])
    with pytest.raises(ValueError, match="advantages must be finite"):
        build_group_rl_datums([3], samples * 2, [1.7e308, -1.7e308, -1.7e308, -1.7e308])


def _tinker_row():
    return {
        "input_ids": [7, 8, 9],
        "target_tokens": [8, 9, 10],
        "logprobs": [0, -1, -2],
        "advantages": [0, 2, -3],
        "sample_id": 7,
    }


def test_tinker_sends_exact_fields_and_dtypes_to_remote_autograd():
    torch = pytest.importorskip("torch")
    module = SimpleNamespace(
        ModelInput=SimpleNamespace(from_ints=lambda values: values),
        TensorData=SimpleNamespace(from_torch=lambda tensor: tensor),
        Datum=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    (row,) = TinkerExecutor._validate_rl_rows([_tinker_row()])
    datum = TinkerExecutor._datum(module, row)
    assert datum.model_input == [7, 8, 9]
    fields = datum.loss_fn_inputs
    assert set(fields) == {"target_tokens", "logprobs", "advantages"}
    assert fields["target_tokens"].dtype == torch.int64
    for key in ("logprobs", "advantages"):
        assert fields[key].dtype == torch.float32
        assert fields[key].tolist() == row[key]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("advantages", float("nan")),
        ("advantages", 1e200),
        ("logprobs", 0.5),
        ("logprobs", float("-inf")),
        ("target_tokens", 1.5),
        ("input_ids", -1),
        ("input_ids", 2**63),
        ("input_ids", True),
    ],
)
def test_tinker_rejects_invalid_policy_data(field, bad):
    row = _tinker_row()
    row[field][1] = bad
    with pytest.raises(ValueError, match=field):
        TinkerExecutor._validate_rl_rows([row])


def test_tinker_rejects_ambiguous_sft_mask_before_training():
    with pytest.raises(ValueError, match="SFT weights"):
        TinkerExecutor._validate_rl_rows([{**_tinker_row(), "weights": [1, 0, 1]}])


def test_real_tinker_sdk_keeps_next_tokens_and_signed_float_advantages():
    tinker = pytest.importorskip("tinker")
    torch = pytest.importorskip("torch")
    (row,) = TinkerExecutor._validate_rl_rows([_tinker_row()])
    datum = TinkerExecutor._datum(tinker, row)
    assert datum.model_input.to_ints() == row["input_ids"]
    for key, value in datum.loss_fn_inputs.items():
        t = value.to_torch()
        assert t.dtype == (torch.int64 if key == "target_tokens" else torch.float32)
        assert t.tolist() == row[key]


def test_real_river_sdk_and_driver_have_distinct_zero_credit_populations():
    river = pytest.importorskip("river_client")
    from river_client.rl import GroupCentered, Truncation
    from river_client.rl.advantages import build_batch
    from river_client.rl.trajectory import Trajectory

    rewards = [-1.0, 0.0, 1.0]
    lengths = [1, 3, 2]
    members, samples = [], []
    for reward, n in zip(rewards, lengths, strict=True):
        tokens = list(range(20, 20 + n))
        scores = [-0.25] * n
        traj = Trajectory([{"type": "text", "tokens": [3, 4]}])
        sample = river.Sample(
            tokens=tokens,
            text="x",
            logprobs=scores,
            stop_reason="stop",
            model_step=0,
            token_data_is_exact=True,
        )
        traj.append_span(sample, policy_step=0)
        traj.done, traj.reward = True, reward
        members.append(traj)
        samples.append({"tokens": tokens, "logprobs": scores})
    native, _ = build_batch(
        [members],
        estimator=GroupCentered(),
        normalize="token",
        truncation=Truncation(),
        min_members=2,
        current_step=0,
        max_staleness=0,
    )
    driver, count = normalize_datums(
        build_group_rl_datums([3, 4], samples, rewards), "token"
    )
    assert count == 3
    assert len(native) == len(driver) == 2
    for sdk, local in zip(native, driver, strict=True):
        assert sdk["input_ids"] == local["input_ids"]
        assert sdk["old_logprobs"] == local["old_logprobs"]
        assert sdk["advantages"] == pytest.approx([a / 2 for a in local["advantages"]])


def test_real_river_sdk_and_episode_driver_sequence_mean():
    river = pytest.importorskip("river_client")
    from river_client.rl import GroupCentered, Truncation
    from river_client.rl.advantages import build_batch
    from river_client.rl.trajectory import Trajectory

    from stateset_agents.remote.river import _episode_rl_datums
    from stateset_agents.remote.river_rl import sample_record

    traj = Trajectory([{"type": "text", "tokens": [3, 4]}])
    records = []
    prefix = [3, 4]
    for i, tokens in enumerate(([5], [8, 9, 10])):
        if i:
            traj.append_framing([{"type": "text", "tokens": [7]}])
            prefix = [*prefix, 7]
        sample = river.Sample(
            tokens=tokens,
            text="x",
            logprobs=[-0.5] * len(tokens),
            stop_reason="stop",
            model_step=0,
        )
        records.append(sample_record(sample, prefix))
        traj.append_span(sample, policy_step=0)
        prefix = [*prefix, *tokens]
    traj.done, traj.reward = True, 1.0
    native, _ = build_batch(
        [[traj]],
        estimator=GroupCentered(),
        advantages=[[1.0]],
        normalize="sequence",
        truncation=Truncation(),
        min_members=1,
        current_step=0,
        max_staleness=0,
    )
    local, count = normalize_datums(_episode_rl_datums(records, 1.0), "sequence")

    def coefficients(rows):
        return {
            token: a
            for row in rows
            for token, a in zip(
                row["input_ids"][1:], row["advantages"][:-1], strict=True
            )
            if a
        }

    assert count == 4
    assert coefficients(native) == dict.fromkeys([5, 8, 9, 10], 0.25)
    assert coefficients(local) == pytest.approx({5: 0.5, 8: 1 / 6, 9: 1 / 6, 10: 1 / 6})


def test_token_ratio_corrects_conditional_action_but_leaves_prefix_measure():
    torch = pytest.importorskip("torch")
    z = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    behavior_first = (0.5, 0.5)
    target_first = (0.25, 0.75)
    second = torch.stack((1 - z.sigmoid(), z.sigmoid()))
    conditional, path, target = 0.0, 0.0, 0.0
    for i, j in itertools.product(range(2), repeat=2):
        reward = float(i == j == 1)
        behavior = behavior_first[i] * 0.5
        token_ratio = second[j] / 0.5
        conditional += behavior * token_ratio * reward
        path += behavior * (target_first[i] / behavior_first[i]) * token_ratio * reward
        target += target_first[i] * second[j] * reward
    assert float(conditional.detach()) == 0.25
    torch.testing.assert_close(path, target)
    grads = [
        torch.autograd.grad(value, z, retain_graph=True)[0].item()
        for value in (conditional, path, target)
    ]
    assert grads == [0.125, 0.1875, 0.1875]


def test_real_river_sdk_decoupled_ppo_keeps_signed_capped_behavior_correction():
    np = pytest.importorskip("numpy")
    pytest.importorskip("river_client")
    from river_client.rl.advantages import decouple_ppo

    data = [
        {"old_logprobs": [0.0, -3.0, -1.0, 0.0], "advantages": [0.0, -2.0, 3.0, 0.0]}
    ]
    prox = [0.0, -1.0, -2.0]
    (row,), _ = decouple_ppo(data, [prox], max_importance_weight=2.0)
    assert row["advantages"] == pytest.approx([0.0, -4.0, 3 / math.e, 0.0])
    assert row["old_logprobs"] == prox + [0.0]
    for cur in ([-0.1, -0.5, -1.0], [0.0, -2.0, -4.0]):
        score = np.array(cur[1:])
        r = np.exp(score - np.array(prox[1:]))
        adjusted = np.array(row["advantages"][1:3])
        old_adv = np.array(data[0]["advantages"][1:3])
        loss = -np.minimum(r * adjusted, r.clip(0.8, 1.2) * adjusted)
        oracle = -np.minimum(r * old_adv, r.clip(0.8, 1.2) * old_adv) * [
            2.0,
            1 / math.e,
        ]
        np.testing.assert_allclose(loss, oracle)


def test_tinker_executor_accumulates_the_documented_is_sum_through_partial_window(
    tmp_path,
):
    tinker = pytest.importorskip("tinker")
    torch = pytest.importorskip("torch")
    from stateset_agents.remote.job import RemoteJobSpec

    rows = [{**_tinker_row(), "advantages": [0, i, -2 * i]} for i in (1, 2, 3)]
    source = tmp_path / "rl.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))

    class Transport:
        def __init__(self):
            self.grad = 0.0
            self.updates = []

        def forward_backward(self, batch, loss_fn):
            assert loss_fn == "importance_sampling"
            theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            for datum in batch:
                fields = datum.loss_fn_inputs
                direction = fields["target_tokens"].to_torch().double()
                learner = -2 + theta * direction
                behavior = fields["logprobs"].to_torch().double()
                credit = fields["advantages"].to_torch().double()
                loss = -((learner - behavior).exp() * credit).sum()
                self.grad += torch.autograd.grad(loss, theta)[0].item()

        def optim_step(self, params):
            self.updates.append(self.grad)
            self.grad = 0.0

        def save_weights_for_sampler(self, name):
            return {"path": "local-sampler"}

        def save_state(self, name):
            return {"path": "local-state"}

    sink = Transport()
    service = SimpleNamespace(create_lora_training_client=lambda **kwargs: sink)
    executor = TinkerExecutor(service, tinker_module=tinker)
    handle = executor.submit(
        RemoteJobSpec(
            job_kind="rl",
            dataset=source,
            base_model="test/RL",
            output_dir=tmp_path / "out",
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            num_epochs=1,
        )
    )
    result = executor.wait(handle, poll_interval_s=0.001)
    assert result.succeeded, list(executor.logs(handle))
    # Independent signed derivative per row. Both logical windows have total i=3.
    expected = 3 * (20 - 9 * math.exp(-1))
    assert sink.updates == pytest.approx([expected, expected])


def test_tinker_rejects_invalid_later_row_before_allocating_model(tmp_path):
    from stateset_agents.remote.job import JobStatus, RemoteJobSpec

    rows = [_tinker_row(), {**_tinker_row(), "logprobs": [0, 1, -2]}]
    source = tmp_path / "rl.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    allocations = []
    service = SimpleNamespace(
        create_lora_training_client=lambda **kwargs: allocations.append(kwargs)
    )
    executor = TinkerExecutor(service, tinker_module=SimpleNamespace())
    handle = executor.submit(
        RemoteJobSpec(
            job_kind="rl", dataset=source, base_model="test/RL", output_dir=tmp_path
        )
    )
    result = executor.wait(handle, poll_interval_s=0.001)
    assert result.status is JobStatus.FAILED
    assert "RL row 1 invalid logprobs" in " ".join(result.logs)
    assert allocations == []
