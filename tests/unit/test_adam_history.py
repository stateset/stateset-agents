"""AdamW parameter-local clocks, working precision, and trainer checkpoint replay."""

import copy
import itertools
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import ppo_trainer as ppo  # noqa: E402
from stateset_agents.training.adamw_weights import ModelAdamW, model_adamw  # noqa: E402
from tests.unit.test_ppo_safety import (  # noqa: E402
    CharacterTokenizer,
    TinyPolicy,
    assert_same,
    snapshot,
)
from tests.unit.test_runtime_checkpoint_safety import republish  # noqa: E402


def closed_update(history, initial, rate, decay, beta1, beta2, eps):
    """Rebuild the normalized moments from local history at high precision."""
    with localcontext() as ctx:
        ctx.prec = 90
        ages = range(len(history))
        # Newest gradient at age zero. Normalize by the sum of geometric weights.
        m = sum(beta1**i * g for i, g in zip(ages, reversed(history), strict=True))
        m /= sum(beta1**i for i in ages)
        v = sum(beta2**i * g**2 for i, g in zip(ages, reversed(history), strict=True))
        v /= sum(beta2**i for i in ages)
        return (1 - rate * decay) * initial - rate * m / (v.sqrt() + eps)


def test_all_tlc_histories_agree_with_decimal_and_quantized_working_grid():
    half = Decimal("0.5")
    for table in itertools.product((None, -1, 0, 1), repeat=3):
        weight = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
        opt = model_adamw([weight], lr=0.25, weight_decay=0.5, eps=1, betas=(0.5, 0.5))
        history = []
        working = 8
        for item in table:
            weight.grad = None if item is None else weight.new_tensor(item)
            opt.step()
            if item is not None:
                history.append(Decimal(item))
                exact = closed_update(
                    history,
                    Decimal(working) / 8,
                    Decimal("0.25"),
                    half,
                    half,
                    half,
                    Decimal(1),
                )
                assert weight.item() == pytest.approx(
                    float(exact), rel=3e-15, abs=1e-16
                )
                assert opt.state[weight]["step"].item() == len(history)
                working = round(exact * 8)
            else:
                assert weight.item() == working / 8
                assert (opt.state.get(weight, {}).get("step", 0)) == len(history)
            assert round(weight.item() * 8) == working
            # Start the next transition at the finite model's working weight.
            with torch.no_grad():
                weight.fill_(working / 8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_working_trajectory_tracks_local_history_and_external_edits(dtype):
    p = torch.nn.Parameter(torch.ones((), dtype=dtype))
    q = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    opt = model_adamw([p, q], lr=0.0002, betas=(0.75, 0.5), weight_decay=0.125)
    master = opt.masters[p]
    history = []
    expected = Decimal(1)
    for i, raw in enumerate((0, 1e-5, 0.5, None, -0.25, 0, None, 0.5, -0.5)):
        if i == 4:
            with torch.no_grad():
                p.fill_(0.75)
            expected = Decimal("0.75")
        opt.zero_grad(set_to_none=i % 2 == 0)
        q.grad = torch.tensor(0.125)
        p.grad = None if raw is None else p.new_tensor(raw)
        opt.step()
        assert opt.state[q]["step"].item() == i + 1
        if raw is not None:
            history.append(Decimal.from_float(p.grad.item()))
            expected = closed_update(
                history,
                expected,
                Decimal("0.0002"),
                Decimal("0.125"),
                Decimal("0.75"),
                Decimal("0.5"),
                Decimal("1e-8"),
            )
        state = opt.state[master]
        assert state["step"].item() == len(history)
        assert state["master_weight"].item() == pytest.approx(float(expected), abs=2e-7)
        assert p.item() == p.new_tensor(float(expected)).item()
        assert state["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("legacy", [False, True])
async def test_ppo_low_precision_runtime_checkpoint_continues_both_heads(
    monkeypatch, tmp_path, dtype, legacy
):
    monkeypatch.setattr(ppo, "_load_transformers_ppo", lambda: True)
    monkeypatch.setattr(ppo, "get_cosine_schedule_with_warmup", None)

    def build():
        torch.manual_seed(17)
        model = TinyPolicy().to(dtype)
        tokenizer = CharacterTokenizer()
        monkeypatch.setattr(
            model,
            "save_pretrained",
            lambda path: torch.save(
                model.state_dict(), Path(path) / "pytorch_model.bin"
            ),
            raising=False,
        )
        monkeypatch.setattr(tokenizer, "save_pretrained", lambda _: None, raising=False)
        trainer = ppo.PPOTrainer(
            config=ppo.PPOConfig(
                num_episodes=8,
                num_ppo_epochs=1,
                max_prompt_length=4,
                max_completion_length=4,
                value_hidden_size=4,
                report_to="none",
                learning_rate=0.0002,
                entropy_coef=0.0,
            ),
            model=model,
            tokenizer=tokenizer,
            reward_fn=lambda *_: 1.0,
        )
        assert isinstance(trainer.policy_optimizer, ModelAdamW)
        return trainer

    async def update(t):
        await t.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])

    orig = build()
    await update(orig)
    assert all(torch.isfinite(p).all() for p in orig.model.parameters())
    path = Path(orig.save_checkpoint(str(tmp_path / "new")))
    opt = orig.policy_optimizer
    assert any(
        (master != source.float()).any() for source, master in opt.masters.items()
    )
    if legacy:

        def old_slots(state, unused):
            for slots in state["components"]["policy_optimizer"]["state"].values():
                slots.pop("master_weight")
                for name in ("exp_avg", "exp_avg_sq"):
                    slots[name] = slots[name].to(dtype)

        old = tmp_path / "legacy"
        republish(path, old, old_slots)
        path = old
        # A legacy run can only continue from the moments and weights it stored.
        assert orig.load_runtime_checkpoint(str(path))
    resumed = build()
    assert resumed.load_runtime_checkpoint(str(path))
    for _ in range(3):
        await update(orig)
        await update(resumed)
        assert_same(snapshot(resumed), snapshot(orig))
        assert_same(resumed.model.state_dict(), orig.model.state_dict())


def test_different_decay_witness_exceeds_learning_rate():
    p = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = model_adamw([p], lr=1, betas=(0.0, 0.5), eps=0.1, weight_decay=0)
    for g in (0, 1):
        p.grad = p.new_tensor(g)
        opt.step()
    assert p.item() == pytest.approx(-1 / ((2 / 3) ** 0.5 + 0.1))
    assert p.item() < -1


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("amp", [False, True])
def test_grpo_partial_window_uses_working_gradients_for_unscale_and_clip(
    tmp_path, kind, amp
):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer
    from stateset_agents.training.trainer_utils import safe_optimizer_step
    from tests.unit.test_partial_accumulation_mean import make_trainer

    cls = SingleTurnGRPOTrainer if kind == "single" else MultiTurnGRPOTrainer
    t = make_trainer(tmp_path, cls, amp=amp, max_norm=1.0)
    t.agent.model.half()
    t.config.adam_beta1, t.config.adam_beta2 = 0.75, 0.5
    t._setup_optimizer()
    t.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        t.optimizer, step_size=1, gamma=0.5
    )
    p = t.agent.model.weight
    master = t.optimizer.masters[p]
    assert t.optimizer.param_groups[0]["betas"] == (0.75, 0.5)
    if amp:
        # Initialize the scaler's device, as in a real autocast backward pass.
        t.scaler.scale(p.sum()).backward()
        p.grad = torch.full_like(p, float("inf"))
        assert not safe_optimizer_step(t, torch, max_grad_norm=1, scaler=t.scaler)
        assert (
            t.global_step == 0
            and not t.optimizer.state
            and t.lr_scheduler.last_epoch == 0
        )
    scale = t.scaler.get_scale() if amp else 1
    p.grad = p.new_tensor([[3 * scale, -4 * scale]])
    assert safe_optimizer_step(
        t, torch, max_grad_norm=1, scaler=t.scaler, gradient_scale=2
    )
    g = torch.tensor([[6, -8]]) / (10 + 1e-6)
    torch.testing.assert_close(t.optimizer.state[master]["exp_avg"], 0.25 * g)
    torch.testing.assert_close(
        t.optimizer.state[master]["exp_avg_sq"], 0.5 * g.square()
    )
    assert p.grad is None and master.grad is None
    assert t.global_step == t.lr_scheduler.last_epoch == 1


@pytest.mark.parametrize("bad", ["negative", "maximum"])
def test_checkpoint_rejects_invalid_second_moment_before_loading(bad):
    from stateset_agents.training.ddp_checkpoint_state import _optimizer_stage

    p = torch.nn.Parameter(torch.ones(3, dtype=torch.float16))
    opt = model_adamw([p], lr=0.001, amsgrad=True)
    p.grad = torch.ones_like(p)
    opt.step()
    saved = opt.state_dict()
    slots = {k: v.clone() for k, v in saved["state"][0].items()}
    if bad == "negative":
        slots["exp_avg_sq"][1] = -1
        match = "nonnegative"
    else:
        slots["max_exp_avg_sq"][2] = 0
        match = "maximum"
    with pytest.raises(ValueError, match=match):
        _optimizer_stage(opt, {**saved, "state": {0: slots}}, torch)
    assert (opt.state[opt.masters[p]]["exp_avg_sq"] > 0).all()


def check_first_moments(optimizer):
    assert isinstance(optimizer, ModelAdamW)
    seen = 0
    b1, b2 = optimizer.param_groups[0]["betas"]
    for p, master in optimizer.masters.items():
        if p.grad is None:
            assert master not in optimizer.state
            continue
        state = optimizer.state[master]
        torch.testing.assert_close(state["exp_avg"], (1 - b1) * p.grad.float())
        torch.testing.assert_close(
            state["exp_avg_sq"], (1 - b2) * p.grad.float().square()
        )
        assert torch.isfinite(p).all()
        torch.testing.assert_close(
            p, state["master_weight"].to(p.dtype), rtol=0, atol=0
        )
        seen += 1
    assert seen > 0


@pytest.mark.parametrize("token", [False, True])
async def test_gspo_low_precision_updates_honor_configured_moments(monkeypatch, token):
    from tests.unit.test_gspo_update_integrity import (
        GSPOTokenTrainer,
        GSPOTrainer,
        _train,
        build_trainer,
    )

    trainer = build_trainer(
        GSPOTokenTrainer if token else GSPOTrainer,
        monkeypatch,
        dtype=torch.float16,
        adam_beta1=0.75,
        adam_beta2=0.5,
        weight_decay=0.125,
    )
    await _train(trainer, ["good"])
    assert trainer.optimizer.param_groups[0]["betas"] == (0.75, 0.5)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.125
    check_first_moments(trainer.optimizer)


async def test_gepo_low_precision_public_step(monkeypatch):
    from tests.unit.test_gepo_trainer_behavioral import (
        _fixed_group_responses,
        _tiny_gepo_trainer,
    )

    trainer = _tiny_gepo_trainer(model_dtype=torch.float16)
    rows = _fixed_group_responses()

    async def generate(prompt, group_size):
        return rows

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    await trainer.train_step(["p"])
    check_first_moments(trainer.optimizer)


async def test_dapo_low_precision_public_step(monkeypatch):
    from tests.unit.test_dapo_regularized_batch import (
        DAPOConfig,
        DAPOTrainer,
        TablePolicy,
        fixed_samples,
    )

    model = TablePolicy().half()
    trainer = DAPOTrainer(
        DAPOConfig(num_gradient_updates=1, max_completion_length=5, use_vllm=False),
        model,
        None,
        lambda *_: 0,
    )
    rows = fixed_samples(model)

    async def collect(prompts, target_batch_size):
        return rows, 0.0

    monkeypatch.setattr(trainer, "collect_samples_with_dynamic_sampling", collect)
    await trainer.train_step(["p", "q", "empty"])
    check_first_moments(trainer.optimizer)


async def test_vapo_low_precision_joint_step(monkeypatch):
    from tests.unit.test_vapo_joint_update import DATA, make_trainer, prepare

    trainer, reference = make_trainer(model_dtype=torch.float16, num_gradient_updates=1)
    _, rows = prepare(trainer, reference)

    async def generate(prompt):
        return rows[prompt]

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    await trainer.train_step(list(DATA))
    check_first_moments(trainer.actor_optimizer)
    assert trainer.critic_optimizer.state


def test_initially_frozen_parameter_starts_and_restores_its_own_clock():
    from stateset_agents.training.ddp_checkpoint_state import _optimizer_stage

    p = torch.nn.Parameter(torch.ones(8, dtype=torch.float16), requires_grad=False)
    opt = model_adamw([p], lr=0.0002)
    master = opt.masters[p]
    assert master.numel() == 0
    opt.step()
    assert not opt.state
    p.requires_grad_(True)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.isfinite(p).all() and master.shape == p.shape
    assert opt.state[master]["step"].item() == 1

    replica = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
    restored = model_adamw([replica], lr=0.0002)
    staged = _optimizer_stage(restored, opt.state_dict(), torch)
    assert restored.masters[replica].numel() == 0
    restored.load_state_dict(staged.state_dict())
    replica.requires_grad_(True)
    replica.grad = torch.ones_like(replica)
    p.grad = torch.ones_like(p)
    # PyTorch state_dict can share storage. Give each branch its own state.
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt.step()
    restored.step()
    torch.testing.assert_close(restored.masters[replica], master, rtol=0, atol=0)
    assert restored.state[restored.masters[replica]]["step"].item() == 2


def test_scheduled_adamw_matches_full_history_discounted_sum():
    from math import prod

    p = torch.nn.Parameter(torch.ones((), dtype=torch.float16))
    other = torch.nn.Parameter(torch.zeros(()))
    opt = model_adamw(
        [p, other], lr=0.125, weight_decay=0.25, betas=(0.75, 0.5), eps=0.125
    )
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    moments = []
    terms = []
    for g in (None, 1.0, -0.25, None, 0.0, 0.5, -1.0):
        rate = Decimal.from_float(opt.param_groups[0]["lr"])
        p.grad = None if g is None else p.new_tensor(g)
        other.grad = torch.ones_like(other)
        if g is not None:
            moments.append(Decimal.from_float(g))
            contribution = -closed_update(
                moments,
                Decimal(0),
                rate,
                Decimal(0),
                Decimal("0.75"),
                Decimal("0.5"),
                Decimal("0.125"),
            )
            terms.append((1 - rate * Decimal("0.25"), contribution))
        else:
            terms.append((Decimal(1), Decimal(0)))
        opt.step()
        schedule.step()
        expected = prod(a for a, _ in terms) - sum(
            u * prod(a for a, _ in terms[i + 1 :]) for i, (_, u) in enumerate(terms)
        )
        master = opt.masters[p]
        assert master.item() == pytest.approx(float(expected), abs=1.5e-7)
        assert p.item() == p.new_tensor(float(expected)).item()
        assert schedule.last_epoch == len(terms)
        assert opt.state.get(master, {}).get("step", 0) == len(moments)


def test_added_half_group_retains_its_own_controls_and_step_count():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    q = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt = model_adamw([p], lr=0.1, weight_decay=0)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.add_param_group({"params": (x for x in [q]), "lr": 0.001, "betas": (0.5, 0.75)})
    q.grad = torch.tensor([0, 0.5], dtype=q.dtype)
    opt.step()
    state = opt.state[opt.masters[q]]
    torch.testing.assert_close(state["exp_avg"], torch.tensor([0, 0.25]))
    assert opt.state[opt.masters[p]]["step"].item() == 2 and state["step"].item() == 1
    assert q[0].item() == 1 and state["master_weight"][1].item() == pytest.approx(0.999)
    with pytest.raises(ValueError, match="more than one"):
        opt.add_param_group({"params": [p]})
    assert len(opt.param_groups) == len(opt.masters) == 2
