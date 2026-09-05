"""Capture loss goldens for every native trainer's objective assembly.

Run BEFORE migrating a trainer to ``objectives.py`` and commit the JSON; the
paired test (``tests/unit/test_objective_goldens.py``) asserts the migrated
trainer reproduces these numbers.

    .venv/bin/python scripts/capture_objective_goldens.py [--only dapo,...]

Deterministic: fixed seeds, CPU, dropout-free tiny GPT-2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

OUT = ROOT / "tests" / "unit" / "goldens" / "objective_goldens.json"


def tiny_model(vocab: int = 200):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=vocab,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def tensors(seed: int, n: int = 4, t: int = 10):
    g = torch.Generator().manual_seed(seed)
    cur = (-torch.rand(n, t, generator=g) * 3).requires_grad_(True)
    old = cur.detach() + 0.05 * torch.randn(n, t, generator=g)
    ref = cur.detach() + 0.1 * torch.randn(n, t, generator=g)
    mask = torch.ones(n, t)
    mask[:, :3] = 0
    mask[0, 8:] = 0
    adv = torch.tensor([1.0, -0.5, 0.25, -0.75])
    return cur, old, ref, mask, adv


def golden_dapo() -> dict:
    from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

    out = {}
    for token_level in (True, False):
        cfg = DAPOConfig(
            model_name="gpt2", group_size=4, use_token_level_loss=token_level
        )
        tr = DAPOTrainer(
            config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 0.0
        )
        cur, old, _, mask, adv = tensors(1)
        ratios = tr.compute_importance_ratio(cur, old)
        loss = tr.compute_dapo_loss(ratios, adv.unsqueeze(1).expand_as(ratios), mask)
        out[f"token_level={token_level}"] = loss.item()
    return out


def golden_vapo() -> dict:
    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    out = {}
    for token_level in (True, False):
        cfg = VAPOConfig(
            model_name="gpt2",
            group_size=4,
            use_token_level_loss=token_level,
            per_device_train_batch_size=4,
        )
        tr = VAPOTrainer(
            config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 1.0
        )
        cur, old, _, mask, _ = tensors(2)
        g = torch.Generator().manual_seed(22)
        pol_adv = torch.randn(4, 10, generator=g)
        crit_adv = torch.randn(4, 10, generator=g)
        values = torch.randn(4, 10, generator=g)
        old_values = values + 0.1 * torch.randn(4, 10, generator=g)
        positive = torch.zeros(4, 10)
        positive[0] = mask[0]
        p, v, lm = tr.compute_vapo_losses(
            cur, old, pol_adv, crit_adv, values, old_values, mask, positive
        )
        out[f"token_level={token_level}"] = {
            "policy": p.item(),
            "value": v.item(),
            "positive_lm": lm.item(),
        }
    return out


def _gspo_trainer(cls, beta: float):
    from stateset_agents.training.gspo_config import GSPOConfig

    cfg = GSPOConfig(model_name="gpt2", num_generations=4, beta=beta)
    return cls(
        config=cfg,
        model=tiny_model(),
        tokenizer=None,
        agent=None,
        environment=None,
        reward_model=None,
        ref_model=None,
    )


def golden_gspo() -> dict:
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    out = {}
    for beta in (0.0, 0.05):
        tr = _gspo_trainer(GSPOTrainer, beta)
        cur_tok, old_tok, ref_tok, mask, adv = tensors(3)
        lengths = mask.sum(-1).clamp(min=1.0)
        cur = (cur_tok * mask).sum(-1)
        old = (old_tok * mask).sum(-1)
        ref = (ref_tok * mask).sum(-1)
        ratios = tr.compute_sequence_importance_ratio(cur, old, lengths)
        loss = tr.compute_gspo_loss(
            ratios, adv, cur, lengths, ref if beta > 0 else None
        )
        out[f"beta={beta}"] = loss.item()
    return out


def golden_gspo_token() -> dict:
    """GSPO-token: pin the GRADIENT w.r.t. token log-probs, not the loss value.

    The migrated objective reports the surrogate value (same quantity GSPO
    reports) instead of the log-prob-weighted quantity; gradients are
    identical, so that is what the golden pins.
    """
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer

    out = {}
    for beta in (0.0, 0.05):
        tr = _gspo_trainer(GSPOTokenTrainer, beta)
        cur_tok, old_tok, ref_tok, _, adv = tensors(4)
        # Trainer convention: each gathered row runs prompt (zeros) then the
        # response through the END of the row (no trailing padding), so give
        # row 0 a longer prompt instead of trailing pad positions.
        mask = torch.ones_like(cur_tok)
        mask[:, :3] = 0
        mask[0, :5] = 0
        lengths = mask.sum(-1).clamp(min=1.0)
        # Leaf BEFORE masking, as in the trainer (gather_token_logprobs masks
        # the gathered rows), so prompt/pad positions carry no gradient.
        leaf = cur_tok.detach().clone().requires_grad_(True)
        rows = leaf * mask
        cur = rows.sum(-1)
        old = (old_tok * mask).sum(-1)
        ref = (ref_tok * mask).sum(-1)
        ratios = tr.compute_sequence_importance_ratio(cur, old, lengths).detach()
        # Make one sample sit outside the trust region on its advantage's side.
        ratios = ratios.clone()
        ratios[1] = 1.5
        token_lists = [rows[i] for i in range(4)]
        loss = tr.compute_gspo_token_loss(
            token_lists, lengths, ratios, adv, cur, ref if beta > 0 else None
        )
        loss.backward()
        out[f"beta={beta}"] = leaf.grad.tolist()
    return out


def golden_gepo() -> dict:
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    cfg = GEPOConfig(model_name="gpt2", group_size=4)
    tr = GEPOTrainer(
        config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 0.0
    )
    cur_tok, old_tok, _, mask, adv = tensors(5)
    cur = (cur_tok * mask).sum(-1)
    old = (old_tok * mask).sum(-1)
    return {"default": tr.compute_gepo_loss(cur, old, adv).item()}


def golden_ppo() -> dict:
    from stateset_agents.training.ppo_trainer import PPOConfig, PPOTrainer

    cfg = PPOConfig(model_name="gpt2")
    tr = PPOTrainer(
        config=cfg, model=tiny_model(), tokenizer=None, reward_fn=lambda p, r: 0.0
    )
    cur, old, ref, mask, _ = tensors(6)
    g = torch.Generator().manual_seed(66)
    adv = torch.randn(4, 10, generator=g)
    loss, clip_fraction = tr.ppo_loss(cur, old, adv, mask)
    kl = tr.compute_kl_divergence(cur, ref, mask)
    return {
        "policy": loss.item(),
        "clip_fraction": clip_fraction.item(),
        "kl": kl.item(),
    }


def golden_grpo() -> dict:
    from test_loss_computation_behavioral import _config, _make_agent, _make_group

    from stateset_agents.training import loss_computation as lc

    out = {}
    cases = {
        "reinforce": {"group": _make_group(8, 4), "cfg": _config(clip_ratio=0.0)},
        "clipped_inside": {
            "group": _make_group(8, 4, log_probs=-(0.5 + 1e-4) * 4),
            "cfg": _config(clip_ratio=0.2, seq_clip_ratio=3e-4),
        },
        "clipped_outside": {
            "group": _make_group(8, 4, log_probs=-(0.5 + 1e-3) * 4),
            "cfg": _config(clip_ratio=0.2, seq_clip_ratio=3e-4),
        },
    }
    for name, case in cases.items():
        agent = _make_agent(8, 4)
        loss, _ = lc._compute_group_policy_loss(
            case["group"], torch.tensor([1.0]), case["cfg"], agent
        )
        loss.backward()
        out[name] = {"loss": loss.item(), "grad": float(agent.model.p.grad)}
    return out


def golden_grpo_enhanced() -> dict:
    """Real tiny GPT-2 through compute_enhanced_grpo_loss (plain + exact KL)."""
    from stateset_agents.training import loss_computation as lc

    class _Tok:
        def apply_chat_template(
            self,
            messages,
            *,
            return_dict=False,
            return_assistant_tokens_mask=False,
            **kw,
        ):
            ids = torch.tensor([[5, 7, 9, 11, 13, 17, 19, 23, 29, 31]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "assistant_tokens_mask": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
            }

    def traj(reward: float, lp_sum: float | None):
        t = SimpleNamespace(
            turns=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            total_reward=reward,
            metadata={},
        )
        if lp_sum is not None:
            t.log_probs = lp_sum
        return t

    model = tiny_model()
    ref = tiny_model()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.01)
    agent = SimpleNamespace(tokenizer=_Tok(), model=model)
    cfg = SimpleNamespace(
        max_prompt_length=32,
        max_completion_length=32,
        clip_ratio=0.2,
        seq_clip_ratio=3e-4,
        bf16=False,
        fp16=False,
    )
    out = {}
    for beta, use_ref in ((0.0, False), (0.05, True)):
        groups = [
            SimpleNamespace(
                trajectories=[traj(1.0, -20.0), traj(0.0, -21.0), traj(0.5, None)]
            )
        ]
        model.zero_grad()
        res = lc.compute_enhanced_grpo_loss(
            groups, beta, cfg, agent, reference_model=ref if use_ref else None
        )
        res["total_loss"].backward()
        grad_norm = torch.sqrt(
            sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
        ).item()
        out[f"beta={beta}"] = {
            "total": res["total_loss"].item(),
            "policy": res["policy_loss"].item(),
            "kl": float(res["kl_penalty"].detach()),
            "grad_norm": grad_norm,
        }
    return out


CAPTURES = {
    "dapo": golden_dapo,
    "vapo": golden_vapo,
    "gspo": golden_gspo,
    "gspo_token": golden_gspo_token,
    "gepo": golden_gepo,
    "ppo": golden_ppo,
    "grpo": golden_grpo,
    "grpo_enhanced": golden_grpo_enhanced,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default="", help="comma-separated subset to (re)capture"
    )
    args = parser.parse_args()
    names = [n for n in args.only.split(",") if n] or list(CAPTURES)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    torch.use_deterministic_algorithms(True)
    for name in names:
        existing[name] = CAPTURES[name]()
        print(f"captured {name}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
