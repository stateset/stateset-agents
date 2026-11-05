"""Compare a wholly handwritten 100k ReLU network with the same network trained.

    python benchmarks/weight_space/handwritten_decoder_job.py --out /tmp/decoder

CPU only by default. The handwritten construction never reads examples or
fits a parameter. Adam baselines get online samples spanning the full 0..95
flip range, BCE loss, cosine learning-rate decay, and validation checkpoint
selection. Two learning rates are tried per seed; only validation chooses
which trained checkpoint reaches the test set. No test metric enters fitting.

Uniform random flips are the in-distribution test. A separate, explicitly
shifted stress test flips 95 positions toward each nearest-distance rival.
It uses the code, not a model's predictions, to choose those corruptions.
This tests one constructed task, architecture and finite optimizer budget;
it cannot establish superiority over every possible training procedure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import platform
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from stateset_agents.experimental.weight_space import handwritten_decoder as hd

LOGGER = logging.getLogger(__name__)
ERROR_LEVELS = (0, 16, 32, 48, 64, 80, 95)


def sample_batch(
    code: torch.Tensor,
    n: int,
    generator: torch.Generator,
    errors: int | None = None,
    balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly choose distinct flip positions and messages, on CPU."""
    ids = (
        torch.arange(n) % hd.HIDDEN
        if balanced
        else torch.randint(hd.HIDDEN, (n,), generator=generator)
    )
    counts = (
        torch.randint(hd.MAX_ERRORS + 1, (n,), generator=generator)
        if errors is None
        else torch.full((n,), errors)
    )
    received = code[ids].clone()
    maximum = int(counts.max())
    if maximum:
        positions = (
            torch.rand(n, hd.CODE_BITS, generator=generator)
            .topk(maximum, dim=1)
            .indices
        )
        flips = torch.arange(maximum)[None, :] < counts[:, None]
        received.scatter_(
            1, positions, received.gather(1, positions) * (1 - 2 * flips.float())
        )
    return received, ids


def fixed_suite(
    code: torch.Tensor, n: int, seed: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Balanced, independent noise draws for each predeclared error count."""
    generator = torch.Generator().manual_seed(seed)
    return {
        f"flips_{e}": sample_batch(code, n, generator, e, balanced=True)
        for e in ERROR_LEVELS
    }


def boundary_suite(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All 65,024 ordered pairs at distance 192, moved 95 flips toward the rival."""
    ids, rivals = torch.where((code @ code.T) == 0)
    received = code[ids].clone()
    differences = code[ids] != code[rivals]
    # A fixed, model-independent subset: the first 95 differing channel positions.
    flips = differences & (differences.long().cumsum(1) <= hd.MAX_ERRORS)
    received *= 1 - 2 * flips.float()
    return received, ids


@torch.inference_mode()
def evaluate(model: torch.nn.Module, suite: dict, batch_size: int = 1024) -> dict:
    """Count whole-message errors, bit errors, erasures, and BCE loss."""
    model.eval()
    bits = hd.message_bits()
    result = {}
    for label, (received, targets) in suite.items():
        correct = bit_correct = erasures = 0
        loss = 0.0
        for start in range(0, len(targets), batch_size):
            ids = targets[start : start + batch_size]
            logits = model(received[start : start + batch_size])
            predicted = hd.read_messages(logits)
            correct += int((predicted == ids).sum())
            erasures += int((predicted < 0).sum())
            bit_correct += int(((logits > 0) == bits[ids].bool()).sum())
            loss += float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, bits[ids], reduction="sum"
                )
            )
        n = len(targets)
        result[label] = {
            "n": n,
            "correct": correct,
            "message_accuracy": correct / n,
            "bit_accuracy": bit_correct / (8 * n),
            "erasures": erasures,
            "bce": loss / (8 * n),
        }
    return result


def validation_key(scores: dict) -> tuple[float, float]:
    """Select whole-message accuracy first, then lower BCE; never use test scores."""
    return (
        sum(row["correct"] for row in scores.values())
        / sum(row["n"] for row in scores.values()),
        -sum(row["bce"] for row in scores.values()) / len(scores),
    )


def train_one(
    code, validation, *, seed, lr, steps, batch_size, eval_every
) -> tuple[dict, dict]:
    """One independently initialized optimizer run with a fixed sample stream."""
    model = hd.make_decoder(handwritten=False, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(10_000 + seed)
    bits = hd.message_bits()
    best_key = (-1.0, -math.inf)
    best_state, best_step = None, 0
    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            scores = evaluate(model, validation)
            key = validation_key(scores)
            history.append(
                {
                    "step": step,
                    "validation_message_accuracy": key[0],
                    "validation_bce": -key[1],
                }
            )
            if key > best_key:
                best_key, best_step = key, step
                best_state = copy.deepcopy(model.state_dict())
            LOGGER.info("seed=%d lr=%g step=%d validation=%.4f", seed, lr, step, key[0])
        if step == steps:
            break
        model.train()
        received, ids = sample_batch(code, batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(received), bits[ids]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = lr * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step + 1) / steps))
            )
    return best_state, {
        "seed": seed,
        "initial_lr": lr,
        "steps": steps,
        "batch_size": batch_size,
        "training_draws": steps * batch_size,
        "selected_step": best_step,
        "validation_message_accuracy": best_key[0],
        "validation_bce": -best_key[1],
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def sha256(path: Path) -> str:
    """Hash an exact source or artifact file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def verify_single_flips(model: torch.nn.Module, code: torch.Tensor) -> dict:
    """Exhaust all 256 messages times 384 possible single-bit corruptions."""
    targets = torch.arange(hd.HIDDEN)
    correct = 0
    for position in range(hd.CODE_BITS):
        received = code.clone()
        received[:, position] *= -1
        correct += int((hd.read_messages(model(received)) == targets).sum())
    return {"n": hd.HIDDEN * hd.CODE_BITS, "correct": correct}


def main(argv: list[str] | None = None) -> None:
    """Write weights, run the predeclared comparison, and retain all trial summaries."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--test-per-level", type=int, default=4096)
    ap.add_argument("--validation-per-level", type=int, default=512)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    for name in (
        "steps",
        "batch_size",
        "eval_every",
        "test_per_level",
        "validation_per_level",
        "threads",
    ):
        if getattr(a, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if any(not math.isfinite(lr) or lr <= 0 for lr in a.learning_rates):
        ap.error("learning rates must be finite and positive")
    if len(set(a.seeds)) != len(a.seeds):
        ap.error("seeds must be distinct")
    if a.test_per_level % hd.HIDDEN or a.validation_per_level % hd.HIDDEN:
        ap.error("per-level counts must be multiples of 256 for message balance")
    a.out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    code = hd.make_codebook()
    started = time.perf_counter()
    handwritten = hd.make_decoder()
    construct_seconds = time.perf_counter() - started
    weight_path = a.out / "handwritten.safetensors"
    save_file(
        {k: v.contiguous() for k, v in handwritten.state_dict().items()}, weight_path
    )
    config = {
        **{k: v for k, v in vars(a).items() if k != "out"},
        "architecture": [384, 256, 8],
        "activation": "ReLU",
        "parameters": sum(p.numel() for p in handwritten.parameters()),
        "nonzero_parameters": sum(
            int(p.count_nonzero()) for p in handwritten.parameters()
        ),
        "training_noise": "uniform integer flip count 0..95; uniform distinct positions",
        "training_messages": "uniform 0..255, online draws with replacement",
        "validation_seed": 70000,
        "test_seed": 80000,
        "error_levels": ERROR_LEVELS,
        "selection": "best validation exact-message accuracy, then lowest validation BCE",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "source_sha256": {
            "model": sha256(Path(hd.__file__)),
            "benchmark": sha256(Path(__file__)),
        },
    }
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    validation = fixed_suite(code, a.validation_per_level, 70000)
    result = {
        "config": config,
        "handwritten": {
            "construction_seconds": construct_seconds,
            "optimizer_steps": 0,
            "training_examples": 0,
            "weights_sha256": sha256(weight_path),
        },
        "trials": [],
        "selected": [],
    }

    def save_progress() -> None:
        (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save_progress()
    for seed in a.seeds:
        chosen_state, chosen_trial = None, None
        for lr in a.learning_rates:
            state, trial = train_one(
                code,
                validation,
                seed=seed,
                lr=lr,
                steps=a.steps,
                batch_size=a.batch_size,
                eval_every=a.eval_every,
            )
            result["trials"].append(trial)
            if chosen_trial is None or (
                trial["validation_message_accuracy"],
                -trial["validation_bce"],
            ) > (
                chosen_trial["validation_message_accuracy"],
                -chosen_trial["validation_bce"],
            ):
                chosen_state, chosen_trial = state, trial
            save_progress()
        trained_path = a.out / f"trained_seed{seed}.safetensors"
        save_file({k: v.contiguous() for k, v in chosen_state.items()}, trained_path)
        result["selected"].append(
            {
                "seed": seed,
                "lr": chosen_trial["initial_lr"],
                "step": chosen_trial["selected_step"],
                "weights_sha256": sha256(trained_path),
            }
        )
        save_progress()

    # No test input or metric is constructed before every training choice is sealed.
    from safetensors.torch import load_file

    tests = fixed_suite(code, a.test_per_level, 80000)
    tests["boundary_95_shifted"] = boundary_suite(code)
    result["handwritten"]["test"] = evaluate(handwritten, tests)
    distances = (hd.CODE_BITS - code @ code.T) / 2
    distances.fill_diagonal_(hd.CODE_BITS)
    result["certificate"] = {
        "codewords": len(code),
        "minimum_pairwise_distance": int(distances.min()),
        "guaranteed_radius": hd.MAX_ERRORS,
        "true_hidden_lower_bound": 192 - 2 * hd.MAX_ERRORS,
        "other_hidden_upper_bound": 0,
        "scope": "all binary corruptions of at most 95 distinct positions; analytical bound",
        "exhaustive_single_flip_test": verify_single_flips(handwritten, code),
    }
    for selected in result["selected"]:
        model = hd.make_decoder(handwritten=False)
        model.load_state_dict(
            load_file(a.out / f"trained_seed{selected['seed']}.safetensors")
        )
        selected["test"] = evaluate(model, tests)
        save_progress()
    save_progress()
    LOGGER.info("Complete: %s", a.out / "summary.json")


if __name__ == "__main__":
    main()
