"""A behavioural golden over every ``RiverExecutor.submit`` mode.

River drives the training loop *in process*: ``submit()`` opens a session,
creates a model, samples, trains and saves — there is no payload to inspect
and no job to poll. So the only way to pin what a mode does is to record what
it asks River to do. That is what this file is: eighteen scenarios, each
recording

* every SDK call in order, with every keyword argument
  (``create_model`` / ``sample`` / ``train_step`` / ``forward_backward`` /
  ``optim_step`` / ``save_weights`` / session open+close),
* the final :class:`JobStatus`,
* the complete ``logs()`` list,
* ``job_cost()``,
* every artifact file written, with its contents,
* the ledger line,

and comparing it to ``data/river_submit_golden.json``. Read that file as the
specification of what each mode does; read a diff against it as the exact
behaviour a change alters.

Regenerate deliberately, and review the diff::

    STATESET_REGEN_RIVER_GOLDEN=1 pytest tests/unit/test_river_submit_golden.py

Everything runs offline against fakes. ``river_client.renderers`` — which the
multi-turn *episode* paths import unconditionally — is stubbed here; without
that stub those two paths cannot execute at all, which is why they had no
coverage before this file existed.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.river import RiverExecutor

GOLDEN_PATH = Path(__file__).parent / "data" / "river_submit_golden.json"
REGEN_ENV = "STATESET_REGEN_RIVER_GOLDEN"

#: Stand-ins for values that change every run.
DURATION = "<duration>"
TIMESTAMP = "<timestamp>"
TMPDIR = "<tmp>"


# -- the SDK stub the episode paths need -----------------------------------


class _StubTokenizer:
    """Deterministic, content-sensitive ids — enough to prove the RL datum
    layout carries the prompt ids the sampler was given."""

    def encode(self, text: str) -> list[int]:
        return [len(text) % 97 + 1, sum(map(ord, text[:8])) % 89 + 1]


class _StubRenderer:
    tokenizer = _StubTokenizer()

    def build_sample_prompt(self, messages: list[dict[str, str]]) -> Any:
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return types.SimpleNamespace(prompt=text)

    def get_stop_strings(self) -> list[str]:
        return ["<|end|>"]


@pytest.fixture(autouse=True)
def stub_river_renderers(monkeypatch):
    """Install ``river_client.renderers`` for the duration of a test.

    ``_rollout_episodes`` and ``_rl_sample_groups`` import it directly. Kept
    in ``sys.modules`` via monkeypatch so it is torn down again and cannot
    leak into the rest of the suite.
    """
    river_client = types.ModuleType("river_client")
    renderers = types.ModuleType("river_client.renderers")
    renderers.get_renderer = lambda base_model, thinking=False: _StubRenderer()
    river_client.renderers = renderers
    monkeypatch.setitem(sys.modules, "river_client", river_client)
    monkeypatch.setitem(sys.modules, "river_client.renderers", renderers)
    return renderers


# -- recording fakes -------------------------------------------------------


@dataclass
class FakeLoraConfig:
    rank: int
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False


@dataclass
class FakeCheckpoint:
    path: str
    step: int = 0
    checkpoint_type: str = "inference"


@dataclass
class FakeSample:
    text: str
    tokens: list[int] = field(default_factory=lambda: [11, 12])
    logprobs: list[float] = field(default_factory=lambda: [-0.5, -0.25])
    prompt_token_ids: list[int] = field(default_factory=lambda: [7, 8, 9])


class RecordingModel:
    """Answers River's whole model surface and records every call."""

    def __init__(self, calls, base_model, lora, checkpoint, texts):
        self._calls = calls
        self.base_model = base_model
        self.lora = lora
        self.checkpoint = checkpoint
        self._texts = texts
        self._served = 0

    def sample(
        self,
        prompts=None,
        *,
        prompt_token_ids=None,
        num_samples=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=300,
        stop=None,
        **kwargs,
    ):
        self._calls.append(
            {
                "call": "sample",
                "prompts": list(prompts) if prompts is not None else None,
                "prompt_token_ids": prompt_token_ids,
                "num_samples": num_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        count = len(prompts) if prompts is not None else len(prompt_token_ids)
        groups = []
        for _ in range(count):
            group = []
            for _ in range(num_samples):
                group.append(FakeSample(self._texts[self._served % len(self._texts)]))
                self._served += 1
            groups.append(group)
        return groups

    def forward_backward(self, batch, loss_fn="cross_entropy"):
        self._calls.append(
            {"call": "forward_backward", "batch_size": len(batch), "loss_fn": loss_fn}
        )
        return {"loss": 0.5, "num_tokens": sum(len(d["input_ids"]) for d in batch)}

    def optim_step(self, **kwargs):
        self._calls.append({"call": "optim_step", **kwargs})
        return {"ok": True}

    def save_weights(self, name, mode="inference"):
        self._calls.append({"call": "save_weights", "name": name, "mode": mode})
        return f"river://checkpoints/{name}"


class TrainStepModel(RecordingModel):
    """The preferred surface: forward+backward+optimiser in one server call."""

    def train_step(self, data, lr, loss_fn="cross_entropy", **kwargs):
        self._calls.append(
            {
                "call": "train_step",
                "lr": lr,
                "loss_fn": loss_fn,
                "datums": data,
            }
        )
        return {"loss_mean": 0.1}, {"ok": True}


class RecordingSession:
    def __init__(self, calls, model_cls, texts):
        self._calls = calls
        self._model_cls = model_cls
        self._texts = texts

    def create_model(self, base_model, lora=None, checkpoint=None):
        self._calls.append(
            {
                "call": "create_model",
                "base_model": base_model,
                "lora": repr(lora),
                "checkpoint": repr(checkpoint),
            }
        )
        return self._model_cls(self._calls, base_model, lora, checkpoint, self._texts)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._calls.append({"call": "session_close"})
        return False


class RecordingClient:
    """Stand-in for ``river_client.Client`` that records the call sequence."""

    LoraConfig = FakeLoraConfig
    Checkpoint = FakeCheckpoint

    def __init__(self, *, texts=("done vpn profile", "oops nope"), model_cls=None):
        self.calls: list[dict[str, Any]] = []
        self._texts = list(texts)
        self._model_cls = model_cls or TrainStepModel

    def session(self, project=None):
        self.calls.append({"call": "session", "project": project})
        return RecordingSession(self.calls, self._model_cls, self._texts)


class ExplodingClient(RecordingClient):
    """Fails at session open — the shortest path to each mode's error handling."""

    message = "kaboom"

    def session(self, project=None):
        raise RuntimeError(self.message)


class UnfundedClient(ExplodingClient):
    """River's live 402 envelope, which every mode should name rather than wrap."""

    message = "Billing: insufficient_funds"


class FakeTokenizer:
    """The SFT batcher's tokenizer seam (River takes ids, so *we* tokenize)."""

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if not tokenize:
            return text
        return self.encode(text)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def encode(self, text, **kwargs):
        return [ord(c) % 50 + 1 for c in text][:32]


# -- scenarios -------------------------------------------------------------
#
# Each builder returns (spec, client). The name says which submit path it
# lands in: `sft` -> _train, `harvest`/`episode_harvest` -> _submit_harvest /
# _submit_episode_harvest, `rl`/`episode_rl` -> _submit_rl /
# _submit_episode_rl (the episode paths are reached by the shape of row 0 of
# the dataset, not by a spec field).

SINGLE_TURN_PROMPTS = [{"prompt": "fix my vpn", "expect": ["vpn profile"]}]
EPISODE_SCRIPTS = [
    {
        "turns": ["hi", "and then?"],
        "turn_expect": [["done"], ["vpn"]],
        "forbid": ["oops"],
    }
]


def _write(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _chat_dataset(tmp_path: Path, rows: int = 4) -> Path:
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": f"answer {i}"},
                    ]
                }
            )
            for i in range(rows)
        )
        + "\n"
    )
    return path


def _spec(tmp_path: Path, **overrides) -> RemoteJobSpec:
    defaults: dict[str, Any] = {
        "base_model": "Qwen/Qwen3.5-9B",
        "output_dir": tmp_path / "out",
    }
    defaults.update(overrides)
    return RemoteJobSpec(**defaults)


def scenario_sft(tmp_path):
    """Supervised fine-tune: epochs x batches of `train_step`, then save."""
    spec = _spec(
        tmp_path,
        dataset=_chat_dataset(tmp_path),
        num_epochs=2,
        lora_r=16,
        per_device_batch_size=2,
        eval_prompts=[{"prompt": "eval me", "expect": ["done"]}],
    )
    return spec, RecordingClient(texts=["done at last"])


def scenario_sft_without_train_step(tmp_path):
    """Older SDK vintage: forward_backward + optim_step, one pair per batch."""
    spec = _spec(
        tmp_path,
        dataset=_chat_dataset(tmp_path, rows=2),
        num_epochs=1,
        per_device_batch_size=1,
    )
    return spec, RecordingClient(model_cls=RecordingModel)


def scenario_sft_episode_eval(tmp_path):
    """SFT whose eval set is multi-turn: scored with the episode scorer."""
    spec = _spec(
        tmp_path,
        dataset=_chat_dataset(tmp_path, rows=1),
        num_epochs=1,
        lora_r=8,
        eval_prompts=[
            {
                "turns": ["t1", "t2"],
                "turn_expect": [["done"], ["vpn"]],
                "forbid": ["oops"],
            }
        ],
    )
    return spec, RecordingClient(texts=["done vpn profile"])


def scenario_sft_dry_run(tmp_path):
    """Dry run stops after batching: River is never contacted."""
    spec = _spec(tmp_path, dataset=_chat_dataset(tmp_path), num_epochs=3, dry_run=True)
    return spec, RecordingClient()


def scenario_sft_empty_dataset(tmp_path):
    """No trainable rows: fail before opening a session."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    return _spec(tmp_path, dataset=path), RecordingClient()


def scenario_harvest(tmp_path):
    """Best-of-N sampling, keep what passes its own checks."""
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", SINGLE_TURN_PROMPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        harvest={"best_of": 4, "temperature": 0.9},
        eval_prompts=[{"prompt": "eval me", "expect": ["done"]}],
    )
    texts = [
        "done, and eval passes",
        "re-provisioned your vpn profile",
        "no idea",
        "vpn profile reset",
        "sorry",
    ]
    return spec, RecordingClient(texts=texts)


def scenario_harvest_dry_run(tmp_path):
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", SINGLE_TURN_PROMPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        dry_run=True,
        harvest={"best_of": 4},
    )
    return spec, RecordingClient()


def scenario_harvest_from_pointer(tmp_path):
    """Second flywheel generation: sample from the previous river:// weights."""
    pointer_dir = tmp_path / "gen1"
    pointer_dir.mkdir()
    _write(
        pointer_dir / "river_checkpoint.json",
        {"checkpoint": "river://abc/sampler_weights/gen1"},
    )
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", SINGLE_TURN_PROMPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        harvest={"adapter_dir": str(pointer_dir), "best_of": 2},
    )
    return spec, RecordingClient(texts=["vpn profile ok"])


def scenario_harvest_string_best_of(tmp_path):
    """`best_of` arriving as a string (it comes from a free-form dict) must
    still land in the summary as a number, the same as the episode mode."""
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", SINGLE_TURN_PROMPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        harvest={"best_of": "2"},
    )
    return spec, RecordingClient(texts=["vpn profile ok"])


def scenario_episode_harvest(tmp_path):
    """Multi-turn harvest: whole conversations, episode-level pass."""
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", EPISODE_SCRIPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        harvest={"best_of": 2, "temperature": 0.9},
        eval_prompts=[{"turns": ["e1"], "turn_expect": [["done"]]}],
    )
    return spec, RecordingClient(texts=["done vpn profile", "oops nope"])


def scenario_episode_harvest_dry_run(tmp_path):
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", EPISODE_SCRIPTS),
        output_dir=tmp_path / "harvest",
        job_kind="harvest",
        dry_run=True,
        harvest={"best_of": 3},
    )
    return spec, RecordingClient()


def scenario_rl(tmp_path):
    """GRPO-shaped rounds: sample a group, grade it, train on the spread."""
    spec = _spec(
        tmp_path,
        dataset=_write(
            tmp_path / "p.json",
            [{"prompt": "fix it", "expect": ["done"], "forbid": ["oops"]}],
        ),
        output_dir=tmp_path / "rl",
        job_kind="rl",
        harvest={"best_of": 2, "rounds": 2, "loss_fn": "cispo"},
        eval_prompts=[{"prompt": "check me", "expect": ["done"]}],
    )
    return spec, RecordingClient(texts=["done fine", "oops nope"])


def scenario_rl_zero_variance(tmp_path):
    """Every sample scores the same: nothing to learn, so nothing is trained."""
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", [{"prompt": "fix it", "expect": ["done"]}]),
        output_dir=tmp_path / "rl",
        job_kind="rl",
        harvest={"best_of": 2, "rounds": 1},
    )
    return spec, RecordingClient(texts=["done"])


def scenario_rl_dry_run(tmp_path):
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", [{"prompt": "fix it", "expect": ["done"]}]),
        output_dir=tmp_path / "rl",
        job_kind="rl",
        dry_run=True,
        harvest={"rounds": 3},
    )
    return spec, RecordingClient()


def scenario_episode_rl(tmp_path):
    """Multi-turn RL: one episode advantage broadcast across its turns."""
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", EPISODE_SCRIPTS),
        output_dir=tmp_path / "rl",
        job_kind="rl",
        harvest={"best_of": 2, "rounds": 2, "loss_fn": "cispo"},
        eval_prompts=[{"turns": ["e1"], "turn_expect": [["done"]]}],
    )
    return spec, RecordingClient(texts=["done vpn profile", "oops nope"])


def scenario_episode_rl_dry_run(tmp_path):
    spec = _spec(
        tmp_path,
        dataset=_write(tmp_path / "p.json", EPISODE_SCRIPTS),
        output_dir=tmp_path / "rl",
        job_kind="rl",
        dry_run=True,
        harvest={"rounds": 5},
    )
    return spec, RecordingClient()


def _failure_scenario(job_kind, dataset_payload, client_cls):
    def build(tmp_path):
        spec = _spec(
            tmp_path,
            dataset=_write(tmp_path / "p.json", dataset_payload),
            output_dir=tmp_path / "out",
            job_kind=job_kind,
        )
        return spec, client_cls()

    return build


def _sft_failure_scenario(client_cls):
    def build(tmp_path):
        spec = _spec(tmp_path, dataset=_chat_dataset(tmp_path, rows=1), num_epochs=1)
        return spec, client_cls()

    return build


#: name -> builder. Ordering is the reading order of the golden file.
SCENARIOS = {
    "sft": scenario_sft,
    "sft_without_train_step": scenario_sft_without_train_step,
    "sft_episode_eval": scenario_sft_episode_eval,
    "sft_dry_run": scenario_sft_dry_run,
    "sft_empty_dataset": scenario_sft_empty_dataset,
    "harvest": scenario_harvest,
    "harvest_dry_run": scenario_harvest_dry_run,
    "harvest_from_pointer": scenario_harvest_from_pointer,
    "harvest_string_best_of": scenario_harvest_string_best_of,
    "episode_harvest": scenario_episode_harvest,
    "episode_harvest_dry_run": scenario_episode_harvest_dry_run,
    "rl": scenario_rl,
    "rl_zero_variance": scenario_rl_zero_variance,
    "rl_dry_run": scenario_rl_dry_run,
    "episode_rl": scenario_episode_rl,
    "episode_rl_dry_run": scenario_episode_rl_dry_run,
    # Failure handling, per mode: an unknown SDK error is wrapped with the
    # mode's own label; an account-state error is named instead.
    "fail_sft": _sft_failure_scenario(ExplodingClient),
    "fail_harvest": _failure_scenario("harvest", SINGLE_TURN_PROMPTS, ExplodingClient),
    "fail_episode_harvest": _failure_scenario(
        "harvest", EPISODE_SCRIPTS, ExplodingClient
    ),
    "fail_rl": _failure_scenario("rl", SINGLE_TURN_PROMPTS, ExplodingClient),
    "fail_episode_rl": _failure_scenario("rl", EPISODE_SCRIPTS, ExplodingClient),
    "unfunded_sft": _sft_failure_scenario(UnfundedClient),
    "unfunded_harvest": _failure_scenario(
        "harvest", SINGLE_TURN_PROMPTS, UnfundedClient
    ),
    "unfunded_episode_harvest": _failure_scenario(
        "harvest", EPISODE_SCRIPTS, UnfundedClient
    ),
    "unfunded_rl": _failure_scenario("rl", SINGLE_TURN_PROMPTS, UnfundedClient),
    "unfunded_episode_rl": _failure_scenario("rl", EPISODE_SCRIPTS, UnfundedClient),
}


# -- recording -------------------------------------------------------------


def _scrub(value: Any, tmp_root: str) -> Any:
    """Replace anything that changes between runs, so the golden is stable."""
    if isinstance(value, dict):
        return {k: _scrub(v, tmp_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, tmp_root) for v in value]
    if isinstance(value, str):
        return value.replace(tmp_root, TMPDIR)
    return value


def _ledger_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        entry = json.loads(raw)
        entry["recorded_at"] = TIMESTAMP
        if entry.get("duration_s") is not None:
            entry["duration_s"] = DURATION
        lines.append(entry)
    return lines


def _artifacts(root: Path, ledger: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != ledger:
            out[path.relative_to(root).as_posix()] = path.read_text()
    return out


def record_scenario(name: str, tmp_path: Path) -> dict[str, Any]:
    """Run one scenario and return everything worth pinning about it."""
    spec, client = SCENARIOS[name](tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    executor = RiverExecutor(
        client=client, tokenizer=FakeTokenizer(), ledger_path=ledger
    )
    executor._sleep = lambda seconds: None  # no real backoff in tests

    outcome: dict[str, Any] = {}
    try:
        handle = executor.submit(spec)
    except Exception as exc:  # noqa: BLE001 - the failure IS the behaviour here
        outcome["raised"] = f"{type(exc).__name__}: {exc}"
    else:
        duration, cost = executor.job_cost(handle)
        outcome["status"] = executor.status(handle).value
        outcome["logs"] = list(executor.logs(handle))
        outcome["cost_usd"] = cost
        outcome["duration_recorded"] = duration is not None

    recorded = {
        "outcome": outcome,
        "sdk_calls": client.calls,
        "artifacts": _artifacts(tmp_path, ledger),
        "ledger": _ledger_lines(ledger),
    }
    return _scrub(recorded, str(tmp_path))


def _load_golden() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    return _load_golden()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_submit_matches_the_recorded_behaviour(name, tmp_path, golden):
    """Every mode does exactly what ``river_submit_golden.json`` says.

    A failure here is not necessarily a bug — it is a behaviour change. Read
    the diff, decide whether it was intended, and regenerate if it was.
    """
    if os.environ.get(REGEN_ENV):
        pytest.skip("regenerating the golden")
    assert (
        name in golden
    ), f"{name!r} has no golden entry; regenerate with {REGEN_ENV}=1"
    assert record_scenario(name, tmp_path) == golden[name]


@pytest.mark.skipif(not os.environ.get(REGEN_ENV), reason="regeneration is opt-in")
def test_regenerate_the_golden(tmp_path_factory):
    """Rewrite the golden file. Opt-in, and the diff is the review."""
    fresh = {
        name: record_scenario(name, tmp_path_factory.mktemp(name)) for name in SCENARIOS
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")


@pytest.mark.skipif(
    bool(os.environ.get(REGEN_ENV)), reason="the golden is being rewritten"
)
class TestTheGoldenSaysWhatWeThinkItSays:
    """Named claims about the recording, so a silent regeneration cannot
    quietly erase the properties we actually care about."""

    def test_every_mode_is_covered(self, golden):
        assert set(golden) == set(SCENARIOS)

    def test_the_episode_paths_actually_ran(self, golden):
        # Both were unreachable before this file stubbed the renderers.
        assert golden["episode_rl"]["outcome"]["status"] == "succeeded"
        assert golden["episode_harvest"]["outcome"]["status"] == "succeeded"

    def test_dry_runs_never_contact_river(self, golden):
        for name in SCENARIOS:
            if name.endswith("dry_run"):
                assert golden[name]["sdk_calls"] == [], name

    def test_every_run_is_ledgered_with_an_unknown_cost(self, golden):
        for name, entry in golden.items():
            if name.endswith("dry_run") or name == "sft_empty_dataset":
                continue  # never reached the training loop
            assert entry["ledger"], f"{name} was not ledgered"
            line = entry["ledger"][0]
            assert line["cost_usd"] is None, f"{name} invented a dollar figure"
            assert line["gpu"] == "river-managed"

    def test_a_failed_run_is_still_ledgered(self, golden):
        for name, entry in golden.items():
            if not name.startswith(("fail_", "unfunded_")):
                continue
            assert entry["ledger"][0]["status"] == "failed", name

    def test_an_unfunded_account_is_named_not_wrapped(self, golden):
        for name, entry in golden.items():
            if not name.startswith("unfunded_"):
                continue
            raised = entry["outcome"]["raised"]
            assert "insufficient_funds" in raised and "river.ai" in raised, name

    def test_rl_trains_on_the_whole_group_with_the_configured_loss(self, golden):
        steps = [c for c in golden["rl"]["sdk_calls"] if c["call"] == "train_step"]
        assert [s["loss_fn"] for s in steps] == ["cispo", "cispo"]
        assert all(len(s["datums"]) == 2 for s in steps)

    def test_zero_variance_groups_train_nothing(self, golden):
        entry = golden["rl_zero_variance"]
        assert not [c for c in entry["sdk_calls"] if c["call"] == "train_step"]
        assert any("zero-variance" in line for line in entry["outcome"]["logs"])

    def test_episode_rl_broadcasts_one_advantage_across_a_turn(self, golden):
        steps = [
            c for c in golden["episode_rl"]["sdk_calls"] if c["call"] == "train_step"
        ]
        assert steps, "episode RL never trained"
        datum = steps[0]["datums"][0]
        # River's pre-shifted RL contract: prompt positions carry no
        # advantage, response positions all carry the episode's, and the
        # sequence is padded by one at the tail.
        assert len(datum["input_ids"]) == len(datum["attention_mask"])
        assert len(datum["old_logprobs"]) == len(datum["advantages"])
        nonzero = {a for a in datum["advantages"] if a}
        assert len(nonzero) == 1, "an episode's turn should carry ONE advantage"

    def test_greedy_evals_are_deterministic(self, golden):
        for name in ("harvest", "rl", "episode_harvest", "episode_rl"):
            samples = [c for c in golden[name]["sdk_calls"] if c["call"] == "sample"]
            assert samples[0]["temperature"] == 0.0, f"{name} eval was not greedy"
            assert samples[0]["num_samples"] == 1, name

    def test_checkpoints_are_saved_for_inference_only(self, golden):
        for name in ("sft", "rl", "episode_rl"):
            saves = [
                c for c in golden[name]["sdk_calls"] if c["call"] == "save_weights"
            ]
            assert len(saves) == 1, name
            assert saves[0]["mode"] == "inference", name

    def test_harvest_writes_the_pod_contract_artifacts(self, golden):
        for name in ("harvest", "episode_harvest"):
            files = set(golden[name]["artifacts"])
            assert "harvest/harvest.jsonl" in files, name
            assert "harvest/harvest_summary.json" in files, name

    def test_best_of_is_a_number_in_every_summary(self, golden):
        for name in ("harvest", "harvest_string_best_of", "episode_harvest"):
            summary = json.loads(
                golden[name]["artifacts"]["harvest/harvest_summary.json"]
            )
            assert isinstance(summary["best_of"], int), name

    def test_both_harvest_modes_report_the_same_progress_lines(self, golden):
        """The two modes drifted apart: one announced what it sampled from,
        the other announced its eval score, and only one said anything on a
        dry run. All three lines now exist in both."""
        for name in ("harvest", "episode_harvest"):
            logs = golden[name]["outcome"]["logs"]
            assert any("via River" in line for line in logs), name
            assert any("current generation" in line for line in logs), name
        for name in ("harvest_dry_run", "episode_harvest_dry_run"):
            assert golden[name]["outcome"]["logs"] == [
                "dry run: harvest plan only, nothing sampled"
            ], name

    def test_a_previous_generation_pointer_is_resolved_to_its_uri(self, golden):
        created = [
            c
            for c in golden["harvest_from_pointer"]["sdk_calls"]
            if c["call"] == "create_model"
        ]
        assert "river://abc/sampler_weights/gen1" in created[0]["checkpoint"]
