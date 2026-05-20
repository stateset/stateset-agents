# Testing patterns for StateSet Agents

Five reusable pytest patterns, picked to cover the questions that come up
when you actually start writing tests for an agent system: *does the reward
function do what I think?*, *does the agent loop wire up correctly without
a GPU?*, *do my contracts hold across many inputs?*, *does my environment
terminate?*, *is my judge stable?*

Each file is a single `pytest -k` target. Together they take <5 seconds on
CPU. No model downloads, no GPU. Drop them into your own project's `tests/`
directory and adapt — they're the same patterns that gate the framework's
own CI.

| File | What it tests | Pattern |
|------|---------------|---------|
| [`test_custom_reward.py`](./test_custom_reward.py) | A custom `RewardFunction` against handcrafted (input, expected score) pairs. | Table-driven |
| [`test_stub_integration.py`](./test_stub_integration.py) | The full agent → env → reward wiring end-to-end with the stub backend. | Integration (no mocks) |
| [`test_hypothesis_properties.py`](./test_hypothesis_properties.py) | Invariants that must hold for any valid input — rewards in `[0, 1]`, turn counts non-negative. | Property-based |
| [`test_env_smoke.py`](./test_env_smoke.py) | Every bundled scenario can be reset, stepped, and reaches a terminal state. | Smoke |
| [`test_judge_stability.py`](./test_judge_stability.py) | An LLM-judge's score variance across N runs is below a stability floor. | Stability budget |

## Run them

```bash
pip install "stateset-agents[testing]"  # adds hypothesis, pytest-asyncio
cd examples/testing
pytest -q
```

Expected: all green in under 5 seconds.

## When to reach for which pattern

- **Reward changed?** → `test_custom_reward.py`. Pin the (input, expected score) table; let it gate the diff.
- **New environment / new agent subclass?** → `test_stub_integration.py`. Cheapest possible end-to-end check that wiring is right.
- **Reward needs to be robust to noisy or adversarial inputs?** → `test_hypothesis_properties.py`. Hypothesis generates 100s of inputs per run.
- **Adding scenarios to your corpus?** → `test_env_smoke.py`. Catches malformed JSONL before it hits a training run.
- **About to publish a judge-based number?** → `test_judge_stability.py`. Bounds the noise floor of your eval.

## Anti-patterns we explicitly avoid

- **Mocking the model.** Use `AgentConfig(use_stub_model=True)` instead — `StubModel`/`StubTokenizer` exercise the real adapter wiring whereas `MagicMock` just confirms your test agrees with itself.
- **Mocking the database.** If your reward reads from an external source, use a `tmp_path` fixture and write a fake JSONL. (See [`feedback_mocking_anti_patterns.md`](../../docs/COOKBOOK.md) for the framework's own incident write-up.)
- **`time.sleep` in async tests.** Use `asyncio.wait_for` or pytest-asyncio's `event_loop` fixture. Sleeps mask flakiness; timeouts surface it.
- **Asserting on free-form text.** Score with a reward, then assert on the score. Stub-backed text is deterministic but not meaningful.

## See also

- [`TESTING.md`](../../TESTING.md) — the framework's own test architecture
- [`stateset_agents/testing/`](../../stateset_agents/testing/) — bundled fixtures, hypothesis strategies, matchers
- [`tests/integration/test_stub_training_loop.py`](../../tests/integration/test_stub_training_loop.py) — a real, full training-loop integration test
