# τ-bench retail: prompted vs fine-tuning vs RL

No result has been produced yet. This page describes how to run the study
and what its numbers will and will not mean.

Three arms start from one base model and one untrained adapter:

| Arm | Training signal |
|---|---|
| `prompted` | None |
| `sft` | Imitates assistant turns from the base model's own rollouts that the official checker scored 1.0 |
| `rl` | Policy-gradient updates on the official checker's reward |

The two trained arms draw the same rollouts-per-update schedule over the 74
official train tasks, take the same number of optimizer updates at the same
learning rate, and neither sees the checker's reference actions. All arms are
scored on the 40 official test tasks with identical customer-simulator seeds.
The primary comparison is `rl` minus `sft`.

## Run it

Python 3.12 or newer and a CUDA GPU are required for a real run.

```bash
pip install -e ".[dev,api,training]"
make benchmark-tau-retail-setup
cp benchmarks/tau_retail_three_arm.protocol.draft.json protocol.json   # fill every REPLACE value
make benchmark-tau-retail-preflight PROTOCOL=protocol.json OUTPUT_DIR=out/preflight
make benchmark-tau-retail-run PROTOCOL=protocol.json OUTPUT_DIR=out/run
```

The preflight runs one update and two test tasks with the protocol's model
and writes `preflight.json`: peak memory, rollouts and tokens per second, and
an order-of-magnitude GPU-hour and cost estimate for the full protocol. Use
it to set the budget fields before committing the protocol.

Outputs per seed are `prompted.json`, `sft.json`, `rl.json` (one row per
scored conversation) and `result.json` (budgets, throughput, GPU cost, peak
memory, the `tau2` data digest). `comparison.json` holds the paired analysis.

## Rules the runner enforces

- A protocol with `"measured": true` must pin `tau2_retail_sha256`, price GPU
  time with `gpu_hourly_usd`, name a `cost_source`, and cannot mock the
  customer or judge or run on a task subset.
- The installed `tau2` retail data must match the pinned digest.
- A conversation the official checker fails to score aborts the seed. `tau2`
  itself reports reward 0.0 when its customer-simulator or judge call fails;
  counting that as a failed task would bias every arm.

## What differs from an official τ-bench score

- The agent loop is this repository's text-action loop over `tau2`'s gym
  interface, not `tau2 run` with native tool calling. The customer simulator,
  database, tools, policy and checker are `tau2`'s.
- `environment.judge_llm: null` keeps `tau2`'s pinned judge, which needs an
  OpenAI API key. Any other value replaces the judge and must be reported.
- Reported cost is GPU wall time at the stated hourly rate. It excludes
  customer-simulator and judge API usage.

## Things to know before reading a result

- An agent that does nothing passes some tasks, because refusing is the
  correct outcome there. Compare arms with each other, not with zero.
- The fine-tuning arm uses no human or reference demonstrations. A
  demonstration-trained baseline is a different comparison.
- With `bounded_token_backward` the trainer replays turns during backward to
  bound memory. It has been validated on CPU only.
