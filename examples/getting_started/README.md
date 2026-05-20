# Getting started — runnable examples after `pip install`

Ten small, self-contained scripts. Each one starts with a comment block
listing the exact `pip install` it needs and the expected output. Read them
in order; each builds on the previous.

| # | File | What it shows | GPU? | Extras |
|---|---|---|---|---|
| 01 | [`01_hello_stub.py`](./01_hello_stub.py) | Smallest possible "did this install work?" — instantiate `MultiTurnAgent` with the stub backend, generate a deterministic response. | No | core |
| 02 | [`02_custom_reward.py`](./02_custom_reward.py) | Define a custom `RewardFunction` subclass, score three responses. No training, no GPU. | No | core |
| 03 | [`03_first_finetune.py`](./03_first_finetune.py) | Your first real GSPO fine-tune. Small base (Qwen2.5-0.5B-Instruct), 16 train scenarios, safe-default config from whitepaper §B.1. | **A100** | `[training]` |
| 04 | [`04_llm_judge_eval.py`](./04_llm_judge_eval.py) | LLM-as-judge eval pattern from §11.7 — load a 1.5B instruction-tuned model as a judge, score (query, intent, response) triples. `--stub` flag for GPU-free smoke testing. | A100 (or `--stub`) | `[training]` |
| 05 | [`05_serve_agent.py`](./05_serve_agent.py) | Wire an agent into the OpenAI-compatible FastAPI service. Hit `/v1/chat/completions` with curl or the OpenAI Python SDK. | No (stub-backed) | `[api]` |
| 06 | [`06_multi_turn_episode.py`](./06_multi_turn_episode.py) | Drive a multi-turn episode through `ConversationEnvironment` — `reset()` → loop `step()` → terminal reward. The same loop a GSPO trainer runs internally. | No | core |
| 07 | [`07_tool_calling.py`](./07_tool_calling.py) | `ToolAgent` + bundled `ToolCallReward` — score well-formed, wrong-tool, and malformed responses against the function-calling rubric. | No | core |
| 08 | [`08_eval_driven_loop.py`](./08_eval_driven_loop.py) | The §11.7 development rhythm: pick a rubric, score baseline, change one thing, measure. No GPU — two pure-Python policies stand in for two agent checkpoints. | No | core |
| 09 | [`09_curate_dataset.py`](./09_curate_dataset.py) | Close the chat → grade → curate loop: score 8 synthetic transcripts with the support rubric, write an SFT-ready JSONL of the high-scoring ones. | No | core |
| 10 | [`10_scenario_testing.py`](./10_scenario_testing.py) | Regression-style assertions: must-acknowledge, must-avoid, rubric floors. Exits non-zero on the first regression — pin into your CI. | No | core |

## Common path

```bash
# 1. Verify install works (no GPU):
pip install stateset-agents
python 01_hello_stub.py
python 02_custom_reward.py

# 2. First real training (needs a CUDA host with at least 16 GB VRAM):
pip install "stateset-agents[training]"
python 03_first_finetune.py

# 3. Reusable evaluation primitive:
python 04_llm_judge_eval.py --stub          # smoke without GPU
python 04_llm_judge_eval.py                 # real judge on CUDA

# 4. Production serving:
pip install "stateset-agents[api]"
python 05_serve_agent.py                    # terminal 1
curl -X POST http://localhost:8001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "stub", "messages": [{"role": "user", "content": "hello"}]}'
# (terminal 2)

# 5. The rest of the loop — multi-turn, tools, eval-driven dev, curation, CI:
python 06_multi_turn_episode.py
python 07_tool_calling.py
python 08_eval_driven_loop.py
python 09_curate_dataset.py
python 10_scenario_testing.py     # exits non-zero on assertion regression
```

## Which version?

These examples target **stateset-agents ≥ 0.13.4**. `pip install stateset-agents` from
[PyPI](https://pypi.org/project/stateset-agents/) gets you the current published
version; check `stateset_agents.__version__` from example 01 to confirm.

## After these five

When you've run all five and they make sense, the natural next steps are:

- **`notebooks/customer_support_3seed_judge.ipynb`** — the canonical §11.7
  whitepaper protocol (three seeds, dual eval). Run in Colab.
- **`docs/COOKBOOK.md`** — eight self-contained recipes for common workflows
  (iterating from production logs, debugging a stuck reward, etc.).
- **`docs/WHITEPAPER.md`** — the technical reference. §11.7 is the canonical
  empirical claim; §10.5 is the most important operational caveat.

## Reporting issues

If any of these examples breaks on a fresh `pip install`, please file an
issue at <https://github.com/stateset/stateset-agents/issues> with:

- Output of `pip show stateset-agents`
- The command you ran
- The full traceback
- Your OS + Python version

The CI smoke (`make notebook-lint` + `scripts/lint_notebooks.py`) catches
the eight foot-gun patterns from [issue #16](https://github.com/stateset/stateset-agents/issues/16);
new failure modes are how we learn what to add next.
