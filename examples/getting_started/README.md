# Getting started — runnable examples after `pip install`

Five small, self-contained scripts. Each one starts with a comment block
listing the exact `pip install` it needs and the expected output. Read them
in order; each builds on the previous.

| # | File | What it shows | GPU? | Extras |
|---|---|---|---|---|
| 01 | [`01_hello_stub.py`](./01_hello_stub.py) | Smallest possible "did this install work?" — instantiate `MultiTurnAgent` with the stub backend, generate a deterministic response. | No | core |
| 02 | [`02_custom_reward.py`](./02_custom_reward.py) | Define a custom `RewardFunction` subclass, score three responses. No training, no GPU. | No | core |
| 03 | [`03_first_finetune.py`](./03_first_finetune.py) | Your first real GSPO fine-tune. Small base (Qwen2.5-0.5B-Instruct), 16 train scenarios, safe-default config from whitepaper §B.1. | **A100** | `[training]` |
| 04 | [`04_llm_judge_eval.py`](./04_llm_judge_eval.py) | LLM-as-judge eval pattern from §11.7 — load a 1.5B instruction-tuned model as a judge, score (query, intent, response) triples. `--stub` flag for GPU-free smoke testing. | A100 (or `--stub`) | `[training]` |
| 05 | [`05_serve_agent.py`](./05_serve_agent.py) | Wire an agent into the OpenAI-compatible FastAPI service. Hit `/v1/chat/completions` with curl or the OpenAI Python SDK. | No (stub-backed) | `[api]` |

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
