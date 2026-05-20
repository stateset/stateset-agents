"""09 — Curate an SFT dataset from graded transcripts (chat → grade → curate).

The closure of the §11.7 loop: take production-style transcripts, score each
with the same rubric your trainer uses, and emit a curated JSONL of the
high-scoring examples ready for the next SFT run. This is the local analogue
of `notebooks/grade_and_curate_demo.ipynb` — no GPU, no model downloads.

Install:
    pip install stateset-agents

Run:
    python 09_curate_dataset.py

Expected output:
    Loaded 8 raw transcripts.
    Scored: 8/8.
    Threshold ≥ 0.5: 4 kept, 4 dropped.
    Wrote curated dataset → ./outputs/curated_sft.jsonl
"""

import asyncio
import json
from pathlib import Path

from stateset_agents.core.reward_base import RewardFunction
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data import SupportRewardComposite, load_support_scenarios


# Eight simulated transcripts — half are good, half are poor. In production
# this is a stream of real (query, response) pairs from your serving layer
# (see `examples/api_client_example.py` for how to capture them).
def synth_transcripts() -> list[dict]:
    scenarios = load_support_scenarios()
    out: list[dict] = []
    for s in scenarios[:4]:
        # "good" responses — acknowledge + concrete next step
        ctx = s.to_scenario()
        ack = " and ".join(ctx.get("must_acknowledge", [])) or ctx.get("intent", "your request")
        out.append({
            "query": ctx["user_query"],
            "response": f"Thanks for flagging this {ack}. I'll process the {ctx['intent']} now and confirm in your email.",
            "context": ctx,
        })
    for s in scenarios[4:8]:
        # "bad" responses — generic apology, no acknowledgement
        ctx = s.to_scenario()
        out.append({
            "query": ctx["user_query"],
            "response": "Sorry, I can't help with that. Try again later.",
            "context": ctx,
        })
    return out


async def grade(transcript: dict, rubric: RewardFunction) -> float:
    turns = [ConversationTurn(role="assistant", content=transcript["response"])]
    result = await rubric.compute_reward(turns, context=transcript["context"])
    return result.score


async def main() -> None:
    transcripts = synth_transcripts()
    print(f"Loaded {len(transcripts)} raw transcripts.")

    rubric = SupportRewardComposite()
    scored = []
    for t in transcripts:
        score = await grade(t, rubric)
        scored.append({**t, "score": score})
    print(f"Scored: {len(scored)}/{len(transcripts)}.")

    threshold = 0.5
    kept = [s for s in scored if s["score"] >= threshold]
    dropped = [s for s in scored if s["score"] < threshold]
    print(f"Threshold ≥ {threshold}: {len(kept)} kept, {len(dropped)} dropped.")

    out_dir = Path("./outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "curated_sft.jsonl"
    with out_path.open("w") as f:
        for row in kept:
            # SFT-ready JSONL: a single (instruction, output) pair per line.
            f.write(json.dumps({
                "instruction": row["query"],
                "output": row["response"],
                "score": row["score"],
                "intent": row["context"].get("intent"),
            }) + "\n")
    print(f"Wrote curated dataset → {out_path}")
    print()
    print("Next step: feed `curated_sft.jsonl` into your SFT trainer of choice,")
    print("then re-train your RL adapter on top. See `notebooks/sft_from_curated_demo.ipynb`.")


if __name__ == "__main__":
    asyncio.run(main())
