"""04 — LLM-as-judge evaluation (the §11.7 pattern).

Demonstrates the paraphrase-tolerant LLM-judge eval used in the canonical
§11.7 result: load a small instruction-tuned model as a judge, prompt it for
a 0-10 score, parse, average. Works against any agent — trained or baseline,
stub or real.

Install:
    pip install "stateset-agents[training]"

Run:
    # GPU (~3 min, downloads the 1.5B judge model on first run):
    python 04_llm_judge_eval.py

    # CPU smoke (uses stub responses; verifies the eval scaffolding):
    python 04_llm_judge_eval.py --stub

Expected output (GPU mode):
    Judge model loaded.
    Test 1 (good response):  judge_score = 0.7-1.0
    Test 2 (bad response):   judge_score = 0.0-0.3

Use this pattern to evaluate any (query, intent, response) triple. For the
multi-seed three-seed protocol that produces a publication-grade number,
see notebooks/customer_support_3seed_judge.ipynb.
"""

import argparse
import re
import sys


JUDGE_PROMPT = """You are evaluating a customer service agent's response. Rate the response on a 0-10 scale.

Customer message: {query}
Expected intent: {intent}

Agent response: {response}

Rate the response on these criteria:
- Did it address the customer's intent?
- Is the tone appropriate (polite, professional)?
- Did it offer a clear next step or actionable information?

Output ONLY a single integer 0-10. Do not include any other text.

Score:"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true",
                        help="Skip the real model load; just exercise the scaffolding.")
    parser.add_argument("--judge", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model id for the judge.")
    args = parser.parse_args()

    test_cases = [
        ("Test 1 (good response)", "I need a refund for order #9981", "refund",
         "I can process your refund immediately! Please confirm the original "
         "payment method and order ID and I'll get started right away."),
        ("Test 2 (bad response)", "I need a refund for order #9981", "refund",
         "Random hash 12345 #include <stdio.h>"),
    ]

    if args.stub:
        # Smoke: hard-coded scores so the pipeline runs end-to-end without GPU.
        for label, _q, _i, _r in test_cases:
            print(f"{label}: judge_score = 0.85 (stub)" if "good" in label else
                  f"{label}: judge_score = 0.10 (stub)")
        print("\nDone (stub mode). For a real judgment pass --no-stub on a CUDA host.")
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}. Install with: pip install 'stateset-agents[training]'")
        return 1

    if not torch.cuda.is_available():
        print("No CUDA detected — try --stub for a GPU-free smoke test.")
        return 1

    print(f"Loading judge model: {args.judge}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print("Judge model loaded.")

    @torch.no_grad()
    def judge_score(query: str, intent: str, response: str) -> float:
        prompt = JUDGE_PROMPT.format(query=query, intent=intent, response=response[:1024])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\b(10|[0-9])\b", decoded)
        return min(int(m.group(1)), 10) / 10.0 if m else 0.5

    for label, q, i, r in test_cases:
        score = judge_score(q, i, r)
        print(f"{label}: judge_score = {score:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
